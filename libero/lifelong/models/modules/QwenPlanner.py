"""
Qwen-based planner interface for starVLA.

This module provides a lightweight planning/checking interface that follows
starVLA coding style and config conventions.
"""

import re
import logging
import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# System prompts – v1 (strict 7-verb list) and v2 (flexible verbs)
# -----------------------------------------------------------------------

default_system_prompt_plan = (
    "You are an expert household manipulation planner for a robot system. "
    "You will receive two images for every decision: "
    "(1) a main camera image showing the overall scene and the robotic arm, "
    "and (2) a wrist camera image showing a close-up view. "

    "Your job is to convert a high-level task into a sequence of concise, atomic, spatial sub-tasks. "
    "Each sub-task must strictly follow the format: '<Action Verb> <Target Object> <Target Location/State>'. "
    
    "RULES: "
    "1. Limit the sequence to 2-4 steps maximum. Only the most essential milestones. "
    "2. Use ONLY these verbs: Move to, Pick up, Place, Open, Close, Turn on, Turn off. "
    "3. Explicitly state the target object and the target spatial location. "
    "4. Do not assume the robot is holding anything initially. "
    "5. Format strictness: Each step must be a single imperative sentence. "
    "   Example: 'Pick up the red mug from the table.' "
    "   Example: 'Place the mug on the top shelf.' "
    "6. Do not output explanations. "

    "OUTPUT FORMAT: SUBTASK LIST: 1. ... 2. ... 3. ..."
)

# v2: open verb vocabulary – supports Grasp, Push, Slide, Pull, Insert, etc.
default_system_prompt_plan_v2 = (
    "You are an expert household manipulation planner for a robot system. "
    "You will receive two images for every decision: "
    "(1) a main camera image showing the overall scene and the robotic arm, "
    "and (2) a wrist camera image showing a close-up view. "

    "Your job is to convert a high-level task into a sequence of concise, "
    "atomic sub-tasks that a robotic arm can execute one by one. "
    "Each sub-task must be a single imperative sentence with a clear action verb, "
    "a target object, and a destination or target state. "

    "RULES: "
    "1. Limit the sequence to 2-4 steps maximum. Only the most essential milestones. "
    "2. Start each step with a clear, specific action verb "
    "   (e.g. Pick up, Place, Move to, Open, Close, Push, Pull, Slide, "
    "   Grasp, Release, Turn on, Turn off, Insert, Align, Lift, Lower, Rotate). "
    "   Choose the most precise verb that describes the motion. "
    "3. Explicitly state the target object and the target spatial location or state. "
    "4. Do not assume the robot is holding anything initially. "
    "5. Each step must be a single imperative sentence. "
    "   Example: 'Pick up the red mug from the table.' "
    "   Example: 'Slide the plate to the left side of the counter.' "
    "   Example: 'Push the drawer closed.' "
    "6. Do not output explanations. "

    "OUTPUT FORMAT: SUBTASK LIST: 1. ... 2. ... 3. ..."
)

default_system_prompt_check = (
    "You are an expert household manipulation planner. "
    "You will receive two images for every decision: "
    "(1) a main camera image showing the full scene, the table, and the robotic arm from a human-like perspective, "
    "and (2) a wrist camera image showing a close-up, top-down view of the gripper and nearby objects. "
    "Based on the input images, high-level task, current sub-task, completed sub-tasks, and all sub-tasks, "
    "your goal is to decide if the current sub-task has been completed in the images so it can excute the next sub-task "

    "Use YES if there is clear or partial evidence from either image that the sub-task is completed or nearly completed. "
    "Use NO only if both images together strongly show that the sub-task has NOT been completed. "

    "GUIDANCE ON USING THE TWO VIEWS: "
    "• Use the main camera image to understand global arm position, object placement, scene configuration, and high-level progress. "
    "• Use the wrist camera image to confirm fine-grained interactions such as grasping, touching, alignment, insertion, or placement. "
    "• If the two images appear inconsistent, prioritize the wrist camera for fine-grained details and the main camera for spatial context. "
    "• If one view is unclear but the other suggests completion, choose YES. "

    "RULES: "
    "1. Rely on observations from BOTH images; do not ignore either view. "
    "2. Consider reasonable inferences based on object movement, gripper pose, or scene changes. "
    "3. Consider only the current sub-task; ignore future sub-tasks. "
    "4. Do not provide explanations; only output completion. "
    "5. Do not propose new sub-tasks. "

    "OUTPUT FORMAT: "
    "Your output must follow this format exactly: "
    "COMPLETED: YES or NO"
)

def _to_plain_dict(cfg_obj: Any) -> Dict[str, Any]:
    if cfg_obj is None:
        return {}
    if isinstance(cfg_obj, dict):
        return cfg_obj
    if hasattr(cfg_obj, "items"):
        try:
            return {k: v for k, v in cfg_obj.items()}
        except Exception:
            pass
    return {}


def _dtype_from_string(dtype_name: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(dtype_name, torch.dtype):
        return dtype_name
    name = str(dtype_name).lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return mapping.get(name, torch.bfloat16)


class _QwenPlanner_Interface(nn.Module):
    def __init__(
        self,
        config: Optional[dict] = None,
        system_prompt_plan: str = default_system_prompt_plan,
        system_prompt_check: str = default_system_prompt_check,
        model_path: Optional[str] = None,
        device: str = "cuda",
        dtype: Union[str, torch.dtype] = torch.bfloat16,
        device_map: str = "auto",
        attn_implementation: Optional[str] = None,
        prompt_version: str = "v1",
        **kwargs,
    ):
        """
        loading Qwen model and processor

        Args:
            prompt_version: "v1" uses the strict 7-verb prompt (backward compat),
                            "v2" uses the flexible open-vocabulary prompt.
        """
        super().__init__()

        planner_cfg = {}
        if config is not None and hasattr(config, "framework"):
            framework_cfg = config.framework
            planner_cfg = _to_plain_dict(getattr(framework_cfg, "planner", None))
            if not planner_cfg:
                planner_cfg = _to_plain_dict(getattr(framework_cfg, "qwenplanner", None))

        model_id = (
            model_path
            or planner_cfg.get("base_vlm")
            or planner_cfg.get("model_path")
            or "Qwen/Qwen2.5-VL-7B-Instruct"
        )
        dtype = _dtype_from_string(planner_cfg.get("dtype", dtype))
        device_map = planner_cfg.get("device_map", device_map)
        attn_implementation = planner_cfg.get("attn_implementation", attn_implementation)

        logger.info("Initializing planner VLM from %s", model_id)
        if "Qwen3-VL" in model_id:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map=device_map,
                attn_implementation=attn_implementation,
            )
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map=device_map,
                attn_implementation=attn_implementation,
            )

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device = torch.device(device)
        self.config = config

        # Select planning prompt version
        if prompt_version == "v2":
            system_prompt_plan = default_system_prompt_plan_v2
            logger.info("Using v2 planner prompt (open verb vocabulary)")
        self.system_prompt_plan = system_prompt_plan
        self.system_prompt_check = system_prompt_check

    def _extract_subtasks(self, text: str) -> List[str]:
        """
        input Qwen output (numbered list, possibly all on one line or multi-line)
        return list[str]

        Handles two common output formats:
          Multi-line:  "1. step one\n2. step two\n3. step three"
          Single-line: "1. step one 2. step two 3. step three"
        """
        # Non-greedy match; lookahead stops at the next "N." item or end-of-string.
        # re.DOTALL lets "." cross newlines so multi-line blocks are captured too.
        pattern = r"\d+\.\s*(.*?)(?=\s*\d+\.|$)"
        tasks = re.findall(pattern, text, re.DOTALL)
        # strip whitespace / newlines and drop empty entries
        tasks = [t.strip() for t in tasks if t.strip()]
        return tasks
    
    def _prepare_image(self, img: Union[np.ndarray, Image.Image, str]) -> Union[Image.Image, str]:
        """
        support inputs:
        - numpy ndarray (H,W,3)
        - PIL.Image
        - image path / URL
        return: PIL.Image, path or URL
        """
        if isinstance(img, np.ndarray):
            # HWC
            if img.ndim == 3 and img.shape[0] in [1, 3] and img.shape[2] != 3:
                img = np.transpose(img, (1, 2, 0))
            # uint8
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            return Image.fromarray(img)
        elif isinstance(img, Image.Image):
            return img
        elif isinstance(img, str):
            return img
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    def _build_inputs(self, messages: List[dict]):
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, _ = process_vision_info(messages)
        model_inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        return model_inputs.to(self.model.device)

    def _get_visual_token_ids(self) -> List[int]:
        """Best-effort retrieval of visual token ids used in multimodal sequence."""
        ids: List[int] = []

        # Prefer model config ids if available (most reliable).
        for key in ("image_token_id", "video_token_id", "vision_token_id"):
            v = getattr(self.model.config, key, None)
            if v is not None:
                try:
                    ids.append(int(v))
                except Exception:
                    pass

        if ids:
            return sorted(set(ids))

        # Fallback: try known token strings from tokenizer vocab.
        tok = self.processor.tokenizer
        unk_id = getattr(tok, "unk_token_id", None)
        for token_str in (
            "<|image_pad|>",
            "<|video_pad|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<image>",
        ):
            try:
                tid = tok.convert_tokens_to_ids(token_str)
            except Exception:
                tid = None
            if tid is None:
                continue
            try:
                tid = int(tid)
            except Exception:
                continue
            if unk_id is not None and tid == int(unk_id):
                continue
            ids.append(tid)

        return sorted(set(ids))

    @torch.inference_mode()
    def _generate_text(
        self,
        messages: List[dict],
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
    ) -> str:
        inputs = self._build_inputs(messages)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
        else:
            generation_config = self.model.generation_config
            generation_config = copy.deepcopy(generation_config) if generation_config is not None else None
            if generation_config is not None:
                generation_config.temperature = None
                generation_config.top_p = None
                generation_config.top_k = None
                gen_kwargs["generation_config"] = generation_config

        generated_ids = self.model.generate(
            **inputs,
            **gen_kwargs,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text

    @torch.inference_mode()
    def get_subtasks(
        self,
        high_task: str,
        image_list: Sequence[Union[np.ndarray, Image.Image, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        do_sample: bool = True,
    ) -> List[str]:
        """
        generate subtasks and return a subtask list
            high_task: high level task
            image_list: a list containing multi view images
        """
        #messages
        messages = [
            {
                "role": "system",
                "content": self.system_prompt_plan
            },
            {
                "role": "user",
                "content": []
            }
        ]
        messages[1]["content"].append({"type": "text", "text": "high-level task: " + high_task})
        for img in image_list:
            prepared_img = self._prepare_image(img)
            messages[1]["content"].append({"type": "image", "image": prepared_img})

        output_text = self._generate_text(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
        subtask_list = self._extract_subtasks(output_text)

        return subtask_list

    @torch.inference_mode()
    def check_subtask(
        self,
        high_task: str,
        image_list: Sequence[Union[np.ndarray, Image.Image, str]],
        current_subtask: str,
        all_subtasks: Sequence[str],
        finished_subtasks: Sequence[str],
        max_new_tokens: int = 64,
        temperature: float = 0.1,
        do_sample: bool = False,
        return_text: bool = False,
    ) -> Union[bool, Tuple[bool, str]]:
        """
        check whether the current subtask is completed
        return: completed(bool), output_text(str)
        """
        # System prompt: use system_prompt_check
        messages =[
            {
                "role": "system",
                "content": self.system_prompt_check
            },
            {
                "role": "user",
                "content": []
            },
        ]

        # Build text input
        user_text = (
            f"High-level task: {high_task}\n"
            f"Current sub-task: {current_subtask}\n"
            f"Current all sub-tasks: {all_subtasks}\n"
            f"Completed sub-tasks: {finished_subtasks}"
        )
        messages[1]["content"].append({"type": "text", "text": user_text})

        # Add images
        for img in image_list:
            prepared_img = self._prepare_image(img)
            messages[1]["content"].append({"type": "image", "image": prepared_img})

        output_text = self._generate_text(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
        logger.info("Subtask check output: %s", output_text)

        # Parse completion
        completed = False
        if "YES" in output_text.upper():
            completed = True
        elif "NO" in output_text.upper():
            completed = False

        if return_text:
            return completed, output_text
        return completed

    @torch.inference_mode()
    def encode_env_state(
        self,
        image_list: Sequence[Union[np.ndarray, Image.Image, str]],
        prompt: str = "Observe the current environment state.",
        max_length: int = 512,
    ) -> np.ndarray:
        """
        Encode multi-view environment observation into one vector `s_env`
        using PURE visual-token pooling.

        Returns:
            np.ndarray with shape [hidden_size], dtype float32
        """
        _ = prompt  # keep API compatibility; pure-vision path ignores text prompt.
        messages = [
            {
                "role": "system",
                "content": "",
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": ""}],
            },
        ]
        for img in image_list:
            prepared_img = self._prepare_image(img)
            messages[1]["content"].append({"type": "image", "image": prepared_img})

        model_inputs = self._build_inputs(messages)

        # Optional safety trim for very long contexts.
        if "input_ids" in model_inputs and model_inputs["input_ids"].shape[1] > max_length:
            model_inputs["input_ids"] = model_inputs["input_ids"][:, -max_length:]
            if "attention_mask" in model_inputs:
                model_inputs["attention_mask"] = model_inputs["attention_mask"][:, -max_length:]

        outputs = self.model(
            **model_inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]  # [1, T, D]

        input_ids = model_inputs.get("input_ids", None)
        visual_token_ids = self._get_visual_token_ids()
        visual_mask = None
        if input_ids is not None and len(visual_token_ids) > 0:
            visual_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for vid in visual_token_ids:
                visual_mask |= (input_ids == vid)
            if "attention_mask" in model_inputs:
                visual_mask &= model_inputs["attention_mask"].bool()

        # Pure visual pooling. If detection fails, fallback to valid-token pooling.
        if visual_mask is not None and bool(visual_mask.any().item()):
            mask = visual_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        elif "attention_mask" in model_inputs:
            logger.warning("No visual token mask found; fallback to attention-mask pooling.")
            mask = model_inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            logger.warning("No visual token mask and no attention mask; fallback to mean pooling.")
            pooled = hidden.mean(dim=1)

        return pooled[0].detach().float().cpu().numpy()




# Backward-compatible name
QwenPlanner = _QwenPlanner_Interface


