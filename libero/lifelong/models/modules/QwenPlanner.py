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


default_system_prompt_plan = (
    "You are an expert household manipulation planner for a robot system. "
    "You will receive two images for every decision: "
    "(1) a main camera image showing the overall scene, table, objects, and the robotic arm from a human-like viewpoint, "
    "and (2) a wrist camera image showing a close-up, top-down view of the gripper and nearby objects. "
    
    "Your job is to convert a high-level task into a sequence of concise, atomic sub-tasks that a vision-language-action model can execute. "
    "Each sub-task must describe exactly one physical action and must be observable and feasible for a robot arm in a real environment. "
    "RULES: "
    "1. Use both the input images and the high-level task to decide the needed actions. "
    "2. Break down the task into small, sequential, physical steps. "
    "3. Never combine two actions into one. One step equals one action. "
    "4. Do not assume the robot is already holding any object. "
    "5. Use only simple, physical verbs such as: move to the object or location; pick up the object; place the object on, in, or next to a target; open the container; close the container; turn on or turn off an appliance. "
    "6. Keep each step short, concrete, and unambiguous. "
    "7. Do not output explanations. Only output the sub-task list. "
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
        **kwargs,
    ):
        """
        loading Qwen model and processor
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

        self.system_prompt_plan = system_prompt_plan
        self.system_prompt_check = system_prompt_check

    def _extract_subtasks(self, text: str) -> List[str]:
        """
        input Qwen output
        return list[str]
        """
        pattern = r"\d+\.\s*(.+)"
        tasks = re.findall(pattern, text)
        # remove empty lines and empty space
        tasks = [t.strip() for t in tasks if len(t.strip()) > 0]
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




# Backward-compatible name
QwenPlanner = _QwenPlanner_Interface


