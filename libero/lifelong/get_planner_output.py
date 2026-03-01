"""
Minimal script: get planner output in LIBERO environment.

Run from the LIBERO-VLA repo root:
  python libero/lifelong/get_planner_output.py \
      --model_path Qwen/Qwen3-VL-4B-Instruct

Or from anywhere:
  python /path/to/LIBERO-VLA/libero/lifelong/get_planner_output.py \
      --model_path Qwen/Qwen3-VL-4B-Instruct
"""

import argparse
import importlib.util
import os
import sys
import gc
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

# ---- resolve the LIBERO-VLA repo root ---------------------------------
# File layout: <repo_root>/libero/lifelong/get_planner_output.py
#              <repo_root>/libero/lifelong/models/modules/QwenPlanner.py
_REPO_ROOT = Path(__file__).resolve().parents[2]  # LIBERO-VLA/
_QWEN_PLANNER_FILE = (
    _REPO_ROOT / "libero" / "lifelong" / "models" / "modules" / "QwenPlanner.py"
)

# Load QwenPlanner directly from the local file to avoid name collision with
# the `libero` package installed in site-packages.
_spec = importlib.util.spec_from_file_location("QwenPlanner", _QWEN_PLANNER_FILE)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
QwenPlanner = _module.QwenPlanner

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def build_env_and_task(task_suite_name: str, task_id: int, seed: int):
    task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
    task = task_suite.get_task(task_id)
    task_description = task.language

    task_bddl_file = (
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(seed)

    return env, task_suite, task, task_description, task_bddl_file


def _extract_paren_block(text: str, anchor: str) -> str:
    """Extract the full parenthesised block starting at *anchor* in *text*."""
    start = text.find(anchor)
    if start < 0:
        return f"({anchor.strip()} block not found)"

    depth = 0
    end = None
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end is None:
        return text[start:].strip()
    return text[start:end].strip()


def extract_goal_block(bddl_path: Path) -> str:
    if not bddl_path.exists():
        return "(bddl file not found)"
    text = bddl_path.read_text(encoding="utf-8", errors="ignore")
    return _extract_paren_block(text, "(:goal")


def extract_language_instruction(bddl_path: Path) -> str:
    """Return the official task instruction stored in (:language ...) block."""
    if not bddl_path.exists():
        return "(bddl file not found)"
    text = bddl_path.read_text(encoding="utf-8", errors="ignore")
    anchor = "(:language"
    start = text.find(anchor)
    if start < 0:
        return "(language block not found)"
    # (:language <instruction text>) – may span one line, no nested parens
    block = _extract_paren_block(text, anchor)
    # strip the keyword and outer parens: (:language <text>)
    inner = block[len("(:language"):].rstrip(")")
    return inner.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suite_name", type=str, default="libero_goal")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    env, task_suite, task, task_description, task_bddl_file = build_env_and_task(
        task_suite_name=args.task_suite_name,
        task_id=args.task_id,
        seed=args.seed,
    )
    planner = None
    try:
        initial_states = task_suite.get_task_init_states(args.task_id)
        env.reset()
        obs = env.set_init_state(initial_states[args.episode_idx])

        # align with eval_libero preprocessing (rotate 180°)
        main_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

        planner = QwenPlanner(
            model_path=args.model_path,
            device=args.device,
        )

        subtasks = planner.get_subtasks(
            high_task=task_description,
            image_list=[main_img, wrist_img],
            max_new_tokens=256,
            temperature=0.2,
            do_sample=False,
        )

        print("\n=== LIBERO task metadata ===")
        print(f"suite={args.task_suite_name}")
        print(f"task_id={args.task_id}")
        print(f"problem_folder={task.problem_folder}")
        print(f"bddl_file={task.bddl_file}")
        print(f"bddl_path={task_bddl_file}")

        print("\n=== Official BDDL language instruction ===")
        print(extract_language_instruction(task_bddl_file))

        print("\n=== High-level task (task.language) ===")
        print(task_description)

        print("\n=== BDDL goal (ground-truth symbolic goal) ===")
        print(extract_goal_block(task_bddl_file))

        print("\n=== Planner subtasks ===")
        if len(subtasks) == 0:
            print("(empty output)")
        else:
            for i, st in enumerate(subtasks, 1):
                print(f"{i}. {st}")

        if len(subtasks) > 0:
            completed, raw_text = planner.check_subtask(
                high_task=task_description,
                image_list=[main_img, wrist_img],
                current_subtask=subtasks[0],
                all_subtasks=subtasks,
                finished_subtasks=[],
                max_new_tokens=32,
                temperature=0.0,
                do_sample=False,
                return_text=True,
            )
            print("\n=== Check first subtask ===")
            print(f"completed={completed}")
            print(f"raw_output={raw_text}")
    finally:
        try:
            if planner is not None:
                del planner
            env.close()
        finally:
            gc.collect()


if __name__ == "__main__":
    main()
