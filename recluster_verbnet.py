"""Quick re-cluster using saved embeddings with VerbNet hierarchy.

Reuses embeddings from a previous run (e.g. skill_lib_results_oat/) and
applies VerbNet two-level clustering without needing the heavy planner model.

Usage:
    python recluster_verbnet.py --source_dir skill_lib_results_oat --output_dir skill_lib_results_verbnet
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")

# Load test_skill_lib from local file path directly
_REPO_ROOT = Path(__file__).resolve().parent
_TSL_PATH = _REPO_ROOT / "libero" / "lifelong" / "test_skill_lib.py"
_spec = importlib.util.spec_from_file_location("test_skill_lib", _TSL_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
verbnet_hierarchical_cluster = _mod.verbnet_hierarchical_cluster


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, default="skill_lib_results_oat",
                        help="Directory with saved embeddings + subtask_texts.json")
    parser.add_argument("--output_dir", type=str, default="skill_lib_results_verbnet",
                        help="Output directory for VerbNet clustering results")
    args = parser.parse_args()

    source = Path(args.source_dir)
    output = Path(args.output_dir)

    # Load saved data
    subtask_data = json.loads((source / "subtask_texts.json").read_text())
    subtask_texts = subtask_data["texts"]
    task_ids = subtask_data["task_ids"]
    task_languages = subtask_data["task_languages"]
    embeddings = np.load(source / "subtask_embeddings.npy")

    print(f"[Info] Loaded {len(subtask_texts)} subtasks, embeddings shape: {embeddings.shape}")
    print(f"[Info] Source: {source}")
    print(f"[Info] Output: {output}")

    # Save subtask_texts.json to output (with updated encoder_mode)
    output.mkdir(parents=True, exist_ok=True)
    subtask_data_out = dict(subtask_data)
    subtask_data_out["encoder_mode"] = "verbnet"
    (output / "subtask_texts.json").write_text(
        json.dumps(subtask_data_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Copy embeddings
    np.save(output / "subtask_embeddings.npy", embeddings)
    if (source / "s_goal_embeddings.npy").exists():
        np.save(output / "s_goal_embeddings.npy", np.load(source / "s_goal_embeddings.npy"))
    if (source / "s_env_embeddings.npy").exists():
        np.save(output / "s_env_embeddings.npy", np.load(source / "s_env_embeddings.npy"))

    # Run VerbNet hierarchical clustering
    verbnet_hierarchical_cluster(
        subtask_texts=subtask_texts,
        task_ids=task_ids,
        embeddings=embeddings,
        dist_threshold=0.6,
        output_dir=output,
        task_languages=task_languages,
    )

    print(f"\n[Done] VerbNet results saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
