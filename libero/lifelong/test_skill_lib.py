"""
Skill library builder for LIBERO-90.

For every task in libero_90:
  1. Render the initial observation (agentview + wrist camera).
  2. Call QwenPlanner.get_subtasks() to produce atomic sub-task strings.
  3. Encode each sub-task with BERT (mean-pooled CLS token) to get a vector.

Then cluster all sub-task vectors with distance-threshold agglomerative
clustering (no fixed k – a new cluster is spawned whenever a point is
farther than --dist_threshold from every existing centroid) and visualise:
  - 2-D t-SNE scatter coloured by cluster, ★ marks each centroid representative.
  - Per-cluster text summary with representative highlighted.
  - Per-task cluster assignment JSON.

Run from the LIBERO-VLA repo root:
  python libero/lifelong/test_skill_lib.py \
      --model_path /path/to/Qwen3-VL-4B-Instruct \
      --dist_threshold 0.9 \
      --output_dir skill_lib_results

Optional flags:
  --task_suite_name   libero_90 (default)
  --max_tasks         cap the number of tasks processed (default: all)
  --seed              random seed (default: 7)
  --device            cuda / cpu (default: cuda)
  --dist_threshold    Euclidean distance threshold in L2-normalised BERT space
                      (default 0.9; range ~0–2; smaller → more clusters)
  --no_render         skip environment rendering; use blank images
                      (useful for a quick embedding-only run)
"""

import argparse
import importlib.util
import json
import os
import gc
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # headless backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModel

# ---- resolve repo root & load local QwenPlanner -----------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE_DIR = _REPO_ROOT / "tmp"   # persistent subtask cache
_QWEN_PLANNER_FILE = (
    _REPO_ROOT / "libero" / "lifelong" / "models" / "modules" / "QwenPlanner.py"
)
_spec = importlib.util.spec_from_file_location("QwenPlanner", _QWEN_PLANNER_FILE)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
QwenPlanner = _module.QwenPlanner

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


# -----------------------------------------------------------------------
# BERT text encoder
# -----------------------------------------------------------------------

class BertEncoder:
    """Mean-pool BERT encoder (bert-base-uncased, cached locally)."""

    def __init__(self, model_name: str = "bert-base-uncased", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[BertEncoder] loading {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Return (N, hidden) float32 numpy array."""
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**enc)
            # mean-pool over non-padding tokens
            mask = enc["attention_mask"].unsqueeze(-1).float()
            vecs = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            all_vecs.append(vecs.cpu().float().numpy())
        return np.concatenate(all_vecs, axis=0)


# -----------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------

def build_env(task_bddl_file: Path, seed: int) -> OffScreenRenderEnv:
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(seed)
    return env


def get_initial_obs(env: OffScreenRenderEnv, init_state) -> dict:
    env.reset()
    return env.set_init_state(init_state)


def obs_to_images(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (agentview, wrist) images both rotated 180° (eval convention)."""
    main_img  = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return main_img, wrist_img


# -----------------------------------------------------------------------
# clustering & visualisation
# -----------------------------------------------------------------------

def cluster_and_visualise(
    subtask_texts: list[str],
    task_ids: list[int],        # which libero task each subtask belongs to
    embeddings: np.ndarray,
    dist_threshold: float,      # Euclidean distance threshold in L2-norm space
    output_dir: Path,
    task_languages: list[str],  # high-level language per task_id index
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Agglomerative clustering with distance threshold (no fixed k) ----
    # Embeddings are L2-normalised so Euclidean distance ∈ [0, 2].
    # A new cluster is created whenever a merge would exceed dist_threshold.
    norm_emb = normalize(embeddings)
    print(f"\n[Cluster] Agglomerative clustering (threshold={dist_threshold}) "
          f"on {len(subtask_texts)} subtasks …")
    agg = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=dist_threshold,
        linkage="average",
        metric="euclidean",
    )
    labels = agg.fit_predict(norm_emb)
    n_clusters = int(labels.max()) + 1
    print(f"[Cluster] Found {n_clusters} clusters automatically.")

    # ---- compute centroids manually & find closest sample ----
    center_indices: dict[int, int] = {}   # cluster_id -> sample index
    center_texts:   dict[int, str] = {}   # cluster_id -> representative text
    for cid in range(n_clusters):
        mask = np.where(labels == cid)[0]
        centroid = norm_emb[mask].mean(axis=0)
        dists = np.linalg.norm(norm_emb[mask] - centroid, axis=1)
        closest_global = int(mask[int(np.argmin(dists))])
        center_indices[cid] = closest_global
        center_texts[cid] = subtask_texts[closest_global]

    # ---- t-SNE ----
    print("[Cluster] Running t-SNE for 2-D projection …")
    perplexity = min(30, len(subtask_texts) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    xy = tsne.fit_transform(embeddings)

    # ---- scatter plot ----
    # tab20 has 20 colours; cycle if more clusters
    cmap20 = plt.get_cmap("tab20")
    def _color(cid):
        return cmap20((cid % 20) / 20)

    colors = [_color(l) for l in labels]
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=18, alpha=0.60, linewidths=0, zorder=2)

    # highlight centroid representatives: star marker + annotation
    for cid, idx in center_indices.items():
        cx, cy = xy[idx]
        color = _color(cid)
        ax.scatter(cx, cy, c=[color], s=280, marker="*",
                   edgecolors="black", linewidths=0.6, zorder=5)
        label_text = center_texts[cid]
        if len(label_text) > 28:
            label_text = label_text[:26] + "…"
        ax.annotate(
            f"C{cid}: {label_text}",
            xy=(cx, cy), xytext=(6, 4), textcoords="offset points",
            fontsize=5.5, color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color,
                      alpha=0.85, linewidth=0.8),
            zorder=6,
        )

    patches = [
        mpatches.Patch(
            color=_color(i),
            label=f"C{i}: {center_texts[i][:35]}{'…' if len(center_texts[i]) > 35 else ''}"
        )
        for i in range(n_clusters)
    ]
    ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=6.0, title=f"Cluster (★ rep) – k={n_clusters} auto",
              title_fontsize=7)
    ax.set_title(
        f"Skill library – LIBERO-90 subtask clusters\n"
        f"threshold={dist_threshold}  →  k={n_clusters} clusters   "
        f"★ = sample closest to centroid"
    )
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    plt.tight_layout()
    scatter_path = output_dir / "skill_clusters_tsne.png"
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print(f"[Cluster] Scatter plot saved → {scatter_path}")

    # ---- per-cluster text summary ----
    cluster_info: dict[int, list[str]] = {i: [] for i in range(n_clusters)}
    for text, lbl in zip(subtask_texts, labels):
        cluster_info[lbl].append(text)

    summary_lines = []
    print("\n[Cluster] Representative (★ = closest to centroid) per cluster:")
    for cid in range(n_clusters):
        members = cluster_info[cid]
        rep = center_texts[cid]
        summary_lines.append(f"\n{'='*60}")
        summary_lines.append(f"Cluster {cid}  ({len(members)} subtasks)")
        summary_lines.append(f"  ★ Representative: {rep}")
        summary_lines.append(f"{'='*60}")
        print(f"  C{cid:2d} ★ {rep}")
        seen = set()
        deduped = []
        for m in members:
            key = m.lower().strip()
            if key not in seen:
                seen.add(key)
                deduped.append(m)
        for m in deduped[:30]:
            prefix = "  ★" if m == rep else "  •"
            summary_lines.append(f"{prefix} {m}")
        if len(deduped) > 30:
            summary_lines.append(f"  … ({len(deduped) - 30} more unique entries)")

    summary_text = "\n".join(summary_lines)
    summary_path = output_dir / "skill_clusters_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"\n[Cluster] Text summary saved → {summary_path}")

    # ---- flat records JSON (one entry per subtask) ----
    lbl_list = labels.tolist()
    records = []
    for text, tid, lbl in zip(subtask_texts, task_ids, lbl_list):
        records.append({
            "subtask": text,
            "task_id": tid,
            "task_language": task_languages[tid],
            "cluster": lbl,
            "cluster_representative": center_texts[lbl],
            "is_representative": (text == center_texts[lbl]),
        })
    json_path = output_dir / "skill_lib_records.json"
    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Cluster] Flat records saved → {json_path}")

    # ---- per-task cluster assignment JSON ----
    # Build a dict keyed by task_id listing each subtask + its cluster.
    task_cluster_map: dict[int, dict] = {}
    for text, tid, lbl in zip(subtask_texts, task_ids, lbl_list):
        if tid not in task_cluster_map:
            task_cluster_map[tid] = {
                "task_id": tid,
                "task_language": task_languages[tid],
                "subtasks": [],
            }
        task_cluster_map[tid]["subtasks"].append({
            "text": text,
            "cluster": lbl,
            "cluster_representative": center_texts[lbl],
            "is_representative": (text == center_texts[lbl]),
        })
    # sort by task_id and serialise
    task_list = [task_cluster_map[tid] for tid in sorted(task_cluster_map)]
    task_json_path = output_dir / "task_cluster_assignments.json"
    task_json_path.write_text(
        json.dumps(task_list, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[Cluster] Per-task assignments saved → {task_json_path}")

    # ---- cluster representatives mapping ----
    rep_map = {str(cid): center_texts[cid] for cid in range(n_clusters)}
    rep_path = output_dir / "cluster_representatives.json"
    rep_path.write_text(json.dumps(rep_map, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Cluster] Representatives saved → {rep_path}")

    return labels, cluster_info


# -----------------------------------------------------------------------
# main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suite_name", type=str, default="libero_90")
    parser.add_argument("--max_tasks", type=int, default=None,
                        help="Cap number of tasks (default: all 90)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model_path", type=str,
                        default="/export/ra/liyuxuan/VLA/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--bert_model", type=str, default="/data/models/liyuxuan/bert-base-uncased")
    parser.add_argument("--dist_threshold", type=float, default=0.9,
                        help="Euclidean distance threshold in L2-normalised BERT embedding space. "
                             "Smaller value → more clusters. Range ~0–2 (default: 0.9).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="skill_lib_results")
    parser.add_argument("--no_render", action="store_true",
                        help="Skip environment rendering; use blank 256x256 images")
    parser.add_argument("--subtasks_cache", type=str, default=None,
                        help="Path to a cached subtask JSON file. "
                             "If omitted, auto-detects tmp/<suite>_subtasks.json. "
                             "Pass '--subtasks_cache none' to force re-run.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # ---- load benchmark ----
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = task_suite.n_tasks
    if args.max_tasks is not None:
        n_tasks = min(n_tasks, args.max_tasks)
    print(f"[Info] Processing {n_tasks} tasks from '{args.task_suite_name}'")

    bddl_root = Path(get_libero_path("bddl_files"))

    # ---- determine cache path ----
    _DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    auto_cache = _DEFAULT_CACHE_DIR / f"{args.task_suite_name}_subtasks.json"
    if args.subtasks_cache is None:
        cache_path = auto_cache if auto_cache.exists() else None
    elif args.subtasks_cache.lower() == "none":
        cache_path = None          # force re-run
    else:
        cache_path = Path(args.subtasks_cache)

    all_subtask_texts: list[str] = []
    all_task_ids: list[int] = []
    task_languages: list[str] = []

    if cache_path is not None and cache_path.exists():
        # ---- load subtasks from cache (skip planner) ----
        print(f"[Info] Loading subtasks from cache: {cache_path}")
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        all_subtask_texts = cached["texts"]
        all_task_ids      = cached["task_ids"]
        task_languages    = cached["task_languages"]
        # only keep tasks within n_tasks limit
        if args.max_tasks is not None:
            keep = [i for i, tid in enumerate(all_task_ids) if tid < n_tasks]
            all_subtask_texts = [all_subtask_texts[i] for i in keep]
            all_task_ids      = [all_task_ids[i]      for i in keep]
            task_languages    = task_languages[:n_tasks]
        print(f"[Info] Loaded {len(all_subtask_texts)} subtasks for "
              f"{len(set(all_task_ids))} tasks from cache.")
    else:
        # ---- run planner ----
        print(f"[Info] Loading QwenPlanner from {args.model_path}")
        planner = QwenPlanner(model_path=args.model_path, device=args.device)

        for task_id in range(n_tasks):
            task = task_suite.get_task(task_id)
            task_lang = task.language
            task_languages.append(task_lang)
            task_bddl_file = bddl_root / task.problem_folder / task.bddl_file

            print(f"\n[Task {task_id:3d}/{n_tasks}] {task_lang}")

            if args.no_render:
                blank = np.zeros((256, 256, 3), dtype=np.uint8)
                main_img, wrist_img = blank, blank
            else:
                try:
                    env = build_env(task_bddl_file, args.seed)
                    init_states = task_suite.get_task_init_states(task_id)
                    obs = get_initial_obs(env, init_states[0])
                    main_img, wrist_img = obs_to_images(obs)
                    env.close()
                    del env
                    gc.collect()
                except Exception as exc:
                    print(f"  [Warn] env failed ({exc}); using blank images")
                    blank = np.zeros((256, 256, 3), dtype=np.uint8)
                    main_img, wrist_img = blank, blank

            subtasks = planner.get_subtasks(
                high_task=task_lang,
                image_list=[main_img, wrist_img],
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
            )

            if not subtasks:
                print("  [Warn] planner returned empty list; skipping task")
                continue

            print(f"  → {len(subtasks)} subtasks:")
            for i, st in enumerate(subtasks, 1):
                print(f"     {i}. {st}")

            for st in subtasks:
                all_subtask_texts.append(st)
                all_task_ids.append(task_id)

        # free VRAM before BERT
        del planner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---- persist subtasks to tmp/ cache ----
        cache_data = {
            "task_suite_name": args.task_suite_name,
            "texts": all_subtask_texts,
            "task_ids": all_task_ids,
            "task_languages": task_languages,
        }
        auto_cache.write_text(
            json.dumps(cache_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[Info] Subtasks cached → {auto_cache}")

    if not all_subtask_texts:
        print("[Error] No subtasks collected. Exiting.")
        return

    print(f"\n[Info] Total subtasks collected: {len(all_subtask_texts)}")

    # ---- BERT embeddings ----
    print("[Info] Encoding subtasks with BERT …")
    encoder = BertEncoder(model_name=args.bert_model, device=args.device)
    embeddings = encoder.encode(all_subtask_texts)
    print(f"[Info] Embedding matrix: {embeddings.shape}")

    # ---- save raw data before clustering (checkpoint) ----
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "subtask_embeddings.npy", embeddings)
    (output_dir / "subtask_texts.json").write_text(
        json.dumps(
            {"texts": all_subtask_texts, "task_ids": all_task_ids, "task_languages": task_languages},
            indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    print(f"[Info] Raw embeddings & texts saved to {output_dir}/")

    # ---- clustering & visualisation ----
    cluster_and_visualise(
        subtask_texts=all_subtask_texts,
        task_ids=all_task_ids,
        embeddings=embeddings,
        dist_threshold=args.dist_threshold,
        output_dir=output_dir,
        task_languages=task_languages,
    )

    print("\n[Done] All outputs written to:", output_dir.resolve())


if __name__ == "__main__":
    main()
