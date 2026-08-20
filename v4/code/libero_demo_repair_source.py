from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_DATASET_ROOT = Path(
    "/root/autodl-tmp/research/VLA_SKILL/datasets/HuggingFaceVLA_libero"
)


def _load_task_index(dataset_root: Path, task_language: str) -> Optional[int]:
    tasks = pd.read_parquet(dataset_root / "meta" / "tasks.parquet")
    normalized = task_language.strip().lower()
    for idx, row in tasks.iterrows():
        text = str(idx).strip().lower()
        if text == normalized:
            return int(row["task_index"])
    for idx, row in tasks.iterrows():
        text = str(idx).strip().lower()
        if normalized in text or text in normalized:
            return int(row["task_index"])
    return None


def find_demo_repair_actions(
    dataset_root: Path,
    task_language: str,
    eef_pos: Sequence[float],
    max_steps: int,
) -> dict:
    task_index = _load_task_index(dataset_root, task_language)
    if task_index is None:
        return {"available": False, "reason": "task_language_not_found"}

    target = np.asarray(list(eef_pos)[:3], dtype=np.float64)
    best = None
    columns = ["task_index", "episode_index", "frame_index", "observation.state", "action"]
    for path in sorted((dataset_root / "data").glob("chunk-*/*.parquet")):
        try:
            df = pd.read_parquet(path, columns=columns)
        except Exception:
            continue
        df = df[df["task_index"] == int(task_index)]
        if df.empty:
            continue
        for episode_index, ep in df.groupby("episode_index", sort=True):
            states = np.stack(ep["observation.state"].to_numpy())
            if states.shape[1] < 3:
                continue
            dists = np.linalg.norm(states[:, :3].astype(np.float64) - target[None, :], axis=1)
            nearest_local = int(np.argmin(dists))
            score = float(dists[nearest_local])
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "path": path,
                    "episode_index": int(episode_index),
                    "nearest_local": nearest_local,
                    "episode": ep.reset_index(drop=True),
                }
        if best is not None and best["score"] < 0.03:
            break

    if best is None:
        return {"available": False, "reason": "no_matching_episode"}

    ep = best["episode"]
    start = int(best["nearest_local"])
    end = min(len(ep), start + max(1, int(max_steps)))
    actions = [np.asarray(action, dtype=np.float32).reshape(-1).tolist() for action in ep["action"].iloc[start:end]]
    requested_steps = max(1, int(max_steps))
    return {
        "available": bool(actions),
        "reason": None if actions else "empty_action_suffix",
        "source": "lerobot_demo_nearest_neighbor",
        "dataset_root": str(dataset_root),
        "task_index": int(task_index),
        "episode_index": int(best["episode_index"]),
        "parquet_path": str(best["path"]),
        "nearest_frame_index": int(ep["frame_index"].iloc[start]),
        "nearest_state_l2": float(best["score"]),
        "num_actions": len(actions),
        "requested_steps": int(requested_steps),
        "action_suffix_complete": bool(len(actions) >= requested_steps),
        "actions": actions,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch a LIBERO LeRobot demo suffix for repair replay.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--task-language", required=True)
    parser.add_argument("--eef-pos", required=True, help="Comma separated xyz eef position.")
    parser.add_argument("--max-steps", type=int, default=520)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    eef = [float(x) for x in args.eef_pos.split(",") if x.strip()]
    result = find_demo_repair_actions(args.dataset_root, args.task_language, eef, args.max_steps)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
