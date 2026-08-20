from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq


DEFAULT_DATASET_ROOT = Path(
    "/root/autodl-tmp/research/VLA_SKILL/datasets/HuggingFaceVLA_libero"
)


@dataclass(frozen=True)
class EpisodeInfo:
    episode_index: int
    task_index: int
    length: int
    max_frame_index: int

    def to_dict(self) -> dict:
        return {
            "episode_index": int(self.episode_index),
            "task_index": int(self.task_index),
            "length": int(self.length),
            "max_frame_index": int(self.max_frame_index),
        }


@dataclass
class EpisodeData:
    episode_index: int
    task_index: int
    frame_index: np.ndarray
    timestamp: np.ndarray
    state: np.ndarray
    action: np.ndarray
    images: Optional[Dict[str, List[dict]]] = None

    @property
    def length(self) -> int:
        return int(self.action.shape[0])


class LeRobotLiberoDataset:
    """Small reader for local LeRobot-format parquet datasets."""

    def __init__(self, root: Path = DEFAULT_DATASET_ROOT) -> None:
        self.root = Path(root)
        self.data_root = self.root / "data"
        self.meta_root = self.root / "meta"
        self._dataset: Optional[ds.Dataset] = None
        self._episodes: Optional[List[EpisodeInfo]] = None

    def validate(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        if not self.data_root.exists():
            raise FileNotFoundError(f"Dataset data directory not found: {self.data_root}")
        if not (self.meta_root / "info.json").exists():
            raise FileNotFoundError(f"Dataset metadata not found: {self.meta_root / 'info.json'}")

    @property
    def arrow_dataset(self) -> ds.Dataset:
        if self._dataset is None:
            self.validate()
            self._dataset = ds.dataset(self.data_root, format="parquet")
        return self._dataset

    def load_info(self) -> dict:
        self.validate()
        with (self.meta_root / "info.json").open("r", encoding="utf-8") as f:
            return json.load(f)

    def load_tasks(self) -> Dict[int, str]:
        tasks_path = self.meta_root / "tasks.parquet"
        if not tasks_path.exists():
            return {}
        table = pq.read_table(tasks_path)
        frame = table.to_pandas()
        tasks: Dict[int, str] = {}

        if "task" in frame.columns and "task_index" in frame.columns:
            for _, row in frame.iterrows():
                tasks[int(row["task_index"])] = str(row["task"])
            return tasks

        if frame.index.name == "task" and "task_index" in frame.columns:
            for language, row in frame.iterrows():
                tasks[int(row["task_index"])] = str(language)
            return tasks

        if "task_index" in frame.columns:
            string_columns = [
                col
                for col in frame.columns
                if col != "task_index" and frame[col].dtype == object
            ]
            if string_columns:
                language_col = string_columns[0]
                for _, row in frame.iterrows():
                    tasks[int(row["task_index"])] = str(row[language_col])
        return tasks

    def image_feature_keys(self) -> List[str]:
        features = self.load_info().get("features", {})
        return sorted(
            str(name)
            for name, spec in features.items()
            if str(name).startswith("observation.images.")
            and isinstance(spec, dict)
            and spec.get("dtype") == "video"
        )

    def list_episodes(self) -> List[EpisodeInfo]:
        if self._episodes is not None:
            return self._episodes

        table = self.arrow_dataset.to_table(
            columns=["episode_index", "task_index", "frame_index"]
        )
        frame = table.to_pandas()
        grouped = (
            frame.groupby("episode_index")
            .agg(
                task_index=("task_index", "first"),
                length=("frame_index", "count"),
                max_frame_index=("frame_index", "max"),
            )
            .reset_index()
            .sort_values("episode_index")
        )
        self._episodes = [
            EpisodeInfo(
                episode_index=int(row.episode_index),
                task_index=int(row.task_index),
                length=int(row.length),
                max_frame_index=int(row.max_frame_index),
            )
            for row in grouped.itertuples(index=False)
        ]
        return self._episodes

    def sample_episode_indices(
        self,
        count: int,
        seed: int = 0,
        min_length: int = 80,
        max_length: Optional[int] = None,
    ) -> List[int]:
        episodes = [
            ep
            for ep in self.list_episodes()
            if ep.length >= min_length and (max_length is None or ep.length <= max_length)
        ]
        if not episodes:
            raise ValueError("No episodes match the requested length constraints")
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(episodes), size=min(count, len(episodes)), replace=False)
        return [episodes[int(i)].episode_index for i in chosen]

    def load_episode(
        self, episode_index: int, include_images: bool = False
    ) -> EpisodeData:
        columns: List[str] = [
            "episode_index",
            "task_index",
            "frame_index",
            "timestamp",
            "observation.state",
            "action",
        ]
        if include_images:
            data_columns = set(self.arrow_dataset.schema.names)
            columns.extend(
                key for key in self.image_feature_keys() if key in data_columns
            )

        table = self.arrow_dataset.to_table(
            columns=columns,
            filter=ds.field("episode_index") == int(episode_index),
        )
        if table.num_rows == 0:
            raise KeyError(f"Episode {episode_index} not found")

        frame = table.to_pandas().sort_values("frame_index")
        task_index = int(frame["task_index"].iloc[0])
        images = None
        if include_images:
            images = {
                key: frame[key].tolist()
                for key in self.image_feature_keys()
                if key in frame.columns
            }

        return EpisodeData(
            episode_index=int(episode_index),
            task_index=task_index,
            frame_index=frame["frame_index"].to_numpy(dtype=np.int64),
            timestamp=frame["timestamp"].to_numpy(dtype=np.float32),
            state=np.asarray(frame["observation.state"].tolist(), dtype=np.float32),
            action=np.asarray(frame["action"].tolist(), dtype=np.float32),
            images=images,
        )

    def summarize(self) -> dict:
        info = self.load_info()
        tasks = self.load_tasks()
        episodes = self.list_episodes()
        lengths = np.asarray([ep.length for ep in episodes], dtype=np.float32)
        task_counts: Dict[int, int] = {}
        for ep in episodes:
            task_counts[ep.task_index] = task_counts.get(ep.task_index, 0) + 1

        return {
            "dataset_root": str(self.root),
            "total_episodes_meta": int(info.get("total_episodes", len(episodes))),
            "total_frames_meta": int(info.get("total_frames", int(lengths.sum()))),
            "total_tasks_meta": int(info.get("total_tasks", len(tasks))),
            "episodes_indexed": len(episodes),
            "tasks_indexed": len(tasks),
            "episode_length": {
                "min": int(lengths.min()),
                "p25": float(np.percentile(lengths, 25)),
                "median": float(np.percentile(lengths, 50)),
                "p75": float(np.percentile(lengths, 75)),
                "p90": float(np.percentile(lengths, 90)),
                "max": int(lengths.max()),
                "mean": float(lengths.mean()),
            },
            "task_episode_counts": {str(k): int(v) for k, v in sorted(task_counts.items())},
            "sample_tasks": {
                str(k): tasks[k]
                for k in sorted(tasks.keys())[: min(10, len(tasks))]
            },
        }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Probe the local LeRobot LIBERO dataset.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--include-images", action="store_true")
    args = parser.parse_args(argv)

    dataset = LeRobotLiberoDataset(args.dataset_root)
    print(json.dumps(dataset.summarize(), indent=2, ensure_ascii=False))
    if args.episode is not None:
        episode = dataset.load_episode(args.episode, include_images=args.include_images)
        print(
            json.dumps(
                {
                    "episode_index": episode.episode_index,
                    "task_index": episode.task_index,
                    "length": episode.length,
                    "state_shape": list(episode.state.shape),
                    "action_shape": list(episode.action.shape),
                    "images_loaded": episode.images is not None,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
