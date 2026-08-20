from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "prototype" / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from data_probe import LeRobotLiberoDataset
from vlabench_video_smoke import decode_episode_frame


VLABENCH_ROOT = PROJECT_ROOT / "dataset" / "vlabench_unified"


def test_vlabench_unified_reader_smoke() -> None:
    if not VLABENCH_ROOT.exists():
        pytest.skip(f"VLABench dataset is not present: {VLABENCH_ROOT}")

    dataset = LeRobotLiberoDataset(VLABENCH_ROOT)
    info = dataset.load_info()
    tasks = dataset.load_tasks()
    episode_indices = dataset.sample_episode_indices(
        count=2,
        seed=0,
        min_length=80,
        max_length=160,
    )
    episode = dataset.load_episode(episode_indices[0], include_images=True)

    assert info["total_episodes"] >= 10000
    assert info["total_tasks"] >= 200
    assert len(tasks) == int(info["total_tasks"])
    assert episode.length >= 80
    assert episode.action.shape == episode.state.shape
    assert episode.action.shape[1] == 7
    assert episode.task_index in tasks
    assert episode.images == {}


def test_vlabench_video_frame_smoke() -> None:
    if not VLABENCH_ROOT.exists():
        pytest.skip(f"VLABench dataset is not present: {VLABENCH_ROOT}")

    pytest.importorskip("av")
    result = decode_episode_frame(
        root=VLABENCH_ROOT,
        episode_index=0,
        video_key="observation.images.image",
    )

    assert result["codec"] == "libdav1d"
    assert result["frame_shape"] == [224, 224, 3]
    assert result["frame_dtype"] == "uint8"
    assert result["frame_std"] > 0.0
