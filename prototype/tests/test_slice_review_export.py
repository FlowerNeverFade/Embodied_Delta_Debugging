from __future__ import annotations

import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "prototype" / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from slice_review_export import collect_review_cases, export_reviews, frame_window_for_slice, parse_args


def _write_dummy_video(path: Path, frames: int = 12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (64, 64),
    )
    assert writer.isOpened()
    try:
        for i in range(frames):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            frame[:, :, 0] = i * 10
            frame[10:20, 10:20, 1] = 255
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _fake_report(path: Path, video_path: Path, passed: bool = True) -> None:
    repair_eval = {
        "success": True,
        "same_failure_rate": 0.0,
        "failure_signature": {
            "failure_type": "success",
            "failed_goal_predicates": [],
            "affected_objects": [],
            "evidence": {
                "goal_trace": {
                    "failed_final_predicates": [],
                    "progress_counts": [2],
                }
            },
        },
    }
    best_counterfactual = {
        "strategy": "hold",
        "unit_id": "semantic_anchor_3_7",
        "repair_pass": True,
        "evaluation": repair_eval,
    }
    report = {
        "schema_version": "shed-cfs-causal-v1",
        "selected_failed_rollout": {
            "task_id": 8,
            "init_state_id": 4,
            "task_language": "put both moka pots on the stove",
            "length": 10,
            "video_path": str(video_path),
            "video_frames": 12,
        },
        "video_config": {"record_video": True, "video_fps": 30, "video_every_n": 1},
        "original_failure_signature": {
            "failure_type": "wrong_object",
            "failed_goal_predicates": ["On moka_pot_1 flat_stove_1_cook_region"],
        },
        "reproduction_statistics": {
            "candidate": {
                "level": "minimal_pi05_natural_slice",
                "intervals": [[4, 6]],
                "length": 2,
                "span": [4, 6],
            },
            "same_failure_rate": 1.0,
        },
        "causal_validation": {"passed": passed, "base_same_failure_rate": 1.0},
        "causal_core_units": [
            {
                "unit": {
                    "unit_id": "semantic_anchor_3_7",
                    "kind": "goal_predicate_anchor",
                    "interval": [3, 7],
                },
                "base_same_failure_rate": 1.0,
                "ablated_same_failure_rate": 0.0,
                "causal_effect": 1.0,
                "repair_pass": True,
                "best_counterfactual": best_counterfactual,
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")


def _fake_v2_necessity_report(path: Path, video_path: Path) -> None:
    _fake_report(path, video_path, passed=False)
    report = json.loads(path.read_text(encoding="utf-8"))
    report["schema_version"] = "shed-cfs-causal-v2-split-repair"
    report["same_failure_necessity_pass"] = True
    report["repair_valid_causal_pass"] = False
    report["necessity_core_units"] = report["causal_core_units"]
    report["causal_core_units"] = []
    report["causal_validation"] = {
        "passed": False,
        "same_failure_necessity_pass": True,
        "repair_valid_causal_pass": False,
        "base_same_failure_rate": 1.0,
        "necessity_core_units": report["necessity_core_units"],
        "causal_core_units": [],
    }
    path.write_text(json.dumps(report), encoding="utf-8")


def test_frame_window_for_slice_clamps() -> None:
    assert frame_window_for_slice(4, 6, fps=30, context_seconds=1.0, total_frames=12) == (0, 11)
    assert frame_window_for_slice(100, 102, fps=30, context_seconds=1.0, total_frames=160) == (70, 132)


def test_collect_review_cases_filters_causal_pass(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    _write_dummy_video(video)
    good = tmp_path / "task08_init04_seed27_causal_v1.json"
    bad = tmp_path / "task08_init05_seed27_causal_v1.json"
    _fake_report(good, video, passed=True)
    _fake_report(bad, video, passed=False)

    cases = collect_review_cases([tmp_path])

    assert [case.case_id for case in cases] == ["task08_init04_seed27"]
    assert cases[0].minimal_start == 4
    assert cases[0].minimal_end == 6
    assert cases[0].full_success_repair_pass


def test_collect_review_cases_requires_explicit_necessity_only_flag(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    _write_dummy_video(video)
    necessity = tmp_path / "task08_init04_seed27_causal_v2.json"
    _fake_v2_necessity_report(necessity, video)

    assert collect_review_cases([tmp_path]) == []
    cases = collect_review_cases([tmp_path], include_necessity_only=True)

    assert [case.case_id for case in cases] == ["task08_init04_seed27"]
    assert cases[0].review_semantics == "same_failure_necessity_only"


def test_export_original_context_without_replay(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    _write_dummy_video(video)
    report = tmp_path / "reports" / "task08_init04_seed27_causal_v1.json"
    _fake_report(report, video, passed=True)
    output = tmp_path / "review"
    args = parse_args(
        [
            "--report-dir",
            str(report.parent),
            "--output-dir",
            str(output),
            "--no-record-replays",
            "--context-seconds",
            "0.1",
        ]
    )

    manifest = export_reviews(args)

    assert manifest["num_cases"] == 1
    assert (output / "review_manifest.json").exists()
    assert (output / "review_index.html").exists()
    case_dir = output / "cases" / "task08_init04_seed27"
    assert (case_dir / "original_context.mp4").exists()
    assert (case_dir / "case_review.json").exists()
    review = json.loads((case_dir / "case_review.json").read_text(encoding="utf-8"))
    assert review["schema_version"] == "slice-review-case-v2-split-repair"
    assert "reels" in manifest
