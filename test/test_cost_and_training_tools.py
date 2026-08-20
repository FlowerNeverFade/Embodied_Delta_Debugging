from __future__ import annotations

import copy
import json
import sys
import socket
import time
from types import SimpleNamespace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cost_summary import build_cost_summary
from repair_sft_admission import build_repair_sft_admission
from risk_critic_export import (
    _strict_causal_pass,
    export_risk_critic_dataset,
    export_risk_critic_dataset_from_paths,
)
import run_risk_critic_large_eval as large_eval
from train_risk_critic import train_and_evaluate


def _fake_causal_report() -> dict:
    repair_eval = {
        "candidate": {"level": "causal_ablation_hold", "span": [12, 14], "length": 2},
        "same_failure_rate": 0.0,
        "failure_rate": 0.0,
        "success": True,
        "failure_signature": {
            "failure_type": "success",
            "failed_goal_predicates": [],
            "affected_objects": [],
            "evidence": {
                "goal_trace": {
                    "failed_final_predicates": [],
                    "progress_counts": [2],
                    "final_truth": {
                        "On moka_pot_1 flat_stove_1_cook_region": True,
                        "On moka_pot_2 flat_stove_1_cook_region": True,
                    },
                }
            },
        },
    }
    repair_best = {
        "strategy": "hold",
        "unit_id": "semantic_anchor_012_014",
        "repair_pass": True,
        "repair_evidence": {
            "repair_pass": True,
            "success": True,
            "base_failed_goal_count": 1,
            "counterfactual_failed_goal_count": 0,
            "base_goal_progress": 1,
            "counterfactual_goal_progress": 2,
            "non_worsening": True,
            "improved": True,
        },
        "evaluation": repair_eval,
    }
    return {
        "schema_version": "shed-cfs-causal-v1",
        "created_at": "2026-05-19T00:00:00+00:00",
        "search_config": {
            "task_suite_name": "libero_10",
            "task_ids": [8],
            "init_state_ids": [0],
            "replay_trials": 5,
            "search_replay_trials": 1,
            "confirm_replay_trials": 5,
        },
        "rollout_summaries": [
            {
                "task_id": 8,
                "init_state_id": 0,
                "task_language": "put both moka pots on the stove",
                "success": False,
                "max_window_event": [10, 20, 0.95],
            },
            {
                "task_id": 8,
                "init_state_id": 1,
                "task_language": "put both moka pots on the stove",
                "success": True,
                "max_window_event": [12, 22, 0.2],
            },
        ],
        "selected_failed_rollout": {
            "task_id": 8,
            "init_state_id": 0,
            "task_language": "put both moka pots on the stove",
            "success": False,
            "length": 40,
        },
        "original_failure_signature": {
            "failure_type": "wrong_object",
            "failed_goal_predicates": ["On moka_pot_1 flat_stove_1_cook_region"],
            "affected_objects": ["moka_pot_1"],
            "semantic_quality": "full",
        },
        "failure_event": {"window": [10, 20], "semantic_confidence": 0.95},
        "causal_failure_slice": {"level": "pi05_natural_replay", "span": [12, 14], "length": 2},
        "slice_training_features": {
            "feature_quality": "full",
            "candidate_window": [12, 14],
            "candidate_actions": [[0.1] * 7, [0.2] * 7],
            "candidate_actions_truncated": False,
            "action_summary": {
                "num_actions": 2,
                "action_dim": 7,
                "mean": [0.15] * 7,
                "std": [0.05] * 7,
                "max_abs": [0.2] * 7,
                "gripper_transitions": 0,
            },
            "pre_state_features": {"eef_pos": [0, 0, 0], "gripper_qpos": [0], "goal_truth": {}},
            "post_state_features": {"eef_pos": [1, 0, 0], "gripper_qpos": [1], "goal_truth": {}},
        },
        "reproduction_statistics": {
            "candidate": {"level": "pi05_natural_replay", "span": [12, 14], "length": 2},
            "same_failure": True,
            "same_failure_rate": 1.0,
            "failure_rate": 1.0,
        },
        "causal_validation": {
            "base_same_failure_rate": 1.0,
            "passed": True,
            "causal_core_units": [
                {
                    "unit": {
                        "unit_id": "semantic_anchor_012_014",
                        "kind": "goal_predicate_anchor",
                        "interval": [12, 14],
                    },
                    "base_same_failure_rate": 1.0,
                    "ablated_same_failure_rate": 0.0,
                    "causal_effect": 1.0,
                    "is_causal_core": True,
                    "repair_pass": True,
                    "best_counterfactual": repair_best,
                }
            ],
            "unit_results": [],
            "counterfactual_pass_variants": [],
        },
        "causal_core_units": [
            {
                "unit": {
                    "unit_id": "semantic_anchor_012_014",
                    "kind": "goal_predicate_anchor",
                    "interval": [12, 14],
                },
                "base_same_failure_rate": 1.0,
                "ablated_same_failure_rate": 0.0,
                "causal_effect": 1.0,
                "is_causal_core": True,
                "repair_pass": True,
                "best_counterfactual": repair_best,
            }
        ],
        "counterfactual_pass_variants": [
            repair_best
        ],
        "metrics": {
            "trajectory_reduction_ratio": 20.0,
            "event_reduction_ratio": 5.0,
            "replay_evaluations": 3,
        },
        "runtime_profile": {
            "total_wall_seconds": 12.5,
            "durations_seconds": {"rollout_seconds": 8.0, "minimization_seconds": 4.0},
            "counters": {"policy_queries": 3, "env_resets": 6, "simulator_suffix_steps": 80},
        },
        "cost_summary": {
            "total_wall_seconds": 12.5,
            "policy_queries": 3,
            "env_resets": 6,
            "simulator_suffix_steps_measured": 80,
        },
        "feasibility": {"pi05_natural_pass": True},
    }


def _fake_full_risk_window(sample_kind: str, label: int, offset: float) -> dict:
    return {
        "sample_id": f"{sample_kind}_{offset}",
        "sample_kind": sample_kind,
        "label": label,
        "label_source": "test",
        "task": {
            "task_suite_name": "libero_10",
            "task_id": 8,
            "init_state_id": 0,
            "task_language": "put both moka pots on the stove",
        },
        "candidate": {"level": sample_kind, "span": [12, 14], "length": 2},
        "features": {
            "feature_quality": "full",
            "candidate_window": [12, 14],
            "candidate_actions": [[offset + 0.1] * 7, [offset + 0.2] * 7],
            "candidate_actions_truncated": False,
            "action_summary": {
                "num_actions": 2,
                "action_dim": 7,
                "mean": [offset + 0.15] * 7,
                "std": [0.05] * 7,
                "max_abs": [offset + 0.2] * 7,
                "gripper_transitions": int(label),
            },
            "pre_state_features": {
                "t": 12,
                "success": False,
                "eef_pos": [offset, 0.0, 0.0],
                "gripper_qpos": [0.0, 0.0],
                "goal_truth": {"On moka_pot_1 flat_stove_1_cook_region": False},
                "object_positions": {"moka_pot_1": [offset, 0.0, 0.0]},
            },
            "post_state_features": {
                "t": 14,
                "success": bool(1 - label),
                "eef_pos": [offset + 0.1, 0.0, 0.0],
                "gripper_qpos": [float(label), 0.0],
                "goal_truth": {
                    "On moka_pot_1 flat_stove_1_cook_region": bool(1 - label)
                },
                "object_positions": {"moka_pot_1": [offset + 0.1, 0.0, 0.0]},
            },
            "state_delta_features": {
                "eef_delta": [0.1, 0.0, 0.0],
                "gripper_delta": [float(label), 0.0],
                "goal_truth_delta": {
                    "On moka_pot_1 flat_stove_1_cook_region": int(1 - label)
                },
                "object_position_delta": {
                    "moka_pot_1": {"delta": [0.1, 0.0, 0.0], "l2": 0.1}
                },
            },
        },
        "same_failure_rate": 1.0 if label else 0.0,
        "failure_rate": 1.0 if label else 0.0,
        "causal_effect": 0.8 if label else 0.0,
        "causal_validation_passed": bool(label),
    }


def _fake_v2_necessity_only_report() -> dict:
    report = _fake_causal_report()
    report["schema_version"] = "shed-cfs-causal-v2-split-repair"
    report["same_failure_necessity_pass"] = True
    report["repair_valid_causal_pass"] = False
    destructive = copy.deepcopy(report["causal_core_units"][0])
    destructive["is_causal_core"] = False
    destructive["is_necessity_core"] = True
    destructive["repair_pass"] = False
    destructive["repair_evidence"] = {
        "repair_pass": False,
        "base_failed_goal_count": 1,
        "counterfactual_failed_goal_count": 2,
        "failed_goal_subset_of_original": False,
        "failed_goal_growth": 1,
        "non_worsening": False,
        "improved": False,
    }
    destructive["best_counterfactual"]["destructive_ablation"] = True
    destructive["best_counterfactual"]["repair_pass"] = False
    destructive["best_counterfactual"]["repair_evidence"] = destructive["repair_evidence"]
    report["necessity_core_units"] = [destructive]
    report["causal_core_units"] = []
    report["repair_valid_causal_core_units"] = []
    report["destructive_ablation_variants"] = [destructive["best_counterfactual"]]
    report["repair_pass_variants"] = []
    report["counterfactual_pass_variants"] = []
    report["causal_validation"] = {
        "base_same_failure_rate": 1.0,
        "same_failure_necessity_pass": True,
        "repair_valid_causal_pass": False,
        "passed": False,
        "necessity_core_units": [destructive],
        "causal_core_units": [],
        "unit_results": [destructive],
        "destructive_ablation_variants": [destructive["best_counterfactual"]],
        "repair_pass_variants": [],
    }
    unit = destructive["unit"]
    report["risk_training_windows"] = [
        _fake_full_risk_window("same_failure_necessity_slice", 1, 1.0),
        {
            **_fake_full_risk_window("same_failure_necessity_core", 1, 2.0),
            "causal_unit": unit,
        },
        {
            **_fake_full_risk_window("repair_valid_causal_core", 1, 3.0),
            "causal_unit": unit,
            "repair_evidence": destructive["repair_evidence"],
        },
        _fake_full_risk_window("successful_rollout_window", 0, 10.0),
    ]
    report["feasibility"] = {
        "pi05_natural_pass": False,
        "same_failure_necessity_pass": True,
        "repair_valid_causal_pass": False,
    }
    return report


def test_cost_summary_reads_causal_reports(tmp_path: Path) -> None:
    report_path = tmp_path / "run" / "task8_repeat_01_causal_v1.json"
    report_path.parent.mkdir()
    report_path.write_text(json.dumps(_fake_causal_report()), encoding="utf-8")

    summary = build_cost_summary(tmp_path)

    assert summary["aggregate"]["num_cases"] == 1
    assert summary["aggregate"]["num_causal_passes"] == 1
    assert summary["aggregate"]["mean_wall_seconds_causal_passes"] == 12.5
    assert summary["cases"][0]["trajectory_reduction_ratio"] == 20.0


def test_risk_export_train_and_repair_admission(tmp_path: Path) -> None:
    report_path = tmp_path / "task8_repeat_01_causal_v1.json"
    report_path.write_text(json.dumps(_fake_causal_report()), encoding="utf-8")
    dataset_path = tmp_path / "risk.jsonl"

    export_summary = export_risk_critic_dataset(tmp_path, dataset_path)
    samples = [json.loads(line) for line in dataset_path.read_text().splitlines()]
    train_summary = train_and_evaluate(samples, seed=0, steps=20)
    repair_summary = build_repair_sft_admission(tmp_path)

    assert export_summary["num_positive"] >= 2
    assert export_summary["num_negative"] >= 2
    assert train_summary["metrics"]["val_auroc"] is not None
    assert repair_summary["num_qualified_repair_pairs"] == 1


def test_full_feature_risk_export_and_grouped_training(tmp_path: Path) -> None:
    for i in range(4):
        report = copy.deepcopy(_fake_causal_report())
        report["risk_training_windows"] = [
            _fake_full_risk_window("minimal_same_failure_slice", 1, float(i)),
            _fake_full_risk_window("successful_rollout_window", 0, float(i) + 10.0),
        ]
        report_path = tmp_path / f"task8_seed_{i}_causal_v1.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")

    dataset_path = tmp_path / "risk_full.jsonl"
    export_summary = export_risk_critic_dataset(
        tmp_path, dataset_path, require_full_features=True
    )
    samples = [json.loads(line) for line in dataset_path.read_text().splitlines()]
    train_summary = train_and_evaluate(
        samples,
        seed=0,
        steps=20,
        feature_set="full_state_action_goal",
        split_by="source_report",
        min_class_count=2,
    )

    assert export_summary["num_samples"] == 8
    assert export_summary["num_positive"] == 4
    assert export_summary["num_negative"] == 4
    assert all(sample["schema_version"] == "risk-critic-full-v1" for sample in samples)
    assert train_summary["status"] == "trained"
    assert train_summary["split"]["group_overlap"] == []
    feature_names = train_summary["model"]["feature_names"]
    assert "raw_action_0_0" in feature_names
    assert "pre_eef_pos_0" in feature_names
    assert "post_goal::On moka_pot_1 flat_stove_1_cook_region" in feature_names


def test_export_from_paths_avoids_old_report_pollution(tmp_path: Path) -> None:
    old_report = copy.deepcopy(_fake_causal_report())
    old_report["risk_training_windows"] = [
        _fake_full_risk_window("minimal_same_failure_slice", 1, 100.0)
    ]
    old_path = tmp_path / "old_causal_v1.json"
    old_path.write_text(json.dumps(old_report), encoding="utf-8")

    new_report = copy.deepcopy(_fake_causal_report())
    new_report["risk_training_windows"] = [
        _fake_full_risk_window("successful_rollout_window", 0, 200.0)
    ]
    new_path = tmp_path / "new_causal_v1.json"
    new_path.write_text(json.dumps(new_report), encoding="utf-8")

    dataset_path = tmp_path / "risk_current.jsonl"
    summary = export_risk_critic_dataset_from_paths(
        [new_path], dataset_path, require_full_features=True, outputs_root=tmp_path
    )
    samples = [json.loads(line) for line in dataset_path.read_text().splitlines()]

    assert summary["source_scope"] == "current_rows"
    assert summary["num_source_reports"] == 1
    assert summary["num_positive"] == 0
    assert summary["num_negative"] == 1
    assert {sample["source_report"] for sample in samples} == {str(new_path)}


def test_v2_necessity_only_is_not_repair_positive_or_sft_pair(tmp_path: Path) -> None:
    report = _fake_v2_necessity_only_report()
    report_path = tmp_path / "task08_init04_seed27_causal_v2.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    dataset_path = tmp_path / "risk_full.jsonl"

    export_summary = export_risk_critic_dataset(
        tmp_path, dataset_path, require_full_features=True
    )
    samples = [json.loads(line) for line in dataset_path.read_text().splitlines()]
    repair_summary = build_repair_sft_admission(tmp_path)

    assert not _strict_causal_pass(report)
    assert export_summary["sample_kinds"]["same_failure_necessity_slice"] == 1
    assert export_summary["sample_kinds"]["same_failure_necessity_core"] == 1
    assert "repair_valid_causal_core" not in export_summary["sample_kinds"]
    assert repair_summary["num_qualified_repair_pairs"] == 0
    assert {sample["sample_kind"] for sample in samples if sample["label"] == 1} == {
        "same_failure_necessity_slice",
        "same_failure_necessity_core",
    }


def test_v2_unvalidated_same_failure_slice_is_not_exported_positive(tmp_path: Path) -> None:
    report = _fake_v2_necessity_only_report()
    report["same_failure_necessity_pass"] = False
    report["causal_validation"]["same_failure_necessity_pass"] = False
    report["causal_validation"]["necessity_core_units"] = []
    report["necessity_core_units"] = []
    report["risk_training_windows"] = [
        _fake_full_risk_window("same_failure_necessity_slice", 1, 1.0),
        _fake_full_risk_window("successful_rollout_window", 0, 10.0),
    ]
    report_path = tmp_path / "same_failure_but_no_necessity_causal_v2.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    dataset_path = tmp_path / "risk_full.jsonl"

    export_summary = export_risk_critic_dataset(
        tmp_path, dataset_path, require_full_features=True
    )
    samples = [json.loads(line) for line in dataset_path.read_text().splitlines()]

    assert export_summary["num_positive"] == 0
    assert export_summary["num_negative"] == 1
    assert {sample["sample_kind"] for sample in samples} == {
        "successful_rollout_window"
    }


def _runner_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=tmp_path,
        report_dir=tmp_path / "reports",
        log_dir=tmp_path / "logs",
        manifest_path=tmp_path / "manifest.jsonl",
        policy_host="127.0.0.1",
        policy_port=65530,
        policy_server_kind="openpi",
        policy_config="pi05_libero",
        policy_dir=Path("/tmp/policy"),
        action_tokenizer_path=None,
        text_tokenizer_path=None,
        allow_hub_download=False,
        pytorch_device="",
        pytorch_compile_mode="",
        cuda_visible_devices="0",
        xla_mem_fraction=0.55,
        task_suite_name="libero_10",
        replay_trials=5,
        search_replay_trials=1,
        confirm_replay_trials=5,
        event_window=32,
        continuation="recorded",
        camera_size=256,
        record_video=False,
        video_dir=None,
        video_camera="agentview_image",
        video_fps=20,
        video_every_n=1,
        video_codec="libx264",
        video_quality=8,
        video_no_flip=False,
        require_video=False,
        resume=True,
        case_timeout_seconds=0.1,
        allow_unverified_policy_server=False,
        launch_policy_server=True,
        dry_run=False,
        policy_ready_timeout=0.1,
    )


def test_return_code_one_is_semantic_nonpass() -> None:
    report = {
        "schema_version": "shed-cfs-causal-v1",
        "feasibility": {"pi05_natural_pass": False},
    }

    assert large_eval._semantic_status_from_report(report, 1) == "semantic_nonpass"


def test_case_timeout_is_recorded_and_continues(tmp_path: Path, monkeypatch) -> None:
    args = _runner_args(tmp_path)

    def slow_command(_args, _report_path, _task_id, _init_state_id, _seed):
        return [sys.executable, "-c", "import time; time.sleep(5)"]

    monkeypatch.setattr(large_eval, "_case_command", slow_command)
    row = large_eval._run_case(args, task_id=8, init_state_id=0, seed=7)

    assert row["status"] == "timeout"
    assert row["return_code"] is None
    assert Path(row["log_path"]).exists()


def test_existing_unverified_policy_port_is_rejected(tmp_path: Path) -> None:
    args = _runner_args(tmp_path)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        args.policy_port = sock.getsockname()[1]
        try:
            large_eval._start_policy_server(args, tmp_path)
        except RuntimeError as exc:
            assert "no matching runner metadata" in str(exc)
        else:
            raise AssertionError("expected unverified port reuse to be rejected")
