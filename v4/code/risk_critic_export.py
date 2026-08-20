from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence


DEFAULT_OUTPUTS_ROOT = Path(
    "/root/autodl-tmp/research/Embodied_Delta_Debugging/outputs"
)
DEFAULT_OUTPUT = DEFAULT_OUTPUTS_ROOT / "risk_critic_v1.jsonl"
CAUSAL_V2_SCHEMA = "shed-cfs-causal-v2-split-repair"
CAUSAL_V3_SCHEMA = "shed-cfs-causal-v3-multimodal"
CAUSAL_V4_SCHEMA = "shed-cfs-causal-v4-global-multimodal"
CAUSAL_SCHEMAS = {
    "shed-cfs-causal-v1",
    CAUSAL_V2_SCHEMA,
    CAUSAL_V3_SCHEMA,
    CAUSAL_V4_SCHEMA,
}


def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _goal_trace(signature: dict) -> dict:
    evidence = signature.get("evidence") or {}
    trace = evidence.get("goal_trace") if isinstance(evidence, dict) else {}
    return trace if isinstance(trace, dict) else {}


def _failed_goal_count(signature: dict) -> int:
    trace = _goal_trace(signature)
    failed = trace.get("failed_final_predicates")
    if isinstance(failed, list):
        return len(failed)
    return len(signature.get("failed_goal_predicates") or [])


def _goal_progress(signature: dict) -> int:
    trace = _goal_trace(signature)
    progress = trace.get("progress_counts")
    if isinstance(progress, list) and progress:
        return int(progress[-1])
    truth = trace.get("final_truth")
    if isinstance(truth, dict):
        return sum(1 for value in truth.values() if bool(value))
    return 0


def _repair_pass(
    report: dict,
    evaluation: dict,
    explicit: object = None,
    repair_evidence: object = None,
) -> bool:
    if isinstance(repair_evidence, dict):
        evidence_flag = repair_evidence.get("repair_pass")
        if evidence_flag is False:
            return False
        if report.get("schema_version") == CAUSAL_V2_SCHEMA and evidence_flag is True:
            return True
        if report.get("schema_version") in {CAUSAL_V3_SCHEMA, CAUSAL_V4_SCHEMA} and evidence_flag is True:
            return bool(
                repair_evidence.get("policy_repair_pass")
                or repair_evidence.get("policy_strong_repair_pass")
                or repair_evidence.get("raw_policy_repair_pass")
                or repair_evidence.get("language_phrase_repair_pass")
                or repair_evidence.get("visual_mask_repair_pass")
            )
    if explicit is False:
        return False
    if report.get("schema_version") == CAUSAL_V2_SCHEMA and explicit is True:
        return True
    if report.get("schema_version") in {CAUSAL_V3_SCHEMA, CAUSAL_V4_SCHEMA} and explicit is True:
        return False
    reference = report.get("original_failure_signature") or {}
    candidate = evaluation.get("failure_signature") or {}
    if not isinstance(reference, dict) or not isinstance(candidate, dict):
        return False
    base_failed_set = set(reference.get("failed_goal_predicates") or [])
    cf_failed_set = set(candidate.get("failed_goal_predicates") or [])
    ref_trace = _goal_trace(reference)
    cf_trace = _goal_trace(candidate)
    ref_failed = ref_trace.get("failed_final_predicates")
    cf_failed = cf_trace.get("failed_final_predicates")
    if isinstance(ref_failed, list):
        base_failed_set = {str(item) for item in ref_failed}
    if isinstance(cf_failed, list):
        cf_failed_set = {str(item) for item in cf_failed}
    base_failed = _failed_goal_count(reference)
    cf_failed = _failed_goal_count(candidate)
    base_progress = _goal_progress(reference)
    cf_progress = _goal_progress(candidate)
    affected_growth = max(
        0,
        len(set(candidate.get("affected_objects") or []))
        - len(set(reference.get("affected_objects") or [])),
    )
    non_worsening = (
        cf_failed <= base_failed
        and cf_progress >= base_progress
        and cf_failed_set.issubset(base_failed_set)
        and affected_growth == 0
    )
    improved = bool(evaluation.get("success")) or cf_failed < base_failed or cf_progress > base_progress
    return bool(evaluation.get("success") or (non_worsening and improved))


def _strict_causal_pass(report: dict) -> bool:
    causal = report.get("causal_validation") or {}
    if report.get("schema_version") == CAUSAL_V4_SCHEMA:
        if not bool(
            report.get("any_policy_repair_valid_pass")
            or report.get("policy_raw_repair_valid_pass")
            or report.get("policy_language_phrase_repair_valid_pass")
            or report.get("policy_visual_mask_repair_valid_pass")
            or causal.get("raw_policy_repair_valid_pass")
            or causal.get("language_phrase_repair_valid_pass")
            or causal.get("visual_mask_repair_valid_pass")
            or causal.get("repair_valid_causal_pass")
        ):
            return False
        return bool(
            report.get("raw_policy_core_units")
            or report.get("language_phrase_core_units")
            or report.get("visual_mask_core_units")
            or causal.get("raw_policy_core_units")
            or causal.get("language_phrase_core_units")
            or causal.get("visual_mask_core_units")
            or report.get("causal_core_units")
            or causal.get("causal_core_units")
        )
    if report.get("schema_version") == CAUSAL_V3_SCHEMA:
        if not bool(
            report.get("policy_strong_repair_valid_pass")
            or causal.get("policy_strong_repair_valid_pass")
            or report.get("repair_valid_causal_pass")
            or causal.get("repair_valid_causal_pass")
        ):
            return False
        return bool(
            report.get("policy_strong_causal_core_units")
            or causal.get("policy_strong_core_units")
            or report.get("causal_core_units")
            or causal.get("causal_core_units")
        )
    if report.get("schema_version") == CAUSAL_V2_SCHEMA:
        if not bool(
            report.get("repair_valid_causal_pass")
            or causal.get("repair_valid_causal_pass")
            or causal.get("passed")
        ):
            return False
        return bool(report.get("causal_core_units") or causal.get("causal_core_units"))
    if not bool(causal.get("passed")):
        return False
    for unit in report.get("causal_core_units") or []:
        best = unit.get("best_counterfactual") or {}
        evaluation = best.get("evaluation") or {}
        if _repair_pass(
            report,
            evaluation,
            None,
            unit.get("repair_evidence") or best.get("repair_evidence"),
        ):
            return True
    return False


def _same_failure_necessity_pass(report: dict) -> bool:
    causal = report.get("causal_validation") or {}
    return bool(
        report.get("same_failure_necessity_pass")
        or causal.get("same_failure_necessity_pass")
    )


def _strict_unit_pass(report: dict, unit: dict) -> bool:
    if report.get("schema_version") == CAUSAL_V4_SCHEMA:
        evidence = unit.get("repair_evidence") or {}
        return bool(
            unit.get("raw_policy_repair_pass")
            or unit.get("language_phrase_repair_pass")
            or unit.get("visual_mask_repair_pass")
            or evidence.get("raw_policy_repair_pass")
            or evidence.get("language_phrase_repair_pass")
            or evidence.get("visual_mask_repair_pass")
            or evidence.get("policy_repair_pass")
        )
    if report.get("schema_version") == CAUSAL_V3_SCHEMA:
        evidence = unit.get("repair_evidence") or {}
        return bool(
            unit.get("policy_strong_repair_pass")
            or evidence.get("policy_repair_pass")
        )
    best = unit.get("best_counterfactual") or {}
    evaluation = best.get("evaluation") or {}
    return _repair_pass(
        report,
        evaluation,
        unit.get("repair_pass", best.get("repair_pass")),
        unit.get("repair_evidence") or best.get("repair_evidence"),
    )


def _strict_window_unit_pass(report: dict, unit: dict) -> bool:
    unit_id = unit.get("unit_id") if isinstance(unit, dict) else None
    report_units = []
    for key in (
        "causal_core_units",
        "raw_policy_core_units",
        "language_phrase_core_units",
        "visual_mask_core_units",
        "demo_existence_core_units",
    ):
        report_units.extend(report.get(key) or [])
    for report_unit in report_units:
        report_unit_id = (report_unit.get("unit") or {}).get("unit_id")
        if unit_id is None or report_unit_id == unit_id:
            return _strict_unit_pass(report, report_unit)
    return False


def _base_sample(path: Path, report: dict, sample_kind: str, label: int) -> dict:
    search_config = report.get("search_config") or {}
    selected = report.get("selected_failed_rollout") or {}
    rollout = (report.get("rollout_summaries") or [{}])[0]
    signature = report.get("original_failure_signature") or {}
    task_id = selected.get("task_id", rollout.get("task_id"))
    init_state_id = selected.get("init_state_id", rollout.get("init_state_id"))
    task_language = selected.get("task_language", rollout.get("task_language"))
    return {
        "schema_version": "risk-critic-v1",
        "source_report": str(path),
        "source_schema_version": report.get("schema_version"),
        "sample_kind": sample_kind,
        "label": int(label),
        "task": {
            "task_suite_name": search_config.get("task_suite_name"),
            "task_id": task_id,
            "init_state_id": init_state_id,
            "task_language": task_language,
        },
        "failure_signature": {
            "failure_type": signature.get("failure_type"),
            "failed_goal_predicates": signature.get("failed_goal_predicates", []),
            "affected_objects": signature.get("affected_objects", []),
            "semantic_quality": signature.get("semantic_quality"),
        },
    }


def _window_sample(path: Path, report: dict, window: dict, require_full_features: bool) -> Optional[dict]:
    features = window.get("features") or {}
    if require_full_features and features.get("feature_quality") != "full":
        return None
    task = window.get("task") or {}
    failure_signature = window.get("failure_signature") or report.get("original_failure_signature") or {}
    sample_kind = str(window.get("sample_kind") or "risk_window")
    if sample_kind in {"same_failure_necessity_slice", "same_failure_necessity_core"}:
        if (
            report.get("schema_version") in {CAUSAL_V2_SCHEMA, CAUSAL_V3_SCHEMA, CAUSAL_V4_SCHEMA}
            and int(window.get("label") or 0) == 1
            and not _same_failure_necessity_pass(report)
        ):
            return None
    if sample_kind in {
        "causal_core_unit",
        "repair_valid_causal_core",
        "policy_strong_repair_valid_core",
        "policy_raw_repair_valid_core",
        "policy_language_phrase_repair_valid_core",
        "policy_visual_mask_repair_valid_core",
        "demo_existence_repair_valid_core",
    }:
        unit = window.get("causal_unit") or {}
        if not _strict_window_unit_pass(report, unit if isinstance(unit, dict) else {}):
            return None
    if sample_kind == "counterfactual_lower_risk_slice":
        repair_evidence = window.get("repair_evidence") or {}
        if isinstance(repair_evidence, dict) and repair_evidence.get("repair_pass") is False:
            return None
    sample = _base_sample(path, report, sample_kind, int(window.get("label") or 0))
    sample.update(
        {
            "schema_version": "risk-critic-full-v1"
            if features.get("feature_quality") == "full"
            else "risk-critic-v1",
            "sample_id": window.get("sample_id"),
            "label_source": window.get("label_source"),
            "task": {
                "task_suite_name": task.get("task_suite_name"),
                "task_id": task.get("task_id"),
                "init_state_id": task.get("init_state_id"),
                "task_language": task.get("task_language"),
            },
            "failure_signature": {
                "failure_type": failure_signature.get("failure_type"),
                "failed_goal_predicates": failure_signature.get("failed_goal_predicates", []),
                "affected_objects": failure_signature.get("affected_objects", []),
                "semantic_quality": failure_signature.get("semantic_quality"),
            },
            "candidate": window.get("candidate"),
            "features": features,
            "same_failure_rate": window.get("same_failure_rate"),
            "failure_rate": window.get("failure_rate"),
            "causal_effect": window.get("causal_effect"),
            "causal_validation_passed": bool(_strict_causal_pass(report)),
            "source_report": str(path),
        }
    )
    for key in ("causal_unit", "counterfactual_strategy", "label_source"):
        if key in window:
            sample[key] = window.get(key)
    return sample


def _sample_from_causal_report(path: Path, report: dict) -> Iterable[dict]:
    risk_windows = report.get("risk_training_windows") or []
    if risk_windows:
        for window in risk_windows:
            sample = _window_sample(path, report, window, require_full_features=False)
            if sample is not None:
                yield sample
        return

    reproduction = report.get("reproduction_statistics") or {}
    candidate = reproduction.get("candidate") or report.get("causal_failure_slice")
    training_features = report.get("slice_training_features") or {
        "feature_quality": "metadata_only",
        "candidate_actions": None,
        "candidate_actions_truncated": None,
        "action_summary": None,
        "pre_state_features": None,
        "post_state_features": None,
    }
    same_rate = float(reproduction.get("same_failure_rate") or 0.0)
    same_failure = bool(reproduction.get("same_failure"))
    causal_validation = report.get("causal_validation") or {}
    causal_pass = bool(_strict_causal_pass(report))
    necessity_pass = (
        _same_failure_necessity_pass(report)
        if report.get("schema_version") in {CAUSAL_V2_SCHEMA, CAUSAL_V3_SCHEMA, CAUSAL_V4_SCHEMA}
        else bool(same_failure)
    )
    base_kind = (
        "same_failure_necessity_slice"
        if necessity_pass
        else "non_matching_candidate_slice"
    )
    base_label = 1 if same_rate >= 0.8 and same_failure and necessity_pass else 0
    sample = _base_sample(path, report, base_kind, base_label)
    sample.update(
        {
            "candidate": candidate,
            "features": training_features,
            "same_failure_rate": same_rate,
            "failure_rate": reproduction.get("failure_rate"),
            "causal_effect": None,
            "causal_validation_passed": causal_pass,
        }
    )
    if candidate is not None:
        yield sample

    for unit in report.get("necessity_core_units") or []:
        unit_sample = _base_sample(path, report, "same_failure_necessity_core", 1)
        unit_sample.update(
            {
                "candidate": {
                    "level": "same_failure_necessity_core",
                    "span": unit.get("unit", {}).get("interval"),
                    "unit": unit.get("unit"),
                },
                "features": {
                    "feature_quality": training_features.get(
                        "feature_quality", "metadata_only"
                    ),
                    "parent_slice_features": training_features,
                },
                "same_failure_rate": unit.get("base_same_failure_rate"),
                "failure_rate": None,
                "causal_effect": unit.get("causal_effect"),
                "causal_validation_passed": causal_pass,
            }
        )
        yield unit_sample

    repair_unit_groups = [("repair_valid_causal_core", report.get("causal_core_units") or [])]
    if report.get("schema_version") == CAUSAL_V3_SCHEMA:
        repair_unit_groups = [
            ("policy_strong_repair_valid_core", report.get("causal_core_units") or [])
        ]
    elif report.get("schema_version") == CAUSAL_V4_SCHEMA:
        repair_unit_groups = [
            ("policy_raw_repair_valid_core", report.get("raw_policy_core_units") or []),
            (
                "policy_language_phrase_repair_valid_core",
                report.get("language_phrase_core_units") or [],
            ),
            (
                "policy_visual_mask_repair_valid_core",
                report.get("visual_mask_core_units") or [],
            ),
            (
                "demo_existence_repair_valid_core",
                report.get("demo_existence_core_units") or [],
            ),
        ]
    for sample_kind, units in repair_unit_groups:
        for unit in units:
            if report.get("schema_version") != CAUSAL_V2_SCHEMA and not _strict_unit_pass(report, unit):
                continue
            unit_sample = _base_sample(path, report, sample_kind, 1)
            unit_sample.update(
                {
                    "candidate": {
                        "level": sample_kind,
                        "span": unit.get("unit", {}).get("interval"),
                        "unit": unit.get("unit"),
                    },
                    "features": {
                        "feature_quality": training_features.get(
                            "feature_quality", "metadata_only"
                        ),
                        "parent_slice_features": training_features,
                    },
                    "same_failure_rate": unit.get("base_same_failure_rate"),
                    "failure_rate": None,
                    "causal_effect": unit.get("causal_effect"),
                    "causal_validation_passed": causal_pass,
                    "repair_evidence": unit.get("repair_evidence"),
                }
            )
            yield unit_sample

    for variant in (
        report.get("repair_pass_variants")
        or report.get("counterfactual_pass_variants")
        or []
    ):
        evaluation = variant.get("evaluation") or {}
        if not _repair_pass(
            report,
            evaluation,
            variant.get("repair_pass"),
            variant.get("repair_evidence"),
        ):
            continue
        base_rate = float(causal_validation.get("base_same_failure_rate") or same_rate)
        ablated_rate = float(evaluation.get("same_failure_rate") or 0.0)
        if ablated_rate >= base_rate:
            continue
        cf_sample = _base_sample(path, report, "counterfactual_lower_risk_slice", 0)
        cf_sample.update(
            {
                "candidate": evaluation.get("candidate"),
                "features": {
                    "feature_quality": "metadata_only",
                    "counterfactual_strategy": variant.get("strategy"),
                    "unit_id": variant.get("unit_id"),
                },
                "same_failure_rate": ablated_rate,
                "failure_rate": evaluation.get("failure_rate"),
                "causal_effect": float(base_rate - ablated_rate),
                "causal_validation_passed": causal_pass,
            }
        )
        yield cf_sample

    for rollout in report.get("rollout_summaries") or []:
        if not rollout.get("success"):
            continue
        event = rollout.get("max_window_event")
        neg = _base_sample(path, report, "successful_rollout_window", 0)
        neg["task"] = {
            "task_suite_name": (report.get("search_config") or {}).get("task_suite_name"),
            "task_id": rollout.get("task_id"),
            "init_state_id": rollout.get("init_state_id"),
            "task_language": rollout.get("task_language"),
        }
        neg.update(
            {
                "candidate": None if event is None else {"level": "success_window", "span": event[:2]},
                "features": {"feature_quality": "metadata_only"},
                "same_failure_rate": 0.0,
                "failure_rate": 0.0,
                "causal_effect": None,
                "causal_validation_passed": False,
            }
        )
        yield neg


def _sample_from_offline_report(path: Path, report: dict) -> Iterable[dict]:
    reproduction = report.get("reproduction_statistics") or {}
    metrics = report.get("causal_metrics") or {}
    task_env = report.get("task_policy_environment") or {}
    label = 1 if float(reproduction.get("same_failure_rate") or 0.0) >= 0.8 else 0
    sample = {
        "schema_version": "risk-critic-v1",
        "source_report": str(path),
        "source_schema_version": report.get("schema_version"),
        "sample_kind": "offline_injected_failure_slice",
        "label": label,
        "task": {
            "task_suite_name": task_env.get("dataset_name", "offline_lerobot"),
            "task_id": task_env.get("task_index"),
            "init_state_id": task_env.get("episode_index"),
            "task_language": task_env.get("task_language"),
        },
        "failure_signature": {
            "failure_type": (
                report.get("controlled_failure_injection", {})
                .get("spec", {})
                .get("failure_type")
            ),
            "failed_goal_predicates": [],
            "affected_objects": [],
            "semantic_quality": "offline_injected",
        },
        "candidate": reproduction.get("candidate") or report.get("causal_failure_slice"),
        "features": {
            "feature_quality": "metadata_only",
            "controlled_failure_injection": report.get("controlled_failure_injection"),
        },
        "same_failure_rate": reproduction.get("same_failure_rate"),
        "failure_rate": reproduction.get("failure_rate"),
        "causal_effect": metrics.get("causal_effect_score"),
        "causal_validation_passed": label == 1,
    }
    yield sample

    counterfactual = report.get("counterfactual_pass_variant") or {}
    evaluation = counterfactual.get("evaluation") or {}
    if evaluation:
        cf_sample = dict(sample)
        cf_sample.update(
            {
                "sample_kind": "offline_counterfactual_lower_risk_slice",
                "label": 0,
                "candidate": evaluation.get("candidate"),
                "features": {
                    "feature_quality": "metadata_only",
                    "counterfactual_description": counterfactual.get("description"),
                },
                "same_failure_rate": evaluation.get("same_failure_rate"),
                "failure_rate": evaluation.get("failure_rate"),
                "causal_validation_passed": False,
            }
        )
        yield cf_sample


def iter_risk_samples(outputs_root: Path, require_full_features: bool = False) -> Iterable[dict]:
    for path in sorted(outputs_root.rglob("*.json")):
        report = _load_json(path)
        if not isinstance(report, dict):
            continue
        schema = report.get("schema_version")
        if schema in CAUSAL_SCHEMAS:
            for sample in _sample_from_causal_report(path, report):
                if require_full_features and (
                    sample.get("features") or {}
                ).get("feature_quality") != "full":
                    continue
                if require_full_features:
                    sample = dict(sample)
                    sample["schema_version"] = "risk-critic-full-v1"
                yield sample
        elif schema == "shed-cfs-offline-v0":
            for sample in _sample_from_offline_report(path, report):
                if require_full_features and (
                    sample.get("features") or {}
                ).get("feature_quality") != "full":
                    continue
                yield sample


def iter_risk_samples_from_paths(
    report_paths: Iterable[Path], require_full_features: bool = False
) -> Iterable[dict]:
    seen = set()
    for path in report_paths:
        path = Path(path)
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        report = _load_json(path)
        if not isinstance(report, dict):
            continue
        schema = report.get("schema_version")
        if schema in CAUSAL_SCHEMAS:
            for sample in _sample_from_causal_report(path, report):
                if require_full_features and (
                    sample.get("features") or {}
                ).get("feature_quality") != "full":
                    continue
                if require_full_features:
                    sample = dict(sample)
                    sample["schema_version"] = "risk-critic-full-v1"
                yield sample
        elif schema == "shed-cfs-offline-v0":
            for sample in _sample_from_offline_report(path, report):
                if require_full_features and (
                    sample.get("features") or {}
                ).get("feature_quality") != "full":
                    continue
                yield sample


def export_risk_critic_dataset(
    outputs_root: Path, output_path: Path, require_full_features: bool = False
) -> dict:
    samples = list(iter_risk_samples(outputs_root, require_full_features=require_full_features))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    positives = sum(1 for sample in samples if sample["label"] == 1)
    negatives = sum(1 for sample in samples if sample["label"] == 0)
    return {
        "schema_version": "risk-critic-export-summary-v1",
        "source_scope": "report_dir",
        "outputs_root": str(outputs_root),
        "output_path": str(output_path),
        "num_samples": len(samples),
        "num_positive": positives,
        "num_negative": negatives,
        "require_full_features": bool(require_full_features),
        "sample_kinds": {
            kind: sum(1 for sample in samples if sample["sample_kind"] == kind)
            for kind in sorted({sample["sample_kind"] for sample in samples})
        },
    }


def export_risk_critic_dataset_from_paths(
    report_paths: Iterable[Path],
    output_path: Path,
    require_full_features: bool = False,
    outputs_root: Optional[Path] = None,
) -> dict:
    paths = [Path(path) for path in report_paths]
    samples = list(
        iter_risk_samples_from_paths(paths, require_full_features=require_full_features)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    positives = sum(1 for sample in samples if sample["label"] == 1)
    negatives = sum(1 for sample in samples if sample["label"] == 0)
    unique_paths = {str(path) for path in paths}
    exported_reports = {sample["source_report"] for sample in samples}
    return {
        "schema_version": "risk-critic-export-summary-v1",
        "source_scope": "current_rows",
        "outputs_root": None if outputs_root is None else str(outputs_root),
        "output_path": str(output_path),
        "num_source_reports": len(unique_paths),
        "excluded_reports": max(0, len(unique_paths) - len(exported_reports)),
        "num_samples": len(samples),
        "num_positive": positives,
        "num_negative": negatives,
        "require_full_features": bool(require_full_features),
        "sample_kinds": {
            kind: sum(1 for sample in samples if sample["sample_kind"] == kind)
            for kind in sorted({sample["sample_kind"] for sample in samples})
        },
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SHED-CFS risk critic JSONL.")
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-full-features", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary = export_risk_critic_dataset(
        args.outputs_root, args.output, require_full_features=args.require_full_features
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
