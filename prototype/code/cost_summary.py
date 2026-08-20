from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional, Sequence

from risk_critic_export import _strict_causal_pass
from risk_critic_export import CAUSAL_SCHEMAS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUTS_ROOT / "cost_summary.json"


def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _repeat_id(path: Path) -> Optional[str]:
    match = re.search(r"repeat[_-]?(\d+)", path.name)
    if match:
        return match.group(1)
    return None


def _matching_log(path: Path) -> Optional[Path]:
    repeat_id = _repeat_id(path)
    if repeat_id is None:
        return None
    candidates = [
        path.parent / "logs" / f"repeat_{repeat_id}.log",
        path.parent / "logs" / f"repeat_{int(repeat_id):02d}.log",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _candidate_span(report: dict) -> Optional[list]:
    stats = report.get("reproduction_statistics") or {}
    candidate = stats.get("candidate") or report.get("causal_failure_slice") or {}
    return candidate.get("span")


def _case_summary(path: Path, report: dict) -> dict:
    rollout = (report.get("rollout_summaries") or [{}])[0]
    selected = report.get("selected_failed_rollout")
    reproduction = report.get("reproduction_statistics") or {}
    causal_validation = report.get("causal_validation") or {}
    metrics = report.get("metrics") or {}
    runtime = report.get("runtime_profile") or {}
    cost = report.get("cost_summary") or {}
    log_path = _matching_log(path)
    log_stat = log_path.stat() if log_path is not None else None
    full_pass = bool((report.get("feasibility") or {}).get("pi05_natural_pass"))
    same_failure_pass = bool(reproduction.get("same_failure"))
    causal_pass = bool(_strict_causal_pass(report))

    return {
        "path": str(path),
        "schema_version": report.get("schema_version"),
        "created_at": report.get("created_at"),
        "task_suite_name": (report.get("search_config") or {}).get("task_suite_name"),
        "task_ids": (report.get("search_config") or {}).get("task_ids"),
        "init_state_ids": (report.get("search_config") or {}).get("init_state_ids"),
        "replay_trials": (report.get("search_config") or {}).get("replay_trials"),
        "search_replay_trials": (report.get("search_config") or {}).get("search_replay_trials"),
        "confirm_replay_trials": (report.get("search_config") or {}).get("confirm_replay_trials"),
        "rollout_success": rollout.get("success"),
        "natural_failure_found": bool(selected is not None and not selected.get("success", True)),
        "same_failure_pass": same_failure_pass,
        "causal_validation_pass": causal_pass,
        "full_pi05_natural_pass": full_pass,
        "failure_type": (report.get("original_failure_signature") or {}).get("failure_type"),
        "failed_goal_predicates": (
            report.get("original_failure_signature") or {}
        ).get("failed_goal_predicates"),
        "failure_event": (report.get("failure_event") or {}).get("window"),
        "minimal_slice": _candidate_span(report),
        "same_failure_rate": reproduction.get("same_failure_rate"),
        "failure_rate": reproduction.get("failure_rate"),
        "trajectory_reduction_ratio": metrics.get(
            "trajectory_reduction_ratio", cost.get("trajectory_reduction_ratio")
        ),
        "event_reduction_ratio": metrics.get(
            "event_reduction_ratio", cost.get("event_reduction_ratio")
        ),
        "replay_evaluations": metrics.get(
            "replay_evaluations", cost.get("replay_evaluations")
        ),
        "runtime_wall_seconds": runtime.get("total_wall_seconds", cost.get("total_wall_seconds")),
        "runtime_durations_seconds": runtime.get(
            "durations_seconds", cost.get("durations_seconds")
        ),
        "policy_queries": cost.get("policy_queries"),
        "env_resets": cost.get("env_resets"),
        "simulator_suffix_steps": cost.get("simulator_suffix_steps_measured"),
        "causal_validation_skipped_reason": report.get(
            "causal_validation_skipped_reason",
            cost.get("causal_validation_skipped_reason"),
        ),
        "log_path": None if log_path is None else str(log_path),
        "log_size_bytes": None if log_stat is None else int(log_stat.st_size),
        "log_mtime": None if log_stat is None else float(log_stat.st_mtime),
        "legacy_runtime_available": bool(runtime or cost.get("total_wall_seconds") is not None),
    }


def collect_cost_cases(outputs_root: Path, pattern: str = "**/*causal_v*.json") -> list:
    cases = []
    for path in sorted(outputs_root.glob(pattern)):
        report = _load_json(path)
        if not isinstance(report, dict):
            continue
        if report.get("schema_version") not in CAUSAL_SCHEMAS:
            continue
        cases.append(_case_summary(path, report))
    return cases


def collect_cost_cases_from_paths(report_paths: Iterable[Path]) -> list:
    cases = []
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
        if report.get("schema_version") not in CAUSAL_SCHEMAS:
            continue
        cases.append(_case_summary(path, report))
    return cases


def _average(values: Iterable[object]) -> Optional[float]:
    cleaned = [float(v) for v in values if isinstance(v, (int, float))]
    if not cleaned:
        return None
    return float(mean(cleaned))


def summarize_costs(cases: Sequence[dict]) -> dict:
    causal_passes = [case for case in cases if case["full_pi05_natural_pass"]]
    natural_failures = [case for case in cases if case["natural_failure_found"]]
    semantic_passes = [case for case in cases if case["same_failure_pass"]]
    return {
        "num_cases": len(cases),
        "num_natural_failures": len(natural_failures),
        "num_same_failure_passes": len(semantic_passes),
        "num_causal_passes": len(causal_passes),
        "mean_wall_seconds_all_with_runtime": _average(
            case["runtime_wall_seconds"] for case in cases
        ),
        "mean_wall_seconds_natural_failures": _average(
            case["runtime_wall_seconds"] for case in natural_failures
        ),
        "mean_wall_seconds_causal_passes": _average(
            case["runtime_wall_seconds"] for case in causal_passes
        ),
        "mean_trajectory_reduction_same_failure": _average(
            case["trajectory_reduction_ratio"] for case in semantic_passes
        ),
        "mean_replay_evaluations_natural_failures": _average(
            case["replay_evaluations"] for case in natural_failures
        ),
        "legacy_cases_without_runtime": sum(
            1 for case in cases if not case["legacy_runtime_available"]
        ),
    }


def build_cost_summary(outputs_root: Path, pattern: str = "**/*causal_v*.json") -> dict:
    cases = collect_cost_cases(outputs_root, pattern=pattern)
    return {
        "schema_version": "shed-cfs-cost-summary-v1",
        "source_scope": "report_dir",
        "outputs_root": str(outputs_root),
        "pattern": pattern,
        "aggregate": summarize_costs(cases),
        "cases": cases,
    }


def build_cost_summary_from_paths(report_paths: Iterable[Path], outputs_root: Optional[Path] = None) -> dict:
    paths = [Path(path) for path in report_paths]
    cases = collect_cost_cases_from_paths(paths)
    return {
        "schema_version": "shed-cfs-cost-summary-v1",
        "source_scope": "current_rows",
        "outputs_root": None if outputs_root is None else str(outputs_root),
        "pattern": None,
        "num_source_reports": len({str(path) for path in paths}),
        "excluded_reports": max(0, len({str(path) for path in paths}) - len(cases)),
        "aggregate": summarize_costs(cases),
        "cases": cases,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SHED-CFS causal-v1 costs.")
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--pattern", default="**/*causal_v*.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary = build_cost_summary(args.outputs_root, pattern=args.pattern)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
