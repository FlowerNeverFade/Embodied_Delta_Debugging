from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from risk_critic_export import CAUSAL_SCHEMAS, CAUSAL_V3_SCHEMA, _strict_causal_pass


PROJECT_ROOT = Path("/root/autodl-tmp/research/Embodied_Delta_Debugging")
DEFAULT_SCAN_ROOT = PROJECT_ROOT / "model_datasets"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "v2_pass_search_20260526"


def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _same_failure_necessity_pass(report: dict) -> bool:
    causal = report.get("causal_validation") or {}
    return bool(report.get("same_failure_necessity_pass") or causal.get("same_failure_necessity_pass"))


def _same_failure_pass(report: dict) -> bool:
    repro = report.get("reproduction_statistics") or {}
    return bool(repro.get("same_failure"))


def _full_success_repair_pass(report: dict) -> bool:
    causal = report.get("causal_validation") or {}
    if report.get("schema_version") == CAUSAL_V3_SCHEMA:
        if bool(
            report.get("full_success_policy_repair_pass")
            or causal.get("full_success_policy_repair_pass")
        ):
            return True
        variants = (
            report.get("policy_repair_pass_variants")
            or causal.get("policy_repair_pass_variants")
            or []
        )
        return any(
            bool((item.get("evaluation") or {}).get("success"))
            or bool((item.get("repair_evidence") or {}).get("success"))
            for item in variants
        )
    if bool(report.get("full_success_repair_pass") or causal.get("full_success_repair_pass")):
        return True
    for item in report.get("repair_pass_variants") or causal.get("repair_pass_variants") or []:
        evaluation = item.get("evaluation") or {}
        evidence = item.get("repair_evidence") or {}
        if bool(evaluation.get("success")) or bool(evidence.get("success")):
            return True
    return False


def _repair_improved_only(report: dict) -> bool:
    if not _strict_causal_pass(report) or _full_success_repair_pass(report):
        return False
    causal = report.get("causal_validation") or {}
    return bool(report.get("repair_pass_variants") or causal.get("repair_pass_variants"))


def classify_report(report: dict) -> str:
    if report.get("schema_version") not in CAUSAL_SCHEMAS:
        return "not_causal_report"
    if _strict_causal_pass(report):
        if _full_success_repair_pass(report):
            return "repair_valid_success"
        if _repair_improved_only(report):
            return "repair_valid_improved_only"
        return "repair_valid_improved_only"
    if _same_failure_necessity_pass(report):
        return "necessity_only"
    if _same_failure_pass(report):
        return "same_failure_only"
    return "nonpass"


def _case_summary(path: Path, report: dict, category: str) -> dict:
    selected = report.get("selected_failed_rollout") or {}
    signature = report.get("original_failure_signature") or {}
    causal = report.get("causal_validation") or {}
    repro = report.get("reproduction_statistics") or {}
    repair_variants = report.get("repair_pass_variants") or causal.get("repair_pass_variants") or []
    return {
        "category": category,
        "report_path": str(path),
        "task_suite_name": (report.get("search_config") or {}).get("task_suite_name"),
        "task_id": selected.get("task_id"),
        "init_state_id": selected.get("init_state_id"),
        "task_language": selected.get("task_language"),
        "failure_type": signature.get("failure_type"),
        "failed_goal_predicates": signature.get("failed_goal_predicates") or [],
        "same_failure_rate": repro.get("same_failure_rate"),
        "base_same_failure_rate": causal.get("base_same_failure_rate"),
        "same_failure_necessity_pass": _same_failure_necessity_pass(report),
        "repair_valid_causal_pass": _strict_causal_pass(report),
        "policy_strong_repair_valid_pass": bool(
            report.get("policy_strong_repair_valid_pass")
            or causal.get("policy_strong_repair_valid_pass")
        ),
        "demo_existence_repair_pass": bool(
            report.get("demo_existence_repair_pass")
            or causal.get("demo_existence_repair_pass")
        ),
        "full_success_repair_pass": _full_success_repair_pass(report),
        "repair_sources": sorted({str(item.get("source")) for item in repair_variants if item.get("source")}),
        "repair_success_sources": sorted(
            {
                str(item.get("source"))
                for item in repair_variants
                if bool((item.get("evaluation") or {}).get("success"))
                or bool((item.get("repair_evidence") or {}).get("success"))
            }
        ),
        "video_path": selected.get("video_path"),
    }


def scan_reports(roots: Sequence[Path]) -> dict:
    paths: list[Path] = []
    for root in roots:
        if root.is_file():
            paths.append(root)
        elif root.exists():
            paths.extend(root.glob("**/reports/*_causal_v2.json"))
            paths.extend(root.glob("**/*_causal_v2.json"))
            paths.extend(root.glob("**/reports/*_causal_v3.json"))
            paths.extend(root.glob("**/*_causal_v3.json"))
    seen = set()
    unique = []
    for path in sorted(paths):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)

    categories = {
        "repair_valid_success": [],
        "repair_valid_improved_only": [],
        "necessity_only": [],
        "same_failure_only": [],
        "nonpass": [],
        "not_causal_report": [],
        "unreadable": [],
    }
    for path in unique:
        report = _load_json(path)
        if not isinstance(report, dict):
            categories["unreadable"].append({"report_path": str(path)})
            continue
        category = classify_report(report)
        categories.setdefault(category, []).append(_case_summary(path, report, category))
    return {
        "schema_version": "causal-pass-candidate-index-v2-v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scan_roots": [str(root) for root in roots],
        "num_reports_scanned": len(unique),
        "counts": {key: len(value) for key, value in categories.items()},
        "categories": categories,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan SHED-CFS causal-v2/v3 reports and classify repair-valid passes.")
    parser.add_argument("--scan-root", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    if args.scan_root is None:
        args.scan_root = [DEFAULT_SCAN_ROOT]
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = scan_reports(args.scan_root)
    out = args.output_dir / "candidate_index.json"
    out.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(index["counts"], indent=2, ensure_ascii=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
