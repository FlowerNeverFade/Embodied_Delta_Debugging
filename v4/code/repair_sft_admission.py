from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from risk_critic_export import CAUSAL_SCHEMAS, _repair_pass


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT = DEFAULT_OUTPUTS_ROOT / "repair_sft_admission.json"


def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _candidate_repair_pairs(outputs_root: Path) -> list[dict]:
    pairs = []
    for path in sorted(outputs_root.rglob("*causal_v*.json")):
        report = _load_json(path)
        if not isinstance(report, dict) or report.get("schema_version") not in CAUSAL_SCHEMAS:
            continue
        base_rate = float(
            ((report.get("causal_validation") or {}).get("base_same_failure_rate"))
            or ((report.get("reproduction_statistics") or {}).get("same_failure_rate"))
            or 0.0
        )
        for variant in (
            report.get("repair_pass_variants")
            or report.get("counterfactual_pass_variants")
            or []
        ):
            evaluation = variant.get("evaluation") or {}
            ablated_rate = float(evaluation.get("same_failure_rate") or 0.0)
            success = bool(evaluation.get("success"))
            lowers_failure = ablated_rate < base_rate
            repair_pass = _repair_pass(
                report,
                evaluation,
                variant.get("repair_pass"),
                variant.get("repair_evidence"),
            )
            qualifies = bool(lowers_failure and repair_pass and (success or ablated_rate <= 0.2))
            pairs.append(
                {
                    "source_report": str(path),
                    "strategy": variant.get("strategy"),
                    "unit_id": variant.get("unit_id"),
                    "candidate": evaluation.get("candidate"),
                    "base_same_failure_rate": base_rate,
                    "counterfactual_same_failure_rate": ablated_rate,
                    "counterfactual_success": success,
                    "lowers_same_failure": lowers_failure,
                    "repair_pass": repair_pass,
                    "qualifies_for_repair_sft": qualifies,
                    "reason": (
                        "success_counterfactual"
                        if success
                        else "counterfactual_worsens_or_does_not_improve_goals"
                        if not repair_pass
                        else "same_failure_rate_below_0.2"
                        if qualifies
                        else "does_not_reduce_same_failure_enough"
                    ),
                }
            )
    return pairs


def build_repair_sft_admission(outputs_root: Path) -> dict:
    pairs = _candidate_repair_pairs(outputs_root)
    qualified = [pair for pair in pairs if pair["qualifies_for_repair_sft"]]
    return {
        "schema_version": "repair-sft-admission-v1",
        "outputs_root": str(outputs_root),
        "num_counterfactual_variants": len(pairs),
        "num_qualified_repair_pairs": len(qualified),
        "repair_action_sources_priority": [
            "scripted expert local repair for curriculum tasks",
            "successful counterfactual",
            "same task/init successful rollout aligned by state distance",
            "LIBERO demo nearest-neighbor action",
        ],
        "admission_policy": (
            "Do not SFT on failed actions. A pair qualifies only when the replacement "
            "counterfactual lowers same-failure rate and either reaches success or "
            "pushes same-failure rate to <= 0.2."
        ),
        "qualified_pairs": qualified,
        "all_pairs": pairs,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Repair SFT admission candidates.")
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary = build_repair_sft_admission(args.outputs_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "num_counterfactual_variants": summary["num_counterfactual_variants"],
                "num_qualified_repair_pairs": summary["num_qualified_repair_pairs"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
