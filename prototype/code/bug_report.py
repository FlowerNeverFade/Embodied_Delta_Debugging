from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from data_probe import EpisodeData
from failure_oracle import InjectionResult, StochasticFailureOracle
from shed_minimizer import MinimizationResult


def build_bug_report(
    dataset_root: Path,
    episode: EpisodeData,
    task_language: str,
    injection: InjectionResult,
    minimization: MinimizationResult,
    oracle: StochasticFailureOracle,
    causal_effect_trials: int = 32,
    dataset_name: str = "LeRobot dataset",
) -> dict:
    final_slice = minimization.minimal_slice
    reduction_ratio = (
        float(episode.length / final_slice.length) if final_slice.length > 0 else None
    )
    causal_effect = oracle.causal_effect(
        final_slice,
        remove_intervals=[injection.spec.window],
        trials=causal_effect_trials,
    )
    without_injected = final_slice.remove([injection.spec.window])
    counterfactual_eval = oracle.evaluate(
        without_injected, trials=causal_effect_trials
    )

    return {
        "schema_version": "shed-cfs-offline-v0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_policy_environment": {
            "dataset_root": str(dataset_root),
            "dataset_name": dataset_name,
            "dataset_format": "LeRobot parquet",
            "policy_source": "successful demonstration with controlled failure injection",
            "environment": "offline surrogate; simulator smoke test is separate",
            "task_index": int(episode.task_index),
            "task_language": task_language,
            "episode_index": int(episode.episode_index),
        },
        "failure_predicate": {
            "type": "same injected failure reproduced by stochastic oracle",
            "failure_type": injection.spec.failure_type,
            "acceptance": "same_failure_rate >= minimizer threshold",
            "required_overlap": injection.spec.required_overlap,
        },
        "original_rollout": {
            "length": int(episode.length),
            "state_shape": list(episode.state.shape),
            "action_shape": list(episode.action.shape),
        },
        "controlled_failure_injection": injection.to_dict(),
        "causal_failure_slice": final_slice.to_dict(),
        "coarse_action_chunk_slice": minimization.coarse_slice.to_dict(),
        "reproduction_statistics": minimization.final_evaluation.to_dict(),
        "counterfactual_pass_variant": {
            "description": "Remove the known injected causal window from the candidate slice.",
            "candidate_without_injected_window": without_injected.to_dict(),
            "evaluation": counterfactual_eval.to_dict(),
        },
        "causal_metrics": {
            "causal_effect_score": float(causal_effect),
            "injected_window_iou": float(oracle.injected_iou(final_slice)),
            "reduction_ratio": reduction_ratio,
        },
        "search": {
            "evaluations": int(minimization.evaluations),
            "trace": minimization.search_trace,
        },
        "training_conversion": {
            "risk_sample": "observation/state/action context -> injected failure type",
            "recovery_sample": "state before slice + corrupted action chunk -> original action chunk",
            "counterfactual_sample": "slice with injected window removed should lower failure probability",
        },
        "limitations": [
            "This v0 report uses controlled injected failures, not naturally occurring VLA failures.",
            "The oracle is a replay surrogate; simulator or real-robot replay is needed for causal claims.",
        ],
    }


def save_bug_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def write_summary(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
