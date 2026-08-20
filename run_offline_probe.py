from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from bug_report import build_bug_report, save_bug_report, write_summary
from data_probe import DEFAULT_DATASET_ROOT, LeRobotLiberoDataset
from failure_oracle import StochasticFailureOracle, inject_failure
from shed_minimizer import MinimizationConfig, SHEDMinimizer


DEFAULT_OUTPUT_DIR = Path(
    "/root/autodl-tmp/research/Embodied_Delta_Debugging/outputs/offline_probe"
)


def _feasibility_verdict(metrics: dict) -> dict:
    offline_pass = (
        metrics["mean_reduction_ratio"] >= 3.0
        and metrics["mean_same_failure_rate"] >= 0.80
        and metrics["mean_injected_window_iou"] >= 0.50
        and metrics["mean_causal_effect_score"] > 0.50
    )
    if offline_pass:
        verdict = "feasible_for_next_stage"
        recommendation = (
            "Proceed to simulator replay with real policy rollouts; the core slice "
            "minimization loop works on controlled LeRobot trajectories."
        )
    else:
        verdict = "algorithmic_risk_high"
        recommendation = (
            "Tighten the failure predicate or search granularity before investing in "
            "full VLA rollout collection."
        )
    return {
        "offline_pass": bool(offline_pass),
        "verdict": verdict,
        "recommendation": recommendation,
    }


def run_probe(args: argparse.Namespace) -> dict:
    dataset = LeRobotLiberoDataset(args.dataset_root)
    tasks = dataset.load_tasks()
    dataset_summary = dataset.summarize()
    episode_indices = dataset.sample_episode_indices(
        count=args.num_episodes,
        seed=args.seed,
        min_length=args.min_episode_length,
    )

    methods = ["reverse_chunk", "gripper_delay", "action_replace"]
    reports: List[dict] = []
    rows: List[dict] = []

    for i, episode_index in enumerate(tqdm(episode_indices, desc="offline probe")):
        episode = dataset.load_episode(episode_index, include_images=False)
        method = methods[i % len(methods)]
        injection = inject_failure(
            episode.action,
            seed=args.seed + episode_index,
            method=method,
            window_size=args.injection_window,
        )
        oracle = StochasticFailureOracle(injection.spec)
        minimizer = SHEDMinimizer(
            oracle=oracle,
            config=MinimizationConfig(
                chunk_size=args.chunk_size,
                trials_per_candidate=args.trials,
                accept_same_failure_rate=args.accept_same_failure_rate,
            ),
        )
        result = minimizer.minimize(n_steps=episode.length)
        task_language = tasks.get(episode.task_index, f"task_{episode.task_index}")
        report = build_bug_report(
            dataset_root=args.dataset_root,
            episode=episode,
            task_language=task_language,
            injection=injection,
            minimization=result,
            oracle=oracle,
            causal_effect_trials=max(args.trials, 32),
            dataset_name=args.dataset_name,
        )
        report_path = args.output_dir / f"episode_{episode_index:05d}_report.json"
        save_bug_report(report, report_path)
        reports.append(report)

        metrics = report["causal_metrics"]
        final_eval = report["reproduction_statistics"]
        row = {
            "episode_index": int(episode_index),
            "task_index": int(episode.task_index),
            "task_language": task_language,
            "failure_type": injection.spec.failure_type,
            "injected_window": list(injection.spec.window),
            "minimal_slice": report["causal_failure_slice"],
            "original_length": int(episode.length),
            "same_failure_rate": float(final_eval["same_failure_rate"]),
            "failure_rate": float(final_eval["failure_rate"]),
            "causal_effect_score": float(metrics["causal_effect_score"]),
            "injected_window_iou": float(metrics["injected_window_iou"]),
            "reduction_ratio": float(metrics["reduction_ratio"]),
            "evaluations": int(report["search"]["evaluations"]),
            "report_path": str(report_path),
        }
        rows.append(row)

    aggregate = {
        "num_episodes": len(rows),
        "mean_reduction_ratio": float(np.mean([r["reduction_ratio"] for r in rows])),
        "mean_same_failure_rate": float(np.mean([r["same_failure_rate"] for r in rows])),
        "mean_failure_rate": float(np.mean([r["failure_rate"] for r in rows])),
        "mean_causal_effect_score": float(
            np.mean([r["causal_effect_score"] for r in rows])
        ),
        "mean_injected_window_iou": float(
            np.mean([r["injected_window_iou"] for r in rows])
        ),
        "median_evaluations": float(np.median([r["evaluations"] for r in rows])),
    }
    summary = {
        "dataset_summary": dataset_summary,
        "config": {
            "dataset_root": str(args.dataset_root),
            "dataset_name": str(args.dataset_name),
            "output_dir": str(args.output_dir),
            "num_episodes": int(args.num_episodes),
            "seed": int(args.seed),
            "chunk_size": int(args.chunk_size),
            "injection_window": int(args.injection_window),
            "trials": int(args.trials),
            "accept_same_failure_rate": float(args.accept_same_failure_rate),
        },
        "aggregate_metrics": aggregate,
        "feasibility": _feasibility_verdict(aggregate),
        "episodes": rows,
    }
    write_summary(summary, args.output_dir / "summary.json")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the offline SHED-CFS prototype on local LeRobot parquet data."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-name", type=str, default="LIBERO LeRobot")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--min-episode-length", type=int, default=90)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--injection-window", type=int, default=12)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--accept-same-failure-rate", type=float, default=0.80)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_probe(args)
    print(json.dumps(summary["aggregate_metrics"], indent=2))
    print(json.dumps(summary["feasibility"], indent=2, ensure_ascii=False))
    print(f"Wrote reports to {args.output_dir}")


if __name__ == "__main__":
    main()
