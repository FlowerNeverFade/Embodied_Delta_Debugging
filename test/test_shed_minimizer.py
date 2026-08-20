from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from failure_oracle import StochasticFailureOracle, inject_failure
from shed_minimizer import MinimizationConfig, SHEDMinimizer


def _synthetic_actions(n_steps: int = 180) -> np.ndarray:
    t = np.linspace(0.0, 4.0 * np.pi, n_steps, dtype=np.float32)
    actions = np.zeros((n_steps, 7), dtype=np.float32)
    actions[:, 0] = 0.25 * np.sin(t)
    actions[:, 1] = 0.20 * np.cos(t)
    actions[:, 2] = 0.15 * np.sin(t * 0.5)
    actions[:, 3:6] = 0.02 * np.stack([np.sin(t), np.cos(t), np.sin(2 * t)], axis=1)
    actions[:, 6] = np.where(np.arange(n_steps) < n_steps // 2, -1.0, 1.0)
    return actions


def _run_minimizer_case() -> dict:
    actions = _synthetic_actions()
    injection = inject_failure(actions, seed=12, method="reverse_chunk", window_size=12)
    oracle = StochasticFailureOracle(
        injection.spec,
        false_positive_rate=0.0,
        partial_failure_rate=0.0,
        max_same_failure_rate=1.0,
        unrelated_failure_rate=0.0,
    )
    minimizer = SHEDMinimizer(
        oracle,
        MinimizationConfig(
            chunk_size=10,
            trials_per_candidate=16,
            accept_same_failure_rate=0.95,
        ),
    )
    result = minimizer.minimize(n_steps=actions.shape[0])
    causal_effect = oracle.causal_effect(result.minimal_slice, trials=16)
    assert result.final_evaluation.same_failure_rate >= 0.95
    assert oracle.injected_iou(result.minimal_slice) >= 0.50
    assert actions.shape[0] / result.minimal_slice.length >= 3.0
    assert causal_effect > 0.50
    return {
        "injection": injection.spec.to_dict(),
        "minimal_slice": result.minimal_slice.to_dict(),
        "same_failure_rate": result.final_evaluation.same_failure_rate,
        "iou": oracle.injected_iou(result.minimal_slice),
        "causal_effect": causal_effect,
        "evaluations": result.evaluations,
    }


def test_minimizer_finds_injected_window() -> None:
    _run_minimizer_case()


def main() -> None:
    result = _run_minimizer_case()
    output = PROJECT_ROOT / "test" / "outputs" / "test_shed_minimizer.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
