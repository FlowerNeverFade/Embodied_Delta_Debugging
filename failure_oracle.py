from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from edd_types import CandidateSlice, Interval, interval_intersection_length, interval_iou


@dataclass(frozen=True)
class FailureSpec:
    failure_type: str
    start: int
    end: int
    seed: int
    injection_method: str
    required_overlap: float = 0.80
    metadata: Optional[dict] = None

    @property
    def window(self) -> Tuple[int, int]:
        return int(self.start), int(self.end)

    @property
    def length(self) -> int:
        return int(self.end - self.start)

    def to_dict(self) -> dict:
        return {
            "failure_type": self.failure_type,
            "injection_method": self.injection_method,
            "window": [int(self.start), int(self.end)],
            "length": int(self.length),
            "required_overlap": float(self.required_overlap),
            "seed": int(self.seed),
            "metadata": self.metadata or {},
        }


@dataclass(frozen=True)
class InjectionResult:
    spec: FailureSpec
    corrupted_actions: np.ndarray
    delta_norm: float
    max_frame_delta_norm: float

    def to_dict(self) -> dict:
        return {
            "spec": self.spec.to_dict(),
            "delta_norm": float(self.delta_norm),
            "max_frame_delta_norm": float(self.max_frame_delta_norm),
        }


@dataclass(frozen=True)
class OracleEvaluation:
    candidate: CandidateSlice
    trials: int
    failure_count: int
    same_failure_count: int
    overlap_ratio: float
    expected_same_failure_probability: float
    failure_type_counts: Dict[str, int]

    @property
    def failure_rate(self) -> float:
        if self.trials <= 0:
            return 0.0
        return float(self.failure_count / self.trials)

    @property
    def same_failure_rate(self) -> float:
        if self.trials <= 0:
            return 0.0
        return float(self.same_failure_count / self.trials)

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate.to_dict(),
            "trials": int(self.trials),
            "failure_count": int(self.failure_count),
            "same_failure_count": int(self.same_failure_count),
            "failure_rate": float(self.failure_rate),
            "same_failure_rate": float(self.same_failure_rate),
            "overlap_ratio": float(self.overlap_ratio),
            "expected_same_failure_probability": float(
                self.expected_same_failure_probability
            ),
            "failure_type_counts": dict(self.failure_type_counts),
        }


def select_injection_window(
    actions: np.ndarray,
    seed: int,
    window_size: int = 12,
    margin: int = 15,
) -> Tuple[int, int]:
    """Pick a high-motion window from a real demonstration trajectory."""
    n_steps = int(actions.shape[0])
    if n_steps <= window_size + 2:
        return 0, n_steps

    window_size = min(int(window_size), max(2, n_steps // 3))
    margin = min(int(margin), max(0, (n_steps - window_size) // 3))
    valid_start = margin
    valid_end = max(valid_start + 1, n_steps - window_size - margin)

    action = np.asarray(actions, dtype=np.float32)
    motion = np.linalg.norm(action[:, :6], axis=1)
    if action.shape[1] > 6:
        grip_change = np.abs(np.diff(action[:, 6], prepend=action[0, 6]))
        motion = motion + 0.25 * grip_change

    scores = []
    starts = list(range(valid_start, valid_end + 1))
    for start in starts:
        scores.append(float(np.mean(motion[start : start + window_size])))

    scores_np = np.asarray(scores, dtype=np.float64)
    if np.allclose(scores_np, scores_np[0]):
        rng = np.random.default_rng(seed)
        start = int(rng.choice(starts))
    else:
        threshold = float(np.percentile(scores_np, 75))
        candidates = [s for s, score in zip(starts, scores) if score >= threshold]
        rng = np.random.default_rng(seed)
        start = int(rng.choice(candidates))
    return start, start + window_size


def inject_failure(
    actions: np.ndarray,
    seed: int = 0,
    method: str = "reverse_chunk",
    window_size: int = 12,
) -> InjectionResult:
    """Create a controlled failure from a successful demo action trajectory."""
    original = np.asarray(actions, dtype=np.float32)
    corrupted = np.array(original, copy=True)
    start, end = select_injection_window(original, seed=seed, window_size=window_size)
    rng = np.random.default_rng(seed)

    if method == "reverse_chunk":
        corrupted[start:end, :6] = -corrupted[start:end, :6]
        failure_type = "wrong_direction_chunk"
    elif method == "gripper_delay":
        if corrupted.shape[1] < 7:
            raise ValueError("gripper_delay requires a 7D action with gripper command")
        delayed_value = corrupted[max(0, start - 1), 6]
        corrupted[start:end, 6] = delayed_value
        failure_type = "delayed_gripper_chunk"
    elif method == "action_replace":
        source_start = int(rng.integers(0, max(1, original.shape[0] - (end - start))))
        replacement = original[source_start : source_start + (end - start)]
        if replacement.shape[0] != end - start:
            replacement = replacement[::-1]
        corrupted[start:end, :] = replacement
        failure_type = "wrong_action_replacement"
    else:
        raise ValueError(f"Unknown injection method: {method}")

    delta = corrupted[start:end] - original[start:end]
    spec = FailureSpec(
        failure_type=failure_type,
        start=start,
        end=end,
        seed=seed,
        injection_method=method,
        metadata={"window_size": int(window_size)},
    )
    return InjectionResult(
        spec=spec,
        corrupted_actions=corrupted,
        delta_norm=float(np.linalg.norm(delta)),
        max_frame_delta_norm=float(np.linalg.norm(delta, axis=1).max(initial=0.0)),
    )


class StochasticFailureOracle:
    """Offline replay surrogate for first-pass causal slice testing.

    The oracle treats injected frames as the known causal mechanism. Candidate
    slices that preserve enough of that window reproduce the same failure with
    high probability; partial overlap creates weaker, flaky failures.
    """

    def __init__(
        self,
        spec: FailureSpec,
        false_positive_rate: float = 0.02,
        partial_failure_rate: float = 0.20,
        max_same_failure_rate: float = 0.96,
        unrelated_failure_rate: float = 0.03,
    ) -> None:
        self.spec = spec
        self.false_positive_rate = float(false_positive_rate)
        self.partial_failure_rate = float(partial_failure_rate)
        self.max_same_failure_rate = float(max_same_failure_rate)
        self.unrelated_failure_rate = float(unrelated_failure_rate)

    def overlap_ratio(self, candidate: CandidateSlice) -> float:
        denom = max(1, self.spec.length)
        overlap = interval_intersection_length(candidate.intervals, [self.spec.window])
        return float(overlap / denom)

    def expected_same_failure_probability(self, candidate: CandidateSlice) -> float:
        overlap = self.overlap_ratio(candidate)
        if overlap >= self.spec.required_overlap:
            return self.max_same_failure_rate
        if overlap > 0.0:
            return min(
                self.max_same_failure_rate * 0.75,
                self.partial_failure_rate + overlap * 0.55,
            )
        return self.false_positive_rate

    def _candidate_seed(self, candidate: CandidateSlice, trials: int) -> int:
        payload = repr(
            (
                self.spec.failure_type,
                self.spec.start,
                self.spec.end,
                self.spec.seed,
                candidate.intervals,
                int(trials),
            )
        ).encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], "little", signed=False)

    def evaluate(self, candidate: CandidateSlice, trials: int = 16) -> OracleEvaluation:
        p_same = self.expected_same_failure_probability(candidate)
        p_unrelated = self.unrelated_failure_rate if p_same < 0.5 else 0.01
        rng = np.random.default_rng(self._candidate_seed(candidate, trials))

        same = 0
        unrelated = 0
        for _ in range(int(trials)):
            if rng.random() < p_same:
                same += 1
            elif rng.random() < p_unrelated:
                unrelated += 1

        counts: Dict[str, int] = {}
        if same:
            counts[self.spec.failure_type] = same
        if unrelated:
            counts["unrelated_failure"] = unrelated
        return OracleEvaluation(
            candidate=candidate,
            trials=int(trials),
            failure_count=int(same + unrelated),
            same_failure_count=int(same),
            overlap_ratio=self.overlap_ratio(candidate),
            expected_same_failure_probability=p_same,
            failure_type_counts=counts,
        )

    def causal_effect(
        self,
        candidate: CandidateSlice,
        remove_intervals: Optional[Iterable[Interval]] = None,
        trials: int = 32,
    ) -> float:
        if remove_intervals is None:
            remove_intervals = [self.spec.window]
        base = self.evaluate(candidate, trials=trials).same_failure_rate
        ablated = self.evaluate(candidate.remove(remove_intervals), trials=trials).same_failure_rate
        return float(base - ablated)

    def injected_iou(self, candidate: CandidateSlice) -> float:
        return interval_iou(candidate.intervals, [self.spec.window])
