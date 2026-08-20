from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from edd_types import CandidateSlice, ChunkUnit, make_chunk_units
from failure_oracle import OracleEvaluation, StochasticFailureOracle


@dataclass(frozen=True)
class MinimizationConfig:
    chunk_size: int = 10
    trials_per_candidate: int = 24
    accept_same_failure_rate: float = 0.80
    refine_min_window: int = 1


@dataclass
class MinimizationResult:
    original_slice: CandidateSlice
    coarse_slice: CandidateSlice
    minimal_slice: CandidateSlice
    final_evaluation: OracleEvaluation
    coarse_evaluation: OracleEvaluation
    evaluations: int
    search_trace: List[dict]

    def to_dict(self) -> dict:
        return {
            "original_slice": self.original_slice.to_dict(),
            "coarse_slice": self.coarse_slice.to_dict(),
            "minimal_slice": self.minimal_slice.to_dict(),
            "coarse_evaluation": self.coarse_evaluation.to_dict(),
            "final_evaluation": self.final_evaluation.to_dict(),
            "evaluations": int(self.evaluations),
            "search_trace": self.search_trace,
        }


class SHEDMinimizer:
    """Stochastic hierarchical embodied delta debugging prototype."""

    def __init__(
        self, oracle: StochasticFailureOracle, config: MinimizationConfig
    ) -> None:
        self.oracle = oracle
        self.config = config
        self.evaluations = 0
        self.search_trace: List[dict] = []

    def minimize(self, n_steps: int) -> MinimizationResult:
        original = CandidateSlice.full(n_steps=n_steps, level="full_trajectory")
        units = make_chunk_units(n_steps=n_steps, chunk_size=self.config.chunk_size)
        coarse_units = self._ddmin_units(units, n_steps=n_steps)
        coarse = self._candidate_from_units(coarse_units, n_steps=n_steps, level="action_chunk")
        coarse_eval = self._evaluate(coarse, stage="coarse_final")
        minimal = self._refine_window(coarse, n_steps=n_steps)
        final_eval = self._evaluate(minimal, stage="frame_final")
        return MinimizationResult(
            original_slice=original,
            coarse_slice=coarse,
            minimal_slice=minimal,
            final_evaluation=final_eval,
            coarse_evaluation=coarse_eval,
            evaluations=self.evaluations,
            search_trace=self.search_trace,
        )

    def _accepts(self, candidate: CandidateSlice, stage: str) -> bool:
        evaluation = self._evaluate(candidate, stage=stage)
        return evaluation.same_failure_rate >= self.config.accept_same_failure_rate

    def _evaluate(self, candidate: CandidateSlice, stage: str) -> OracleEvaluation:
        evaluation = self.oracle.evaluate(
            candidate, trials=self.config.trials_per_candidate
        )
        self.evaluations += 1
        self.search_trace.append(
            {
                "stage": stage,
                "candidate": candidate.to_dict(),
                "same_failure_rate": float(evaluation.same_failure_rate),
                "failure_rate": float(evaluation.failure_rate),
                "overlap_ratio": float(evaluation.overlap_ratio),
            }
        )
        return evaluation

    @staticmethod
    def _candidate_from_units(
        units: Sequence[ChunkUnit], n_steps: int, level: str
    ) -> CandidateSlice:
        return CandidateSlice.from_intervals(
            [unit.to_interval() for unit in units], n_steps=n_steps, level=level
        )

    def _ddmin_units(self, units: Sequence[ChunkUnit], n_steps: int) -> List[ChunkUnit]:
        current = list(units)
        if not current:
            return current

        # Sanity check: the full trajectory should reproduce the injected failure.
        full_candidate = self._candidate_from_units(current, n_steps=n_steps, level="action_chunk")
        if not self._accepts(full_candidate, stage="coarse_full"):
            return current

        granularity = 2
        while len(current) >= 2:
            subset_size = max(1, (len(current) + granularity - 1) // granularity)
            progressed = False
            for start in range(0, len(current), subset_size):
                stop = min(len(current), start + subset_size)
                complement = current[:start] + current[stop:]
                if not complement:
                    continue
                candidate = self._candidate_from_units(
                    complement, n_steps=n_steps, level="action_chunk"
                )
                if self._accepts(candidate, stage="coarse_remove"):
                    current = complement
                    granularity = max(2, granularity - 1)
                    progressed = True
                    break
            if progressed:
                continue
            if granularity >= len(current):
                break
            granularity = min(len(current), granularity * 2)
        return current

    def _refine_window(self, coarse: CandidateSlice, n_steps: int) -> CandidateSlice:
        if coarse.span_start is None or coarse.span_end is None:
            return coarse
        start = int(coarse.span_start)
        end = int(coarse.span_end)
        min_window = max(1, int(self.config.refine_min_window))

        # Greedily shave prefix/suffix with decreasing step sizes. This keeps
        # physical contiguity for a local replay window while still probing
        # frame-level necessity.
        step = max(1, (end - start) // 2)
        while step >= 1 and end - start > min_window:
            progressed = False
            while end - start - step >= min_window:
                candidate = CandidateSlice.from_window(
                    start + step, end, n_steps=n_steps, level="frame_refine"
                )
                if self._accepts(candidate, stage="refine_drop_prefix"):
                    start += step
                    progressed = True
                else:
                    break
            while end - start - step >= min_window:
                candidate = CandidateSlice.from_window(
                    start, end - step, n_steps=n_steps, level="frame_refine"
                )
                if self._accepts(candidate, stage="refine_drop_suffix"):
                    end -= step
                    progressed = True
                else:
                    break
            if not progressed:
                step //= 2

        return CandidateSlice.from_window(start, end, n_steps=n_steps, level="causal_failure_slice")
