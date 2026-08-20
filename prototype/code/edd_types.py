from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


Interval = Tuple[int, int]


def normalize_intervals(intervals: Iterable[Interval]) -> List[Interval]:
    """Return sorted, merged half-open intervals."""
    cleaned = sorted((int(s), int(e)) for s, e in intervals if int(e) > int(s))
    if not cleaned:
        return []

    merged: List[Interval] = [cleaned[0]]
    for start, end in cleaned[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def interval_length(intervals: Sequence[Interval]) -> int:
    return int(sum(max(0, end - start) for start, end in intervals))


def interval_intersection_length(a: Sequence[Interval], b: Sequence[Interval]) -> int:
    total = 0
    left = normalize_intervals(a)
    right = normalize_intervals(b)
    i = j = 0
    while i < len(left) and j < len(right):
        s1, e1 = left[i]
        s2, e2 = right[j]
        total += max(0, min(e1, e2) - max(s1, s2))
        if e1 < e2:
            i += 1
        else:
            j += 1
    return int(total)


def interval_iou(a: Sequence[Interval], b: Sequence[Interval]) -> float:
    inter = interval_intersection_length(a, b)
    union = interval_length(normalize_intervals(list(a) + list(b)))
    if union == 0:
        return 0.0
    return float(inter / union)


@dataclass(frozen=True)
class CandidateSlice:
    """A candidate embodied failure slice represented as frame intervals."""

    intervals: Tuple[Interval, ...]
    n_steps: int
    level: str = "frame"

    def __post_init__(self) -> None:
        merged = tuple(normalize_intervals(self.intervals))
        object.__setattr__(self, "intervals", merged)

    @classmethod
    def empty(cls, n_steps: int, level: str = "frame") -> "CandidateSlice":
        return cls(tuple(), n_steps=n_steps, level=level)

    @classmethod
    def full(cls, n_steps: int, level: str = "frame") -> "CandidateSlice":
        return cls(((0, int(n_steps)),), n_steps=n_steps, level=level)

    @classmethod
    def from_window(
        cls, start: int, end: int, n_steps: int, level: str = "frame"
    ) -> "CandidateSlice":
        start = max(0, min(int(start), int(n_steps)))
        end = max(0, min(int(end), int(n_steps)))
        return cls(((start, end),), n_steps=n_steps, level=level)

    @classmethod
    def from_intervals(
        cls, intervals: Iterable[Interval], n_steps: int, level: str = "frame"
    ) -> "CandidateSlice":
        bounded = []
        for start, end in intervals:
            s = max(0, min(int(start), int(n_steps)))
            e = max(0, min(int(end), int(n_steps)))
            if e > s:
                bounded.append((s, e))
        return cls(tuple(bounded), n_steps=n_steps, level=level)

    @property
    def length(self) -> int:
        return interval_length(self.intervals)

    @property
    def span_start(self) -> Optional[int]:
        if not self.intervals:
            return None
        return self.intervals[0][0]

    @property
    def span_end(self) -> Optional[int]:
        if not self.intervals:
            return None
        return self.intervals[-1][1]

    def remove(self, intervals_to_remove: Iterable[Interval]) -> "CandidateSlice":
        removals = normalize_intervals(intervals_to_remove)
        if not removals:
            return self

        remaining: List[Interval] = []
        for start, end in self.intervals:
            cursor = start
            for rem_start, rem_end in removals:
                if rem_end <= cursor or rem_start >= end:
                    continue
                if rem_start > cursor:
                    remaining.append((cursor, min(rem_start, end)))
                cursor = max(cursor, rem_end)
                if cursor >= end:
                    break
            if cursor < end:
                remaining.append((cursor, end))
        return CandidateSlice.from_intervals(remaining, n_steps=self.n_steps, level=self.level)

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "intervals": [[int(s), int(e)] for s, e in self.intervals],
            "length": int(self.length),
            "span": None
            if self.span_start is None
            else [int(self.span_start), int(self.span_end)],
        }


@dataclass(frozen=True)
class ChunkUnit:
    index: int
    start: int
    end: int

    def to_interval(self) -> Interval:
        return int(self.start), int(self.end)


def make_chunk_units(n_steps: int, chunk_size: int) -> List[ChunkUnit]:
    units: List[ChunkUnit] = []
    idx = 0
    for start in range(0, int(n_steps), int(chunk_size)):
        units.append(ChunkUnit(index=idx, start=start, end=min(start + chunk_size, n_steps)))
        idx += 1
    return units
