from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "outputs" / "risk_critic_v1.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "risk_critic_metrics.json"


def _read_jsonl(path: Path) -> list:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _span_length(candidate: object) -> float:
    if not isinstance(candidate, dict):
        return 0.0
    span = candidate.get("span")
    if isinstance(span, list) and len(span) == 2:
        return float(max(0, int(span[1]) - int(span[0])))
    intervals = candidate.get("intervals")
    if isinstance(intervals, list):
        total = 0
        for interval in intervals:
            if not isinstance(interval, list) or len(interval) != 2:
                continue
            s, e = interval
            total += max(0, int(e) - int(s))
        return float(total)
    return 0.0


def _float_list(value: object) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    try:
        return [float(x) for x in value]
    except Exception:
        return []


def _pad(values: object, length: int) -> list[float]:
    arr = _float_list(values)
    if len(arr) < length:
        arr = arr + [0.0] * (length - len(arr))
    return arr[:length]


def _legacy_numeric_features(
    sample: dict, include_oracle_features: bool = False
) -> tuple[list[float], list[str]]:
    features = sample.get("features") or {}
    action_summary = features.get("action_summary") or {}
    candidate = sample.get("candidate")
    values = [
        _span_length(candidate),
        float(action_summary.get("num_actions") or 0.0),
        float(action_summary.get("gripper_transitions") or 0.0),
        1.0 if features.get("feature_quality") == "full" else 0.0,
    ]
    names = [
        "span_length",
        "num_actions",
        "gripper_transitions",
        "feature_quality_full",
    ]
    if include_oracle_features:
        values.extend(
            [
                float(sample.get("same_failure_rate") or 0.0),
                float(sample.get("failure_rate") or 0.0),
                float(sample.get("causal_effect") or 0.0),
                1.0 if sample.get("causal_validation_passed") else 0.0,
            ]
        )
        names.extend(
            [
                "oracle_same_failure_rate",
                "oracle_failure_rate",
                "oracle_causal_effect",
                "oracle_causal_validation_passed",
            ]
        )
    for key in ("mean", "std", "max_abs"):
        for i, value in enumerate(_pad(action_summary.get(key), 7)):
            values.append(value)
            names.append(f"action_{key}_{i}")
    return values, names


def _state_features(features: dict, key: str) -> dict:
    value = features.get(key)
    return value if isinstance(value, dict) else {}


def _collect_full_vocabs(samples: Sequence[dict]) -> tuple[list[str], list[str]]:
    predicates = set()
    objects = set()
    for sample in samples:
        features = sample.get("features") or {}
        for state_key in ("pre_state_features", "post_state_features"):
            state = _state_features(features, state_key)
            predicates.update((state.get("goal_truth") or {}).keys())
            objects.update((state.get("object_positions") or {}).keys())
        signature = sample.get("failure_signature") or {}
        predicates.update(signature.get("failed_goal_predicates") or [])
    return sorted(predicates), sorted(objects)


def _full_numeric_features(
    sample: dict,
    predicates: Sequence[str],
    objects: Sequence[str],
    include_oracle_features: bool = False,
    max_action_steps: int = 8,
    action_dim: int = 7,
) -> tuple[list[float], list[str]]:
    features = sample.get("features") or {}
    action_summary = features.get("action_summary") or {}
    pre = _state_features(features, "pre_state_features")
    post = _state_features(features, "post_state_features")
    state_delta = _state_features(features, "state_delta_features")
    candidate = sample.get("candidate")
    values: list[float] = []
    names: list[str] = []

    def add(name: str, value: object) -> None:
        try:
            values.append(float(value))
        except Exception:
            values.append(0.0)
        names.append(name)

    def add_vec(prefix: str, value: object, length: int) -> None:
        for i, item in enumerate(_pad(value, length)):
            add(f"{prefix}_{i}", item)

    add("span_length", _span_length(candidate))
    add("feature_quality_full", 1.0 if features.get("feature_quality") == "full" else 0.0)
    add("candidate_actions_truncated", 1.0 if features.get("candidate_actions_truncated") else 0.0)
    add("num_actions", action_summary.get("num_actions") or 0.0)
    add("action_dim", action_summary.get("action_dim") or 0.0)
    add("gripper_transitions", action_summary.get("gripper_transitions") or 0.0)
    for key in ("mean", "std", "max_abs"):
        add_vec(f"action_{key}", action_summary.get(key), action_dim)

    actions = features.get("candidate_actions") or []
    for step in range(max_action_steps):
        action = actions[step] if step < len(actions) else []
        add_vec(f"raw_action_{step}", action, action_dim)

    add_vec("pre_eef_pos", pre.get("eef_pos"), 3)
    add_vec("post_eef_pos", post.get("eef_pos"), 3)
    add_vec("delta_eef_pos", state_delta.get("eef_delta"), 3)
    add_vec("pre_gripper_qpos", pre.get("gripper_qpos"), 2)
    add_vec("post_gripper_qpos", post.get("gripper_qpos"), 2)
    add_vec("delta_gripper_qpos", state_delta.get("gripper_delta"), 2)

    pre_goal = pre.get("goal_truth") or {}
    post_goal = post.get("goal_truth") or {}
    goal_delta = state_delta.get("goal_truth_delta") or {}
    for predicate in predicates:
        add(f"pre_goal::{predicate}", 1.0 if pre_goal.get(predicate) else 0.0)
        add(f"post_goal::{predicate}", 1.0 if post_goal.get(predicate) else 0.0)
        add(f"delta_goal::{predicate}", goal_delta.get(predicate) or 0.0)

    pre_objects = pre.get("object_positions") or {}
    post_objects = post.get("object_positions") or {}
    object_delta = state_delta.get("object_position_delta") or {}
    for obj in objects:
        pre_pos = _pad(pre_objects.get(obj), 3)
        post_pos = _pad(post_objects.get(obj), 3)
        delta = object_delta.get(obj) if isinstance(object_delta, dict) else None
        if isinstance(delta, dict):
            delta_pos = _pad(delta.get("delta"), 3)
            moved_l2 = float(delta.get("l2") or 0.0)
        else:
            delta_pos = [post_pos[i] - pre_pos[i] for i in range(3)]
            moved_l2 = float(np.linalg.norm(np.asarray(delta_pos, dtype=np.float64)))
        add_vec(f"pre_obj::{obj}", pre_pos, 3)
        add_vec(f"post_obj::{obj}", post_pos, 3)
        add_vec(f"delta_obj::{obj}", delta_pos, 3)
        add(f"delta_obj_l2::{obj}", moved_l2)

    if include_oracle_features:
        add("oracle_same_failure_rate", sample.get("same_failure_rate") or 0.0)
        add("oracle_failure_rate", sample.get("failure_rate") or 0.0)
        add("oracle_causal_effect", sample.get("causal_effect") or 0.0)
        add("oracle_causal_validation_passed", 1.0 if sample.get("causal_validation_passed") else 0.0)

    return values, names


def vectorize(
    samples: Sequence[dict],
    include_oracle_features: bool = False,
    include_failure_metadata: bool = False,
    feature_set: str = "legacy",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    failure_types = []
    sample_kinds = []
    if include_failure_metadata:
        failure_types = sorted(
            {
                str((sample.get("failure_signature") or {}).get("failure_type"))
                for sample in samples
            }
        )
        sample_kinds = sorted({str(sample.get("sample_kind")) for sample in samples})
    vocab = [f"failure_type={x}" for x in failure_types] + [
        f"sample_kind={x}" for x in sample_kinds
    ]

    predicates: list[str] = []
    objects: list[str] = []
    if feature_set == "full_state_action_goal":
        predicates, objects = _collect_full_vocabs(samples)

    rows = []
    labels = []
    feature_names: Optional[list[str]] = None
    for sample in samples:
        if feature_set == "full_state_action_goal":
            row, names = _full_numeric_features(
                sample,
                predicates,
                objects,
                include_oracle_features=include_oracle_features,
            )
        else:
            row, names = _legacy_numeric_features(
                sample, include_oracle_features=include_oracle_features
            )
        failure_type = str((sample.get("failure_signature") or {}).get("failure_type"))
        sample_kind = str(sample.get("sample_kind"))
        row.extend(1.0 if failure_type == x else 0.0 for x in failure_types)
        row.extend(1.0 if sample_kind == x else 0.0 for x in sample_kinds)
        if feature_names is None:
            feature_names = list(names) + vocab
        rows.append(row)
        labels.append(int(sample["label"]))
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(labels, dtype=np.float64),
        feature_names or [],
    )


def _standardize(train_x: np.ndarray, val_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, (val_x - mean) / std, {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def train_logistic_regression(
    train_x: np.ndarray,
    train_y: np.ndarray,
    steps: int = 1000,
    lr: float = 0.05,
    l2: float = 1e-3,
) -> tuple[np.ndarray, float]:
    weights = np.zeros(train_x.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(int(steps)):
        probs = _sigmoid(train_x @ weights + bias)
        error = probs - train_y
        grad_w = train_x.T @ error / train_x.shape[0] + l2 * weights
        grad_b = float(error.mean())
        weights -= lr * grad_w
        bias -= lr * grad_b
    return weights, bias


def _auroc(y: np.ndarray, scores: np.ndarray) -> Optional[float]:
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_rank_sum = float(ranks[y == 1].sum())
    return float((pos_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def _auprc(y: np.ndarray, scores: np.ndarray) -> Optional[float]:
    positives = int(np.sum(y == 1))
    if positives == 0:
        return None
    order = np.argsort(-scores)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / positives
    prev_recall = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - prev_recall) * precision))


def _recall_at_precision(
    y: np.ndarray, scores: np.ndarray, min_precision: float = 0.9
) -> Optional[float]:
    positives = int(np.sum(y == 1))
    if positives == 0:
        return None
    order = np.argsort(-scores)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / positives
    valid = recall[precision >= min_precision]
    if valid.size == 0:
        return 0.0
    return float(valid.max())


def _has_both_labels(y: np.ndarray, indices: Sequence[int]) -> bool:
    if len(indices) == 0:
        return False
    return len(set(int(y[i]) for i in indices)) == 2


def _split_sample_indices(
    samples: Sequence[dict],
    y: np.ndarray,
    seed: int,
    val_fraction: float,
    split_by: str,
) -> Optional[tuple[np.ndarray, np.ndarray, dict]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(samples))

    if split_by == "sample":
        rng.shuffle(indices)
        val_size = max(1, int(round(len(indices) * val_fraction)))
        train_idx = indices[val_size:]
        val_idx = indices[:val_size]
        if not _has_both_labels(y, train_idx) or not _has_both_labels(y, val_idx):
            pos = indices[y[indices] == 1]
            neg = indices[y[indices] == 0]
            if len(pos) < 2 or len(neg) < 2:
                return None
            val_idx = np.asarray([pos[0], neg[0]], dtype=np.int64)
            held = set(int(i) for i in val_idx)
            train_idx = np.asarray([i for i in indices if int(i) not in held], dtype=np.int64)
        return train_idx, val_idx, {"split_by": split_by, "group_overlap": []}

    if split_by != "source_report":
        raise ValueError("Unsupported split_by: %s" % split_by)

    groups: dict[str, list[int]] = defaultdict(list)
    for i, sample in enumerate(samples):
        groups[str(sample.get("source_report", "unknown"))].append(i)
    if len(groups) < 2:
        return None

    group_keys = list(groups)
    target_val = max(1, int(round(len(samples) * val_fraction)))
    for _ in range(500):
        keys = list(group_keys)
        rng.shuffle(keys)
        val_groups = []
        val_indices: list[int] = []
        for key in keys:
            if len(val_indices) >= target_val and _has_both_labels(y, val_indices):
                break
            val_groups.append(key)
            val_indices.extend(groups[key])
        val_set = set(val_groups)
        train_indices = [
            idx for key in group_keys if key not in val_set for idx in groups[key]
        ]
        if _has_both_labels(y, train_indices) and _has_both_labels(y, val_indices):
            return (
                np.asarray(train_indices, dtype=np.int64),
                np.asarray(val_indices, dtype=np.int64),
                {
                    "split_by": split_by,
                    "num_train_groups": len(set(group_keys) - val_set),
                    "num_val_groups": len(val_set),
                    "group_overlap": sorted(set(group_keys) & val_set & (set(group_keys) - val_set)),
                    "val_groups": sorted(val_set),
                },
            )
    return None


def _insufficient_summary(
    samples: Sequence[dict],
    reason: str,
    feature_set: str,
    split_by: str,
    min_class_count: int,
) -> dict:
    positives = sum(1 for sample in samples if int(sample.get("label", 0)) == 1)
    negatives = sum(1 for sample in samples if int(sample.get("label", 0)) == 0)
    return {
        "schema_version": "risk-critic-train-summary-v1",
        "status": "insufficient_data",
        "reason": reason,
        "num_samples": int(len(samples)),
        "num_train": 0,
        "num_val": 0,
        "num_positive": int(positives),
        "num_negative": int(negatives),
        "metrics": {
            "train_auroc": None,
            "train_auprc": None,
            "val_auroc": None,
            "val_auprc": None,
            "val_recall_at_precision_0_90": None,
        },
        "feature_policy": {
            "feature_set": feature_set,
            "split_by": split_by,
            "min_class_count": int(min_class_count),
            "include_oracle_features": False,
            "include_failure_metadata": False,
            "note": "Training skipped because the requested split would not produce a meaningful held-out evaluation.",
        },
        "model": None,
    }


def train_and_evaluate(
    samples: Sequence[dict],
    seed: int = 0,
    val_fraction: float = 0.25,
    steps: int = 1000,
    include_oracle_features: bool = False,
    include_failure_metadata: bool = False,
    feature_set: str = "legacy",
    split_by: str = "sample",
    min_class_count: int = 2,
) -> dict:
    if len(samples) < 4:
        return _insufficient_summary(
            samples, "fewer_than_4_samples", feature_set, split_by, min_class_count
        )
    positives = sum(1 for sample in samples if int(sample.get("label", 0)) == 1)
    negatives = sum(1 for sample in samples if int(sample.get("label", 0)) == 0)
    if positives < min_class_count or negatives < min_class_count:
        return _insufficient_summary(
            samples,
            "class_count_below_minimum",
            feature_set,
            split_by,
            min_class_count,
        )

    x, y, feature_names = vectorize(
        samples,
        include_oracle_features=include_oracle_features,
        include_failure_metadata=include_failure_metadata,
        feature_set=feature_set,
    )
    split = _split_sample_indices(samples, y, seed, val_fraction, split_by)
    if split is None:
        return _insufficient_summary(
            samples, "could_not_create_non_leaky_balanced_split", feature_set, split_by, min_class_count
        )
    train_idx, val_idx, split_info = split
    train_x, val_x = x[train_idx], x[val_idx]
    train_y, val_y = y[train_idx], y[val_idx]
    train_x, val_x, norm = _standardize(train_x, val_x)
    weights, bias = train_logistic_regression(train_x, train_y, steps=steps)
    train_scores = _sigmoid(train_x @ weights + bias)
    val_scores = _sigmoid(val_x @ weights + bias)
    return {
        "schema_version": "risk-critic-train-summary-v1",
        "status": "trained",
        "num_samples": int(len(samples)),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "num_positive": int(np.sum(y == 1)),
        "num_negative": int(np.sum(y == 0)),
        "metrics": {
            "train_auroc": _auroc(train_y, train_scores),
            "train_auprc": _auprc(train_y, train_scores),
            "val_auroc": _auroc(val_y, val_scores),
            "val_auprc": _auprc(val_y, val_scores),
            "val_recall_at_precision_0_90": _recall_at_precision(
                val_y, val_scores, min_precision=0.9
            ),
        },
        "feature_policy": {
            "feature_set": feature_set,
            "split_by": split_by,
            "min_class_count": int(min_class_count),
            "include_oracle_features": bool(include_oracle_features),
            "include_failure_metadata": bool(include_failure_metadata),
            "note": (
                "Default excludes same_failure_rate, causal_effect, sample_kind, "
                "and failure_type from features to avoid label leakage."
            ),
        },
        "split": split_info,
        "model": {
            "type": "numpy_logistic_regression",
            "feature_names": feature_names,
            "weights": weights.tolist(),
            "bias": float(bias),
            "normalization": norm,
        },
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight risk critic.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument(
        "--feature-set",
        choices=("legacy", "full_state_action_goal"),
        default="legacy",
    )
    parser.add_argument(
        "--split-by",
        choices=("sample", "source_report"),
        default="sample",
    )
    parser.add_argument("--min-class-count", type=int, default=4)
    parser.add_argument("--include-oracle-features", action="store_true")
    parser.add_argument("--include-failure-metadata", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    samples = _read_jsonl(args.dataset)
    summary = train_and_evaluate(
        samples,
        seed=args.seed,
        val_fraction=args.val_fraction,
        steps=args.steps,
        include_oracle_features=args.include_oracle_features,
        include_failure_metadata=args.include_failure_metadata,
        feature_set=args.feature_set,
        split_by=args.split_by,
        min_class_count=args.min_class_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary.get("metrics", {}), indent=2, ensure_ascii=False))
    print(f"status={summary.get('status')}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
