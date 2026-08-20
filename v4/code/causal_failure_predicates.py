from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from edd_types import CandidateSlice, Interval, interval_iou


FAILURE_UNSATISFIED_GOAL = "unsatisfied_goal_predicates_at_timeout"
FAILURE_WRONG_OBJECT = "wrong_object"
FAILURE_GRASP_MISS = "grasp_miss_no_transport"
FAILURE_PREMATURE_RELEASE = "premature_release_or_slip"
FAILURE_WRONG_PLACEMENT = "wrong_placement"
FAILURE_STAGNATION = "stagnation_timeout"
FAILURE_UNSAFE_CONTACT = "unsafe_contact"
FAILURE_ORDER_VIOLATION = "order_violation"
CAUSAL_V4_SCHEMA = "shed-cfs-causal-v4-global-multimodal"

FULL_SEMANTIC_QUALITY = "full"
DEGRADED_SEMANTIC_QUALITY = "degraded"


@dataclass(frozen=True)
class GoalPredicate:
    raw: Tuple[str, ...]

    @property
    def name(self) -> str:
        if not self.raw:
            return "unknown"
        return str(self.raw[0]).lower()

    @property
    def args(self) -> Tuple[str, ...]:
        return tuple(str(x) for x in self.raw[1:])

    @property
    def label(self) -> str:
        if not self.raw:
            return "unknown"
        head = str(self.raw[0]).lower()
        head = {"turnon": "Turnon", "turnoff": "Turnoff"}.get(head, head.capitalize())
        return " ".join([head] + [str(x) for x in self.raw[1:]])

    def to_dict(self) -> dict:
        return {"raw": list(self.raw), "label": self.label}


@dataclass(frozen=True)
class StateSnapshot:
    t: int
    success: bool
    goal_truth: Dict[str, bool]
    object_positions: Dict[str, Tuple[float, float, float]]
    eef_pos: Optional[Tuple[float, float, float]] = None
    gripper_qpos: Tuple[float, ...] = ()
    action: Optional[Tuple[float, ...]] = None
    contacts: Tuple[str, ...] = ()
    contact_records: Tuple[Dict[str, object], ...] = ()

    def to_dict(self) -> dict:
        return {
            "t": int(self.t),
            "success": bool(self.success),
            "goal_truth": {k: bool(v) for k, v in self.goal_truth.items()},
            "object_positions": {
                k: [float(x) for x in v] for k, v in self.object_positions.items()
            },
            "eef_pos": None
            if self.eef_pos is None
            else [float(x) for x in self.eef_pos],
            "gripper_qpos": [float(x) for x in self.gripper_qpos],
            "action": None if self.action is None else [float(x) for x in self.action],
            "contacts": list(self.contacts),
            "contact_records": [_jsonable(x) for x in self.contact_records],
        }


@dataclass(frozen=True)
class GoalPredicateTrace:
    predicates: Tuple[GoalPredicate, ...]
    truth_by_t: Tuple[Dict[str, bool], ...]

    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple(p.label for p in self.predicates)

    @property
    def final_truth(self) -> Dict[str, bool]:
        if not self.truth_by_t:
            return {}
        return dict(self.truth_by_t[-1])

    @property
    def failed_final_predicates(self) -> Tuple[str, ...]:
        return tuple(k for k, v in self.final_truth.items() if not bool(v))

    @property
    def final_success(self) -> bool:
        return bool(self.final_truth) and all(bool(v) for v in self.final_truth.values())

    def progress_counts(self) -> List[int]:
        return [int(sum(1 for v in truth.values() if bool(v))) for truth in self.truth_by_t]

    def first_transition_time(self, label: str, new_value: bool) -> Optional[int]:
        previous = None
        for i, truth in enumerate(self.truth_by_t):
            value = bool(truth.get(label, False))
            if previous is not None and value == new_value and previous != value:
                return i
            previous = value
        return None

    def to_dict(self) -> dict:
        first_completed = {}
        for label in self.labels:
            first_completed[label] = next(
                (i for i, truth in enumerate(self.truth_by_t) if bool(truth.get(label, False))),
                None,
            )
        return {
            "predicates": [p.to_dict() for p in self.predicates],
            "final_truth": {k: bool(v) for k, v in self.final_truth.items()},
            "failed_final_predicates": list(self.failed_final_predicates),
            "progress_counts": [int(x) for x in self.progress_counts()],
            "first_completed_step": first_completed,
            "num_steps_recorded": int(len(self.truth_by_t)),
        }


@dataclass(frozen=True)
class FailureSignature:
    failure_type: str
    failed_goal_predicates: Tuple[str, ...]
    affected_objects: Tuple[str, ...]
    anchor_start: int
    anchor_end: int
    semantic_quality: str
    confidence: float
    mechanism: str = ""
    evidence: Dict[str, object] = field(default_factory=dict)

    def anchor_interval(self) -> Interval:
        return int(self.anchor_start), int(self.anchor_end)

    def to_dict(self) -> dict:
        return {
            "failure_type": self.failure_type,
            "mechanism": self.mechanism,
            "failed_goal_predicates": list(self.failed_goal_predicates),
            "affected_objects": list(self.affected_objects),
            "anchor_window": [int(self.anchor_start), int(self.anchor_end)],
            "semantic_quality": self.semantic_quality,
            "confidence": float(self.confidence),
            "evidence": _jsonable(self.evidence),
        }


@dataclass(frozen=True)
class SameFailureResult:
    same_failure: bool
    score: float
    type_match: bool
    failed_predicate_jaccard: float
    affected_object_jaccard: float
    evidence_score: float
    threshold: float
    reasons: Tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "same_failure": bool(self.same_failure),
            "semantic_match_score": float(self.score),
            "type_match": bool(self.type_match),
            "failed_predicate_jaccard": float(self.failed_predicate_jaccard),
            "affected_object_jaccard": float(self.affected_object_jaccard),
            "evidence_score": float(self.evidence_score),
            "threshold": float(self.threshold),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class CausalUnit:
    unit_id: str
    kind: str
    interval: Interval
    evidence: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "unit_id": self.unit_id,
            "kind": self.kind,
            "interval": [int(self.interval[0]), int(self.interval[1])],
            "evidence": _jsonable(self.evidence),
        }


@dataclass(frozen=True)
class CausalUnitResult:
    unit: CausalUnit
    base_same_failure_rate: float
    ablated_same_failure_rate: float
    causal_effect: float
    is_causal_core: bool
    best_counterfactual: Dict[str, object]
    is_necessity_core: bool = False
    repair_pass: bool = False
    repair_evidence: Dict[str, object] = field(default_factory=dict)
    repair_evaluations: Tuple[Dict[str, object], ...] = field(default_factory=tuple)
    policy_strong_repair_pass: bool = False
    demo_existence_repair_pass: bool = False
    raw_policy_repair_pass: bool = False
    language_phrase_repair_pass: bool = False
    visual_mask_repair_pass: bool = False

    def to_dict(self) -> dict:
        return {
            "unit": self.unit.to_dict(),
            "base_same_failure_rate": float(self.base_same_failure_rate),
            "ablated_same_failure_rate": float(self.ablated_same_failure_rate),
            "causal_effect": float(self.causal_effect),
            "is_causal_core": bool(self.is_causal_core),
            "is_necessity_core": bool(self.is_necessity_core),
            "repair_pass": bool(self.repair_pass),
            "policy_strong_repair_pass": bool(self.policy_strong_repair_pass),
            "demo_existence_repair_pass": bool(self.demo_existence_repair_pass),
            "repair_evidence": _jsonable(self.repair_evidence),
            "repair_evaluations": [_jsonable(item) for item in self.repair_evaluations],
            "best_counterfactual": _jsonable(self.best_counterfactual),
            "raw_policy_repair_pass": bool(self.raw_policy_repair_pass),
            "language_phrase_repair_pass": bool(self.language_phrase_repair_pass),
            "visual_mask_repair_pass": bool(self.visual_mask_repair_pass),
        }


@dataclass(frozen=True)
class CausalValidationResult:
    base_same_failure_rate: float
    ce_threshold: float
    same_failure_necessity_pass: bool
    repair_valid_causal_pass: bool
    necessity_core_units: Tuple[CausalUnitResult, ...]
    causal_core_units: Tuple[CausalUnitResult, ...]
    unit_results: Tuple[CausalUnitResult, ...]
    destructive_ablation_variants: Tuple[Dict[str, object], ...]
    repair_pass_variants: Tuple[Dict[str, object], ...]
    counterfactual_pass_variants: Tuple[Dict[str, object], ...]
    both_source_repair_pass: bool
    full_success_repair_pass: bool
    passed: bool
    policy_strong_repair_valid_pass: bool = False
    demo_existence_repair_pass: bool = False
    policy_strong_core_units: Tuple[CausalUnitResult, ...] = ()
    demo_existence_core_units: Tuple[CausalUnitResult, ...] = ()
    policy_repair_pass_variants: Tuple[Dict[str, object], ...] = ()
    demo_repair_pass_variants: Tuple[Dict[str, object], ...] = ()
    full_success_policy_repair_pass: bool = False
    full_success_demo_repair_pass: bool = False
    raw_policy_repair_valid_pass: bool = False
    language_phrase_repair_valid_pass: bool = False
    visual_mask_repair_valid_pass: bool = False
    raw_policy_core_units: Tuple[CausalUnitResult, ...] = ()
    language_phrase_core_units: Tuple[CausalUnitResult, ...] = ()
    visual_mask_core_units: Tuple[CausalUnitResult, ...] = ()
    k_minimal_causal_sets: Tuple[Dict[str, object], ...] = ()
    hierarchical_pruning_trace: Tuple[Dict[str, object], ...] = ()
    repair_scheduler_trace: Tuple[Dict[str, object], ...] = ()
    deferred_repair_sources: Tuple[Dict[str, object], ...] = ()

    def to_dict(self) -> dict:
        return {
            "base_same_failure_rate": float(self.base_same_failure_rate),
            "ce_threshold": float(self.ce_threshold),
            "same_failure_necessity_pass": bool(self.same_failure_necessity_pass),
            "repair_valid_causal_pass": bool(self.repair_valid_causal_pass),
            "passed": bool(self.passed),
            "policy_strong_repair_valid_pass": bool(
                self.policy_strong_repair_valid_pass
            ),
            "demo_existence_repair_pass": bool(self.demo_existence_repair_pass),
            "necessity_core_units": [r.to_dict() for r in self.necessity_core_units],
            "causal_core_units": [r.to_dict() for r in self.causal_core_units],
            "policy_strong_core_units": [
                r.to_dict() for r in self.policy_strong_core_units
            ],
            "demo_existence_core_units": [
                r.to_dict() for r in self.demo_existence_core_units
            ],
            "unit_results": [r.to_dict() for r in self.unit_results],
            "destructive_ablation_variants": [
                _jsonable(x) for x in self.destructive_ablation_variants
            ],
            "repair_pass_variants": [
                _jsonable(x) for x in self.repair_pass_variants
            ],
            "counterfactual_pass_variants": [
                _jsonable(x) for x in self.counterfactual_pass_variants
            ],
            "policy_repair_pass_variants": [
                _jsonable(x) for x in self.policy_repair_pass_variants
            ],
            "demo_repair_pass_variants": [
                _jsonable(x) for x in self.demo_repair_pass_variants
            ],
            "both_source_repair_pass": bool(self.both_source_repair_pass),
            "full_success_repair_pass": bool(self.full_success_repair_pass),
            "full_success_policy_repair_pass": bool(
                self.full_success_policy_repair_pass
            ),
            "full_success_demo_repair_pass": bool(self.full_success_demo_repair_pass),
            "raw_policy_repair_valid_pass": bool(self.raw_policy_repair_valid_pass),
            "language_phrase_repair_valid_pass": bool(
                self.language_phrase_repair_valid_pass
            ),
            "visual_mask_repair_valid_pass": bool(self.visual_mask_repair_valid_pass),
            "raw_policy_core_units": [
                r.to_dict() for r in self.raw_policy_core_units
            ],
            "language_phrase_core_units": [
                r.to_dict() for r in self.language_phrase_core_units
            ],
            "visual_mask_core_units": [
                r.to_dict() for r in self.visual_mask_core_units
            ],
            "k_minimal_causal_sets": [_jsonable(x) for x in self.k_minimal_causal_sets],
            "hierarchical_pruning_trace": [
                _jsonable(x) for x in self.hierarchical_pruning_trace
            ],
            "repair_scheduler_trace": [
                _jsonable(x) for x in self.repair_scheduler_trace
            ],
            "deferred_repair_sources": [
                _jsonable(x) for x in self.deferred_repair_sources
            ],
        }


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _as_tuple3(value: object) -> Optional[Tuple[float, float, float]]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    if arr.shape != (3,):
        return None
    return float(arr[0]), float(arr[1]), float(arr[2])


def _base_env(env):
    return getattr(env, "env", env)


def get_goal_predicates(env) -> Tuple[GoalPredicate, ...]:
    base = _base_env(env)
    parsed = getattr(base, "parsed_problem", None)
    if not parsed:
        return tuple()
    return tuple(GoalPredicate(tuple(str(x) for x in state)) for state in parsed.get("goal_state", []))


def eval_goal_truth(env, predicates: Optional[Sequence[GoalPredicate]] = None) -> Dict[str, bool]:
    base = _base_env(env)
    if predicates is None:
        predicates = get_goal_predicates(env)
    truth: Dict[str, bool] = {}
    for predicate in predicates:
        try:
            truth[predicate.label] = bool(base._eval_predicate(list(predicate.raw)))
        except Exception:
            truth[predicate.label] = False
    return truth


def semantic_quality_for_env(env) -> str:
    predicates = get_goal_predicates(env)
    base = _base_env(env)
    if predicates and hasattr(base, "_eval_predicate"):
        return FULL_SEMANTIC_QUALITY
    return DEGRADED_SEMANTIC_QUALITY


def object_positions_from_obs(obs: dict) -> Dict[str, Tuple[float, float, float]]:
    positions: Dict[str, Tuple[float, float, float]] = {}
    for key, value in obs.items():
        if not key.endswith("_pos"):
            continue
        if "_to_" in key:
            continue
        if key.startswith("robot") or key.startswith("gripper"):
            continue
        pos = _as_tuple3(value)
        if pos is not None:
            positions[key[:-4]] = pos
    return positions


def contact_records_from_env(env) -> Tuple[Dict[str, object], ...]:
    if env is None:
        return tuple()
    sim = getattr(env, "sim", None)
    if sim is None and hasattr(env, "env"):
        sim = getattr(env.env, "sim", None)
    if sim is None or not hasattr(sim, "data") or not hasattr(sim, "model"):
        return tuple()
    records: List[Dict[str, object]] = []
    try:
        ncon = int(getattr(sim.data, "ncon", 0))
        for i in range(ncon):
            contact = sim.data.contact[i]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            name1 = sim.model.geom_id2name(geom1) or str(geom1)
            name2 = sim.model.geom_id2name(geom2) or str(geom2)
            body1 = None
            body2 = None
            try:
                body1_id = int(sim.model.geom_bodyid[geom1])
                body2_id = int(sim.model.geom_bodyid[geom2])
                body1 = sim.model.body_id2name(body1_id) or str(body1_id)
                body2 = sim.model.body_id2name(body2_id) or str(body2_id)
            except Exception:
                pass
            pos = getattr(contact, "pos", None)
            try:
                pos_list = [float(x) for x in np.asarray(pos).reshape(-1)[:3]]
            except Exception:
                pos_list = None
            dist = getattr(contact, "dist", None)
            try:
                dist_value = None if dist is None else float(dist)
            except Exception:
                dist_value = None
            records.append(
                {
                    "index": int(i),
                    "geom1": str(name1),
                    "geom2": str(name2),
                    "body1": None if body1 is None else str(body1),
                    "body2": None if body2 is None else str(body2),
                    "pair": "%s|%s" % (name1, name2),
                    "position": pos_list,
                    "distance": dist_value,
                    "force_available": False,
                    "contact_quality": "mujoco_contact",
                }
            )
    except Exception:
        return tuple()
    records = sorted(records, key=lambda item: str(item.get("pair", "")))
    return tuple(records)


def contacts_from_env(env) -> Tuple[str, ...]:
    return tuple(
        sorted({str(record.get("pair")) for record in contact_records_from_env(env)})
    )


def make_state_snapshot(
    t: int,
    obs: dict,
    env=None,
    action: Optional[Sequence[float]] = None,
    success: Optional[bool] = None,
    predicates: Optional[Sequence[GoalPredicate]] = None,
) -> StateSnapshot:
    goal_truth = eval_goal_truth(env, predicates) if env is not None else {}
    if success is None:
        success = bool(goal_truth) and all(bool(v) for v in goal_truth.values())
    eef_pos = _as_tuple3(obs.get("robot0_eef_pos")) if isinstance(obs, dict) else None
    gripper = ()
    if isinstance(obs, dict) and "robot0_gripper_qpos" in obs:
        gripper = tuple(float(x) for x in np.asarray(obs["robot0_gripper_qpos"]).reshape(-1))
    action_tuple = None
    if action is not None:
        action_tuple = tuple(float(x) for x in np.asarray(action).reshape(-1))
    contact_records = contact_records_from_env(env)
    return StateSnapshot(
        t=int(t),
        success=bool(success),
        goal_truth=goal_truth,
        object_positions=object_positions_from_obs(obs),
        eef_pos=eef_pos,
        gripper_qpos=gripper,
        action=action_tuple,
        contacts=tuple(sorted({str(record.get("pair")) for record in contact_records})),
        contact_records=contact_records,
    )


def goal_trace_from_snapshots(
    snapshots: Sequence[StateSnapshot],
    predicates: Sequence[GoalPredicate],
) -> GoalPredicateTrace:
    return GoalPredicateTrace(
        predicates=tuple(predicates),
        truth_by_t=tuple(dict(s.goal_truth) for s in snapshots),
    )


def infer_failure_signature(
    snapshots: Sequence[StateSnapshot],
    predicates: Sequence[GoalPredicate] = (),
    semantic_quality: str = FULL_SEMANTIC_QUALITY,
    event_window: int = 24,
    task_language: str = "",
) -> FailureSignature:
    n_steps = max(1, len(snapshots) - 1)
    trace = goal_trace_from_snapshots(snapshots, predicates)
    final_truth = trace.final_truth
    failed = tuple(sorted(trace.failed_final_predicates))
    final_success = bool(snapshots[-1].success) or trace.final_success
    affected = _affected_objects_from_failed(predicates, failed)
    evidence = _motion_evidence(snapshots, affected)
    stage_evidence = _stage_oracle_evidence(trace)
    anchor = _infer_anchor_window(snapshots, trace, affected, n_steps, event_window)

    if final_success and not failed:
        return FailureSignature(
            failure_type="success",
            failed_goal_predicates=tuple(),
            affected_objects=tuple(),
            anchor_start=0,
            anchor_end=0,
            semantic_quality=semantic_quality,
            confidence=1.0,
            mechanism="success",
            evidence={
                "goal_trace": trace.to_dict(),
                "stage_oracle": stage_evidence,
                "task_language": task_language,
            },
        )

    mechanism = _infer_mechanism(evidence, failed)
    if "Stageorder valid_sequence_so_far" in failed:
        mechanism = FAILURE_ORDER_VIOLATION
    failure_type = mechanism if mechanism else FAILURE_UNSATISFIED_GOAL
    if failed and failure_type == FAILURE_STAGNATION:
        # Keep the goal-level label when semantic predicates exist; stagnation is
        # reported as the mechanism so same-failure still keys on the same goal.
        failure_type = FAILURE_UNSATISFIED_GOAL
        mechanism = FAILURE_STAGNATION
    if not failed and failure_type == FAILURE_UNSATISFIED_GOAL:
        failure_type = FAILURE_STAGNATION
        mechanism = FAILURE_STAGNATION

    confidence = 0.95 if semantic_quality == FULL_SEMANTIC_QUALITY and failed else 0.65
    if semantic_quality == DEGRADED_SEMANTIC_QUALITY:
        confidence = min(confidence, 0.55)
    return FailureSignature(
        failure_type=failure_type,
        failed_goal_predicates=failed,
        affected_objects=tuple(sorted(affected)),
        anchor_start=int(anchor[0]),
        anchor_end=int(anchor[1]),
        semantic_quality=semantic_quality,
        confidence=float(confidence),
        mechanism=mechanism,
        evidence={
            "goal_trace": trace.to_dict(),
            "stage_oracle": stage_evidence,
            "motion": evidence,
            "task_language": task_language,
        },
    )


def _stage_oracle_evidence(trace: GoalPredicateTrace) -> Dict[str, object]:
    labels = [
        label
        for label in trace.labels
        if label.startswith("Stage") and label != "Stageorder valid_sequence_so_far"
    ]
    if not labels:
        return {}

    def _stage_index(label: str) -> int:
        head = label.split(" ", 1)[0]
        digits = "".join(ch for ch in head if ch.isdigit())
        return int(digits or 0)

    labels = sorted(labels, key=_stage_index)
    final_truth = trace.final_truth
    progress_count = sum(1 for label in labels if bool(final_truth.get(label, False)))
    earliest_label = next(
        (label for label in labels if not bool(final_truth.get(label, False))),
        None,
    )
    first_completed = {}
    for label in labels + ["Stageorder valid_sequence_so_far"]:
        first_completed[label] = next(
            (i for i, truth in enumerate(trace.truth_by_t) if bool(truth.get(label, False))),
            None,
        )
    earliest_failed_stage = None
    if earliest_label is not None:
        earliest_failed_stage = {
            "index": _stage_index(earliest_label),
            "key": earliest_label.split(" ", 1)[1] if " " in earliest_label else earliest_label,
            "label": earliest_label,
        }
    return {
        "schema_version": "stage-oracle-evidence-v1",
        "stage_labels": labels,
        "stage_progress_count": int(progress_count),
        "earliest_failed_stage": earliest_failed_stage,
        "stage_first_completed_step": first_completed,
        "order_valid": bool(final_truth.get("Stageorder valid_sequence_so_far", True)),
    }


def _stage_match_result(
    reference: FailureSignature,
    candidate: FailureSignature,
    threshold: float,
) -> Optional[SameFailureResult]:
    ref_stage = (reference.evidence or {}).get("stage_oracle")
    cand_stage = (candidate.evidence or {}).get("stage_oracle")
    if not isinstance(ref_stage, dict) or not isinstance(cand_stage, dict):
        return None
    if not ref_stage or not cand_stage:
        return None

    ref_order = bool(ref_stage.get("order_valid", True))
    cand_order = bool(cand_stage.get("order_valid", True))
    ref_failed = ref_stage.get("earliest_failed_stage")
    cand_failed = cand_stage.get("earliest_failed_stage")
    ref_label = ref_failed.get("label") if isinstance(ref_failed, dict) else None
    cand_label = cand_failed.get("label") if isinstance(cand_failed, dict) else None
    ref_progress = int(ref_stage.get("stage_progress_count") or 0)
    cand_progress = int(cand_stage.get("stage_progress_count") or 0)

    type_match = reference.failure_type == candidate.failure_type
    if reference.failure_type == FAILURE_ORDER_VIOLATION or candidate.failure_type == FAILURE_ORDER_VIOLATION:
        type_match = (
            reference.failure_type == candidate.failure_type
            and ref_order is False
            and cand_order is False
        )
    elif ref_label == cand_label and ref_progress == cand_progress:
        type_match = True

    same_stage = bool(ref_label and cand_label and ref_label == cand_label)
    progress_match = ref_progress == cand_progress
    order_match = ref_order == cand_order
    pred_j = _jaccard(reference.failed_goal_predicates, candidate.failed_goal_predicates)
    obj_j = _jaccard(reference.affected_objects, candidate.affected_objects)
    evidence_score = _evidence_similarity(reference, candidate)
    score = (
        (0.40 if same_stage else 0.0)
        + (0.25 if progress_match else 0.10 if abs(ref_progress - cand_progress) <= 1 else 0.0)
        + (0.15 if order_match else 0.0)
        + (0.10 if type_match else 0.0)
        + 0.05 * pred_j
        + 0.05 * evidence_score
    )
    reasons: List[str] = []
    if not same_stage:
        reasons.append("stage_mismatch:%s!=%s" % (ref_label, cand_label))
    if not progress_match:
        reasons.append("stage_progress_mismatch:%s!=%s" % (ref_progress, cand_progress))
    if not order_match:
        reasons.append("stage_order_validity_mismatch")
    if not type_match:
        reasons.append(
            "failure_type_mismatch:%s!=%s"
            % (reference.failure_type, candidate.failure_type)
        )
    same = bool(same_stage and progress_match and order_match and type_match and score >= float(threshold))
    return SameFailureResult(
        same_failure=same,
        score=float(score),
        type_match=bool(type_match),
        failed_predicate_jaccard=float(pred_j),
        affected_object_jaccard=float(obj_j),
        evidence_score=float(evidence_score),
        threshold=float(threshold),
        reasons=tuple(reasons),
    )


def compare_failure_signatures(
    reference: FailureSignature,
    candidate: FailureSignature,
    threshold: float = 0.75,
) -> SameFailureResult:
    stage_result = _stage_match_result(reference, candidate, threshold)
    if stage_result is not None:
        return stage_result

    type_match = reference.failure_type == candidate.failure_type
    pred_j = _jaccard(reference.failed_goal_predicates, candidate.failed_goal_predicates)
    obj_j = _jaccard(reference.affected_objects, candidate.affected_objects)
    evidence_score = _evidence_similarity(reference, candidate)
    score = (
        (0.35 if type_match else 0.0)
        + 0.35 * pred_j
        + 0.20 * obj_j
        + 0.10 * evidence_score
    )
    reasons: List[str] = []
    if not type_match:
        reasons.append(
            "failure_type_mismatch:%s!=%s"
            % (reference.failure_type, candidate.failure_type)
        )
    if pred_j < 0.75:
        reasons.append("failed_goal_predicate_mismatch")
    if obj_j < 0.50:
        reasons.append("affected_object_mismatch")
    same = bool(type_match and score >= float(threshold))
    return SameFailureResult(
        same_failure=same,
        score=float(score),
        type_match=bool(type_match),
        failed_predicate_jaccard=float(pred_j),
        affected_object_jaccard=float(obj_j),
        evidence_score=float(evidence_score),
        threshold=float(threshold),
        reasons=tuple(reasons),
    )


def candidate_overlaps_failure_anchor(
    candidate: CandidateSlice,
    signature: FailureSignature,
    min_iou: float = 0.05,
) -> bool:
    anchor = signature.anchor_interval()
    if anchor[1] <= anchor[0]:
        return True
    return interval_iou(candidate.intervals, [anchor]) >= float(min_iou)


def build_causal_units(
    candidate: CandidateSlice,
    actions: np.ndarray,
    snapshots: Sequence[StateSnapshot],
    signature: FailureSignature,
    task_language: str = "",
    chunk_size: int = 5,
    max_units: int = 12,
    context_before: int = 0,
    context_after: int = 0,
) -> Tuple[CausalUnit, ...]:
    if candidate.span_start is None or candidate.span_end is None:
        return tuple()
    n_actions = int(np.asarray(actions).shape[0])
    candidate_start = int(candidate.span_start)
    candidate_end = int(candidate.span_end)
    anchor = signature.anchor_interval()
    anchor_valid = anchor[1] > anchor[0]
    core_start = min(candidate_start, anchor[0]) if anchor_valid else candidate_start
    core_end = max(candidate_end, anchor[1]) if anchor_valid else candidate_end
    start = max(0, core_start - max(0, int(context_before)))
    end = min(n_actions, core_end + max(0, int(context_after)))
    if end <= start:
        start = max(0, min(candidate_start, n_actions))
        end = min(n_actions, max(candidate_end, start + 1))
    chunk_size = max(1, int(chunk_size))
    pad = max(1, chunk_size // 2)
    units: List[CausalUnit] = []

    idx = 0
    for s in range(start, end, chunk_size):
        e = min(end, s + chunk_size)
        interval = _expand_interval((s, e), 0, n_actions, pad)
        units.append(
            CausalUnit(
                unit_id="action_chunk_%03d" % idx,
                kind="action_chunk",
                interval=interval,
                evidence={
                    **_action_chunk_evidence(actions, s, e),
                    "context_unit": bool(s < candidate_start or e > candidate_end),
                    "candidate_span": [candidate_start, candidate_end],
                },
            )
        )
        idx += 1

    for interval in _gripper_transition_intervals(actions, start, end):
        interval = _expand_interval(interval, 0, n_actions, pad)
        units.append(
            CausalUnit(
                unit_id="gripper_%03d_%03d" % interval,
                kind="gripper_transition",
                interval=interval,
                evidence={"source": "action_gripper_change"},
            )
        )

    motion_interval = _object_motion_interval(snapshots, signature.affected_objects, start, end)
    if motion_interval is not None:
        motion_interval = _expand_interval(motion_interval, 0, n_actions, pad)
        units.append(
            CausalUnit(
                unit_id="object_motion_%03d_%03d" % motion_interval,
                kind="object_movement_event",
                interval=motion_interval,
                evidence={
                    "affected_objects": list(signature.affected_objects),
                    "search_context": [start, end],
                },
            )
        )

    if anchor[1] > anchor[0]:
        overlap = (max(start, anchor[0]), min(end, anchor[1]))
        if overlap[1] > overlap[0]:
            overlap = _expand_interval(overlap, 0, n_actions, pad)
            units.append(
                CausalUnit(
                    unit_id="semantic_anchor_%03d_%03d" % overlap,
                    kind="goal_predicate_anchor",
                    interval=overlap,
                    evidence={
                        "failed_goal_predicates": list(signature.failed_goal_predicates),
                        "candidate_span": [candidate_start, candidate_end],
                        "search_context": [start, end],
                    },
                )
            )
            units.append(
                CausalUnit(
                    unit_id="goal_transition_%03d_%03d" % overlap,
                    kind="goal_predicate_transition",
                    interval=overlap,
                    evidence={
                        "failed_goal_predicates": list(signature.failed_goal_predicates),
                        "candidate_span": [candidate_start, candidate_end],
                        "search_context": [start, end],
                        "source": "failure_signature_anchor",
                    },
                )
            )

    units.extend(
        _stage_phase_units(
            actions,
            snapshots,
            signature,
            start,
            end,
            n_actions,
            pad,
            candidate_span=(candidate_start, candidate_end),
        )
    )
    units.extend(
        _target_object_units(
            snapshots,
            signature,
            start,
            end,
            n_actions,
            pad,
            candidate_span=(candidate_start, candidate_end),
        )
    )
    units.extend(
        _language_phrase_units(
            signature,
            task_language,
            start,
            end,
            n_actions,
            pad,
            candidate_span=(candidate_start, candidate_end),
        )
    )
    units.extend(
        _visual_grounding_units(
            signature,
            start,
            end,
            n_actions,
            pad,
            candidate_span=(candidate_start, candidate_end),
        )
    )
    units.extend(
        _contact_event_units(
            actions,
            snapshots,
            signature,
            start,
            end,
            n_actions,
            pad,
            candidate_span=(candidate_start, candidate_end),
        )
    )
    units.extend(
        _state_anchor_units(
            signature,
            start,
            end,
            n_actions,
            pad,
            candidate_span=(candidate_start, candidate_end),
            chunk_size=chunk_size,
        )
    )

    dedup: Dict[Tuple[str, Interval], CausalUnit] = {}
    for unit in units:
        if unit.interval[1] > unit.interval[0]:
            dedup[(unit.kind, unit.interval)] = unit
    candidate_mid = 0.5 * (candidate_start + candidate_end)
    anchor_mid = 0.5 * (anchor[0] + anchor[1]) if anchor_valid else candidate_mid
    ranked = sorted(
        dedup.values(),
        key=lambda u: (
            0
            if u.kind
            in (
                "goal_predicate_anchor",
                "goal_predicate_transition",
                "target_object_hypothesis",
                "language_phrase",
                "visual_grounding_mask",
                "object_movement_event",
                "stage_phase_window",
                "contact_event",
                "state_anchor_unit",
            )
            else 1,
            abs(0.5 * (u.interval[0] + u.interval[1]) - anchor_mid),
            abs(0.5 * (u.interval[0] + u.interval[1]) - candidate_mid),
            u.interval[0],
            u.interval[1],
            u.kind,
        ),
    )
    return tuple(ranked[: int(max_units)])


def make_causal_validation_result(
    base_same_failure_rate: float,
    unit_results: Sequence[CausalUnitResult],
    ce_threshold: float = 0.30,
    hierarchical_pruning_trace: Sequence[Dict[str, object]] = (),
    repair_scheduler_trace: Sequence[Dict[str, object]] = (),
) -> CausalValidationResult:
    def _source_pass(result: CausalUnitResult, source: str) -> bool:
        if source == "policy_raw" and result.raw_policy_repair_pass:
            return True
        if source == "policy_language_phrase" and result.language_phrase_repair_pass:
            return True
        if source == "policy_visual_mask" and result.visual_mask_repair_pass:
            return True
        if source == "demo_existence" and result.demo_existence_repair_pass:
            return True
        source_names = {
            "policy_raw": {"policy_replan_from_pre_state"},
            "policy_language_phrase": {"policy_language_disambiguation_repair"},
            "policy_visual_mask": {"policy_visual_mask_repair"},
            "demo_existence": {
                "scripted_stage_expert_repair",
                "success_or_demo_nn_repair",
                "demo_repair",
                "success_nn_repair",
            },
        }.get(source, set())
        return any(
            item.get("source") in source_names and bool(item.get("repair_pass"))
            for item in result.repair_evaluations
        )

    def _has_policy_repair(result: CausalUnitResult) -> bool:
        if result.policy_strong_repair_pass:
            return True
        if (
            result.raw_policy_repair_pass
            or result.language_phrase_repair_pass
            or result.visual_mask_repair_pass
        ):
            return True
        if bool(result.repair_evidence.get("policy_repair_pass")):
            return True
        return any(
            (
                item.get("source") == "policy_replan_from_pre_state"
                or str(item.get("source", "")).startswith("policy_")
            )
            and bool(item.get("repair_pass"))
            for item in result.repair_evaluations
        )

    def _has_demo_repair(result: CausalUnitResult) -> bool:
        if result.demo_existence_repair_pass:
            return True
        if (
            bool(result.repair_evidence.get("scripted_expert_repair_pass"))
            or bool(result.repair_evidence.get("success_or_demo_repair_pass"))
            or bool(result.repair_evidence.get("demo_existence_repair_pass"))
        ):
            return True
        return any(
            item.get("source")
            in {
                "scripted_stage_expert_repair",
                "success_or_demo_nn_repair",
                "demo_repair",
                "success_nn_repair",
            }
            and bool(item.get("repair_pass"))
            for item in result.repair_evaluations
        )

    necessity_cores = tuple(
        r
        for r in unit_results
        if (r.is_necessity_core or r.causal_effect >= ce_threshold)
        and r.causal_effect >= ce_threshold
    )
    policy_cores = tuple(
        r
        for r in necessity_cores
        if _has_policy_repair(r)
    )
    raw_policy_cores = tuple(
        r for r in necessity_cores if _source_pass(r, "policy_raw")
    )
    language_phrase_cores = tuple(
        r for r in necessity_cores if _source_pass(r, "policy_language_phrase")
    )
    visual_mask_cores = tuple(
        r for r in necessity_cores if _source_pass(r, "policy_visual_mask")
    )
    demo_cores = tuple(
        r
        for r in necessity_cores
        if _has_demo_repair(r)
    )
    cores = policy_cores
    destructive_variants = tuple(
        {
            **r.best_counterfactual,
            "destructive_ablation": True,
            "repair_pass": False,
        }
        for r in unit_results
        if r.best_counterfactual
    )
    repair_variants = tuple(
        item
        for r in unit_results
        for item in r.repair_evaluations
        if bool(item.get("repair_pass"))
    )
    policy_repair_variants = tuple(
        item
        for item in repair_variants
        if item.get("source") == "policy_replan_from_pre_state"
        or str(item.get("source", "")).startswith("policy_")
    )
    demo_repair_variants = tuple(
        item
        for item in repair_variants
        if item.get("source")
        in {
            "scripted_stage_expert_repair",
            "success_or_demo_nn_repair",
            "demo_repair",
            "success_nn_repair",
        }
    )
    full_success_repair_pass = any(
        bool((item.get("evaluation") or {}).get("success"))
        or bool((item.get("repair_evidence") or {}).get("success"))
        for item in repair_variants
    )
    full_success_policy_repair_pass = any(
        bool((item.get("evaluation") or {}).get("success"))
        or bool((item.get("repair_evidence") or {}).get("success"))
        for item in policy_repair_variants
    )
    full_success_demo_repair_pass = any(
        bool((item.get("evaluation") or {}).get("success"))
        or bool((item.get("repair_evidence") or {}).get("success"))
        for item in demo_repair_variants
    )
    both_source_repair_pass = any(
        bool(r.repair_evidence.get("policy_repair_pass"))
        and bool(r.repair_evidence.get("success_or_demo_repair_pass"))
        for r in unit_results
    )
    same_failure_necessity_pass = bool(base_same_failure_rate >= 0.8 and necessity_cores)
    policy_strong_repair_valid_pass = bool(base_same_failure_rate >= 0.8 and policy_cores)
    raw_policy_repair_valid_pass = bool(base_same_failure_rate >= 0.8 and raw_policy_cores)
    language_phrase_repair_valid_pass = bool(
        base_same_failure_rate >= 0.8 and language_phrase_cores
    )
    visual_mask_repair_valid_pass = bool(
        base_same_failure_rate >= 0.8 and visual_mask_cores
    )
    demo_existence_repair_pass = bool(base_same_failure_rate >= 0.8 and demo_cores)
    repair_valid_causal_pass = policy_strong_repair_valid_pass
    k_minimal_causal_sets = build_k_minimal_causal_sets(
        base_same_failure_rate,
        unit_results,
        top_k=5,
        ce_threshold=ce_threshold,
    )
    deferred_repair_sources = tuple(
        {
            "unit_id": r.unit.unit_id,
            "unit_kind": r.unit.kind,
            "source": item.get("source"),
            "reason": item.get("reason"),
            "deferred_then_confirmed": bool(item.get("deferred_then_confirmed")),
        }
        for r in unit_results
        for item in r.repair_evaluations
        if bool(item.get("deferred"))
        or str(item.get("reason", "")).startswith("source_repair_deferred")
        or str(item.get("reason", "")).startswith("policy_repair_deferred")
    )
    return CausalValidationResult(
        base_same_failure_rate=float(base_same_failure_rate),
        ce_threshold=float(ce_threshold),
        same_failure_necessity_pass=same_failure_necessity_pass,
        repair_valid_causal_pass=repair_valid_causal_pass,
        necessity_core_units=necessity_cores,
        causal_core_units=cores,
        unit_results=tuple(unit_results),
        destructive_ablation_variants=destructive_variants,
        repair_pass_variants=repair_variants,
        counterfactual_pass_variants=repair_variants,
        both_source_repair_pass=both_source_repair_pass,
        full_success_repair_pass=full_success_repair_pass,
        passed=policy_strong_repair_valid_pass,
        policy_strong_repair_valid_pass=policy_strong_repair_valid_pass,
        demo_existence_repair_pass=demo_existence_repair_pass,
        policy_strong_core_units=policy_cores,
        demo_existence_core_units=demo_cores,
        policy_repair_pass_variants=policy_repair_variants,
        demo_repair_pass_variants=demo_repair_variants,
        full_success_policy_repair_pass=full_success_policy_repair_pass,
        full_success_demo_repair_pass=full_success_demo_repair_pass,
        raw_policy_repair_valid_pass=raw_policy_repair_valid_pass,
        language_phrase_repair_valid_pass=language_phrase_repair_valid_pass,
        visual_mask_repair_valid_pass=visual_mask_repair_valid_pass,
        raw_policy_core_units=raw_policy_cores,
        language_phrase_core_units=language_phrase_cores,
        visual_mask_core_units=visual_mask_cores,
        k_minimal_causal_sets=k_minimal_causal_sets,
        hierarchical_pruning_trace=tuple(hierarchical_pruning_trace),
        repair_scheduler_trace=tuple(repair_scheduler_trace),
        deferred_repair_sources=deferred_repair_sources,
    )


def _intervention_unit_from_repair_item(
    item: Dict[str, object],
    base_unit: CausalUnit,
) -> Optional[Dict[str, object]]:
    language = item.get("language_intervention")
    if isinstance(language, dict):
        phrase = language.get("selected_phrase") or {}
        return {
            "unit_id": "language_%s" % base_unit.unit_id,
            "kind": "language_phrase",
            "interval": [int(base_unit.interval[0]), int(base_unit.interval[1])],
            "evidence": {
                "source": item.get("source"),
                "intervention_schema": language.get("schema_version"),
                "phrase": phrase,
                "prompt_diff": language.get("prompt_diff"),
                "repair_pass": bool(item.get("repair_pass")),
            },
        }
    visual = item.get("visual_policy_mask_intervention")
    if isinstance(visual, dict):
        return {
            "unit_id": "visual_%s" % base_unit.unit_id,
            "kind": "visual_grounding_mask",
            "interval": [int(base_unit.interval[0]), int(base_unit.interval[1])],
            "evidence": {
                "source": item.get("source"),
                "intervention_schema": visual.get("schema_version"),
                "mode": visual.get("mode"),
                "target_object": visual.get("target_object"),
                "visual_quality": visual.get("visual_quality"),
                "image_rect": visual.get("image_rect"),
                "applied_to_policy_input": bool(visual.get("applied_to_policy_input")),
                "repair_pass": bool(item.get("repair_pass")),
            },
        }
    return None


def _repair_sources_for_result(result: CausalUnitResult) -> Tuple[str, ...]:
    sources: List[str] = []
    for source, flag in (
        ("policy_raw", result.raw_policy_repair_pass),
        ("policy_language_phrase", result.language_phrase_repair_pass),
        ("policy_visual_mask", result.visual_mask_repair_pass),
        ("demo_existence", result.demo_existence_repair_pass),
    ):
        if flag:
            sources.append(source)
    return tuple(sources)


def _full_success_from_result(result: CausalUnitResult) -> bool:
    for item in result.repair_evaluations:
        evaluation = item.get("evaluation") or {}
        evidence = item.get("repair_evidence") or {}
        if bool(evaluation.get("success")) or bool(evidence.get("success")):
            return True
    return False


def build_global_multimodal_units(
    unit_results: Sequence[CausalUnitResult],
) -> Tuple[Dict[str, object], ...]:
    units: List[Dict[str, object]] = []
    seen = set()
    for result in unit_results:
        unit_dict = result.unit.to_dict()
        key = (unit_dict.get("kind"), unit_dict.get("unit_id"))
        if key not in seen:
            units.append(unit_dict)
            seen.add(key)
        for item in result.repair_evaluations:
            intervention_unit = _intervention_unit_from_repair_item(item, result.unit)
            if intervention_unit is None:
                continue
            key = (intervention_unit.get("kind"), intervention_unit.get("unit_id"))
            if key not in seen:
                units.append(intervention_unit)
                seen.add(key)
    return tuple(units)


def build_k_minimal_causal_sets(
    base_same_failure_rate: float,
    unit_results: Sequence[CausalUnitResult],
    top_k: int = 5,
    ce_threshold: float = 0.30,
) -> Tuple[Dict[str, object], ...]:
    source_rank = {
        "policy_raw": 0,
        "policy_language_phrase": 1,
        "policy_visual_mask": 2,
        "demo_existence": 3,
    }
    candidates: List[Dict[str, object]] = []
    for result in unit_results:
        if not result.is_necessity_core and result.causal_effect < ce_threshold:
            continue
        sources = _repair_sources_for_result(result)
        best_source_rank = min((source_rank.get(src, 9) for src in sources), default=9)
        policy_strong = any(src.startswith("policy_") for src in sources)
        base_unit = result.unit.to_dict()
        unit_dicts = [base_unit]
        for item in result.repair_evaluations:
            if not bool(item.get("repair_pass")):
                continue
            intervention_unit = _intervention_unit_from_repair_item(item, result.unit)
            if intervention_unit is not None:
                unit_dicts.append(intervention_unit)
        unit_length = int(result.unit.interval[1] - result.unit.interval[0])
        candidates.append(
            {
                "schema_version": "shed-cfs-k-minimal-causal-set-v1",
                "set_id": "set_%s" % result.unit.unit_id,
                "bounded_minimal": True,
                "minimality_scope": "generated_multimodal_candidate_units",
                "units": unit_dicts,
                "same_failure_rate": float(base_same_failure_rate),
                "causal_effect": float(result.causal_effect),
                "ablated_same_failure_rate": float(result.ablated_same_failure_rate),
                "repair_sources": list(sources),
                "repair_valid": bool(sources),
                "policy_strong_repair_valid": bool(policy_strong),
                "best_repair_source_rank": int(best_source_rank),
                "same_failure_necessity": bool(result.is_necessity_core),
                "full_success_repair": _full_success_from_result(result),
                "drop_one_unit_checks": [
                    {
                        "unit_id": result.unit.unit_id,
                        "ablated_same_failure_rate": float(
                            result.ablated_same_failure_rate
                        ),
                        "causal_effect": float(result.causal_effect),
                        "passes_ce_threshold": bool(result.causal_effect >= ce_threshold),
                    }
                ],
                "unit_action_length": unit_length,
            }
        )
    ranked = sorted(
        candidates,
        key=lambda item: (
            0 if bool(item.get("full_success_repair")) else 1,
            0 if bool(item.get("policy_strong_repair_valid")) else 1,
            int(item.get("best_repair_source_rank", 9)),
            0 if bool(item.get("repair_valid")) else 1,
            int(len(item.get("units") or [])),
            int(item.get("unit_action_length") or 10**9),
            -float(item.get("causal_effect") or 0.0),
            str(item.get("set_id")),
        ),
    )
    for rank, item in enumerate(ranked[: max(1, int(top_k))], start=1):
        item["rank"] = int(rank)
    return tuple(ranked[: max(1, int(top_k))])


def _affected_objects_from_failed(
    predicates: Sequence[GoalPredicate],
    failed_labels: Sequence[str],
) -> Tuple[str, ...]:
    failed_set = set(failed_labels)
    objects = set()
    for predicate in predicates:
        if predicate.label not in failed_set:
            continue
        for arg in predicate.args:
            if arg.endswith("_region"):
                continue
            objects.add(arg)
    return tuple(sorted(objects))


def _motion_evidence(
    snapshots: Sequence[StateSnapshot], affected_objects: Sequence[str]
) -> Dict[str, object]:
    objects = list(affected_objects)
    if not objects and snapshots:
        objects = sorted(snapshots[0].object_positions.keys())
    per_object = {}
    for obj in objects:
        positions = [
            np.asarray(s.object_positions[obj], dtype=np.float64)
            for s in snapshots
            if obj in s.object_positions
        ]
        if len(positions) < 2:
            continue
        arr = np.stack(positions, axis=0)
        disp = np.linalg.norm(arr[-1] - arr[0])
        max_lift = float(np.max(arr[:, 2] - arr[0, 2]))
        step_motion = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        per_object[obj] = {
            "displacement": float(disp),
            "max_lift": max_lift,
            "max_step_motion": float(step_motion.max(initial=0.0)),
            "total_motion": float(step_motion.sum()),
        }

    non_target = {}
    if snapshots:
        affected_set = set(affected_objects)
        all_objects = sorted(snapshots[0].object_positions.keys())
        for obj in all_objects:
            if obj in affected_set:
                continue
            positions = [
                np.asarray(s.object_positions[obj], dtype=np.float64)
                for s in snapshots
                if obj in s.object_positions
            ]
            if len(positions) < 2:
                continue
            arr = np.stack(positions, axis=0)
            step_motion = np.linalg.norm(np.diff(arr, axis=0), axis=1)
            non_target[obj] = {
                "displacement": float(np.linalg.norm(arr[-1] - arr[0])),
                "total_motion": float(step_motion.sum()),
            }

    eef_min_distance = None
    if snapshots and affected_objects:
        dists = []
        primary = affected_objects[0]
        for s in snapshots:
            if s.eef_pos is not None and primary in s.object_positions:
                dists.append(
                    float(
                        np.linalg.norm(
                            np.asarray(s.eef_pos, dtype=np.float64)
                            - np.asarray(s.object_positions[primary], dtype=np.float64)
                        )
                    )
                )
        if dists:
            eef_min_distance = float(min(dists))

    action_norms = []
    gripper_values = []
    for s in snapshots:
        if s.action is not None:
            action = np.asarray(s.action, dtype=np.float64)
            action_norms.append(float(np.linalg.norm(action[: min(6, action.size)])))
            if action.size >= 7:
                gripper_values.append(float(action[6]))
    return {
        "objects": per_object,
        "non_target_objects": non_target,
        "eef_min_distance_to_primary": eef_min_distance,
        "mean_action_norm": float(np.mean(action_norms)) if action_norms else 0.0,
        "max_action_norm": float(np.max(action_norms)) if action_norms else 0.0,
        "gripper_action_range": (
            float(max(gripper_values) - min(gripper_values)) if gripper_values else 0.0
        ),
    }


def _infer_mechanism(evidence: Dict[str, object], failed: Sequence[str]) -> str:
    objects = evidence.get("objects", {})
    non_target = evidence.get("non_target_objects", {})
    if isinstance(objects, dict) and objects:
        displacements = [float(v.get("displacement", 0.0)) for v in objects.values()]
        lifts = [float(v.get("max_lift", 0.0)) for v in objects.values()]
        max_disp = max(displacements) if displacements else 0.0
        max_lift = max(lifts) if lifts else 0.0
    else:
        max_disp = 0.0
        max_lift = 0.0
    if isinstance(non_target, dict) and non_target:
        non_target_disp = max(
            float(v.get("displacement", 0.0)) for v in non_target.values()
        )
    else:
        non_target_disp = 0.0
    eef_min = evidence.get("eef_min_distance_to_primary")
    mean_action = float(evidence.get("mean_action_norm", 0.0))
    gripper_range = float(evidence.get("gripper_action_range", 0.0))

    if failed and non_target_disp >= 0.04 and max_disp < 0.025:
        return FAILURE_WRONG_OBJECT
    if failed and eef_min is not None and float(eef_min) < 0.08 and max_disp < 0.025:
        return FAILURE_GRASP_MISS
    if failed and max_lift > 0.035 and max_disp < 0.06 and gripper_range > 0.5:
        return FAILURE_PREMATURE_RELEASE
    if failed and max_disp >= 0.04:
        return FAILURE_WRONG_PLACEMENT
    if mean_action < 0.015 and max_disp < 0.015:
        return FAILURE_STAGNATION
    return FAILURE_UNSATISFIED_GOAL if failed else FAILURE_STAGNATION


def _infer_anchor_window(
    snapshots: Sequence[StateSnapshot],
    trace: GoalPredicateTrace,
    affected_objects: Sequence[str],
    n_steps: int,
    event_window: int,
) -> Interval:
    event_window = max(1, min(int(event_window), max(1, n_steps)))
    progress = trace.progress_counts()
    if len(progress) >= 2:
        for i in range(1, len(progress)):
            if progress[i] < progress[i - 1]:
                center = i
                return _centered_window(center, n_steps, event_window)

    motion_scores = np.zeros(n_steps, dtype=np.float64)
    for obj in affected_objects:
        positions = []
        for s in snapshots:
            if obj in s.object_positions:
                positions.append(np.asarray(s.object_positions[obj], dtype=np.float64))
        if len(positions) == len(snapshots) and len(positions) >= 2:
            motion_scores += np.linalg.norm(np.diff(np.stack(positions, axis=0), axis=0), axis=1)
    action_scores = []
    for s in snapshots[1:]:
        if s.action is None:
            action_scores.append(0.0)
        else:
            action = np.asarray(s.action, dtype=np.float64)
            action_scores.append(float(np.linalg.norm(action[: min(6, action.size)])))
    if action_scores:
        motion_scores += 0.25 * np.asarray(action_scores[:n_steps], dtype=np.float64)
    if np.max(motion_scores, initial=0.0) > 0.0:
        center = int(np.argmax(motion_scores)) + 1
        return _centered_window(center, n_steps, event_window)

    center = max(1, n_steps - event_window // 2)
    return _centered_window(center, n_steps, event_window)


def _centered_window(center: int, n_steps: int, width: int) -> Interval:
    width = max(1, int(width))
    center = max(0, min(int(center), int(n_steps)))
    start = max(0, center - width // 2)
    end = min(int(n_steps), start + width)
    start = max(0, end - width)
    return int(start), int(end)


def _expand_interval(
    interval: Interval,
    lower: int,
    upper: int,
    pad: int,
) -> Interval:
    start, end = interval
    start = max(int(lower), int(start) - int(pad))
    end = min(int(upper), int(end) + int(pad))
    if end <= start:
        end = min(int(upper), start + 1)
    return int(start), int(end)


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a = set(left)
    b = set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(len(a & b) / len(a | b))


def _evidence_similarity(reference: FailureSignature, candidate: FailureSignature) -> float:
    ref_motion = reference.evidence.get("motion", {})
    cand_motion = candidate.evidence.get("motion", {})
    if not isinstance(ref_motion, dict) or not isinstance(cand_motion, dict):
        return 0.5
    ref_action = float(ref_motion.get("mean_action_norm", 0.0))
    cand_action = float(cand_motion.get("mean_action_norm", 0.0))
    if ref_action <= 1e-6 and cand_action <= 1e-6:
        return 1.0
    denom = max(ref_action, cand_action, 1e-6)
    return float(max(0.0, 1.0 - abs(ref_action - cand_action) / denom))


def _action_chunk_evidence(actions: np.ndarray, start: int, end: int) -> Dict[str, object]:
    chunk = np.asarray(actions[start:end], dtype=np.float64)
    if chunk.size == 0:
        return {"mean_action_norm": 0.0, "max_action_norm": 0.0}
    motion = np.linalg.norm(chunk[:, : min(6, chunk.shape[1])], axis=1)
    evidence = {
        "mean_action_norm": float(np.mean(motion)),
        "max_action_norm": float(np.max(motion)),
    }
    if chunk.shape[1] >= 7:
        evidence["mean_gripper_action"] = float(np.mean(chunk[:, 6]))
        evidence["gripper_action_range"] = float(np.max(chunk[:, 6]) - np.min(chunk[:, 6]))
    return evidence


def _gripper_transition_intervals(actions: np.ndarray, start: int, end: int) -> List[Interval]:
    arr = np.asarray(actions)
    if arr.ndim != 2 or arr.shape[1] < 7 or end - start <= 1:
        return []
    grip = arr[start:end, 6]
    transitions = np.where(np.abs(np.diff(grip)) >= 0.5)[0]
    intervals: List[Interval] = []
    for idx in transitions:
        s = start + int(idx)
        intervals.append((max(start, s - 1), min(end, s + 2)))
    return intervals


def _object_motion_interval(
    snapshots: Sequence[StateSnapshot],
    affected_objects: Sequence[str],
    start: int,
    end: int,
) -> Optional[Interval]:
    if not affected_objects or end <= start:
        return None
    scores = np.zeros(max(1, end - start), dtype=np.float64)
    for obj in affected_objects:
        for t in range(start, min(end, len(snapshots) - 1)):
            cur = snapshots[t].object_positions.get(obj)
            nxt = snapshots[t + 1].object_positions.get(obj)
            if cur is None or nxt is None:
                continue
            scores[t - start] += float(
                np.linalg.norm(np.asarray(nxt, dtype=np.float64) - np.asarray(cur, dtype=np.float64))
            )
    if np.max(scores, initial=0.0) <= 1e-6:
        return None
    idx = int(np.argmax(scores)) + start
    return max(start, idx - 2), min(end, idx + 3)


def _primary_affected_objects(signature: FailureSignature) -> List[str]:
    objects = [str(obj) for obj in signature.affected_objects if str(obj)]
    if objects:
        return objects
    parsed = []
    for pred in signature.failed_goal_predicates:
        parts = str(pred).split()
        parsed.extend(part for part in parts[1:] if not part.endswith("_region"))
    return sorted(set(parsed))


def _object_motion_peak_interval(
    snapshots: Sequence[StateSnapshot],
    obj: str,
    start: int,
    end: int,
) -> Optional[Interval]:
    if end <= start or len(snapshots) < 2:
        return None
    scores = []
    ts = []
    for t in range(start, min(end, len(snapshots) - 1)):
        cur = snapshots[t].object_positions.get(obj)
        nxt = snapshots[t + 1].object_positions.get(obj)
        if cur is None or nxt is None:
            continue
        scores.append(
            float(np.linalg.norm(np.asarray(nxt, dtype=np.float64) - np.asarray(cur, dtype=np.float64)))
        )
        ts.append(t)
    if not scores or max(scores) <= 1e-6:
        return None
    t = int(ts[int(np.argmax(np.asarray(scores, dtype=np.float64)))])
    return max(start, t - 2), min(end, t + 3)


def _stage_phase_units(
    actions: np.ndarray,
    snapshots: Sequence[StateSnapshot],
    signature: FailureSignature,
    start: int,
    end: int,
    n_actions: int,
    pad: int,
    candidate_span: Interval,
) -> List[CausalUnit]:
    units: List[CausalUnit] = []
    objects = _primary_affected_objects(signature)
    primary = objects[0] if objects else None
    if primary is None or end <= start:
        return units

    dists = []
    lift_scores = []
    motion_scores = []
    for t in range(start, min(end, len(snapshots) - 1)):
        cur = snapshots[t]
        nxt = snapshots[t + 1]
        if cur.eef_pos is not None and primary in cur.object_positions:
            dists.append(
                (
                    t,
                    float(
                        np.linalg.norm(
                            np.asarray(cur.eef_pos, dtype=np.float64)
                            - np.asarray(cur.object_positions[primary], dtype=np.float64)
                        )
                    ),
                )
            )
        cur_pos = cur.object_positions.get(primary)
        nxt_pos = nxt.object_positions.get(primary)
        if cur_pos is not None and nxt_pos is not None:
            cur_arr = np.asarray(cur_pos, dtype=np.float64)
            nxt_arr = np.asarray(nxt_pos, dtype=np.float64)
            lift_scores.append((t, float(nxt_arr[2] - cur_arr[2])))
            motion_scores.append((t, float(np.linalg.norm(nxt_arr - cur_arr))))

    if dists:
        t = min(dists, key=lambda item: item[1])[0]
        interval = _expand_interval((t - 2, t + 3), 0, n_actions, pad)
        units.append(
            CausalUnit(
                unit_id="stage_approach_%03d_%03d" % interval,
                kind="stage_phase_window",
                interval=interval,
                evidence={
                    "phase": "approach",
                    "target_object": primary,
                    "source": "eef_object_distance_minimum",
                    "candidate_span": list(candidate_span),
                },
            )
        )
    if lift_scores:
        t = max(lift_scores, key=lambda item: item[1])[0]
        interval = _expand_interval((t - 2, t + 3), 0, n_actions, pad)
        units.append(
            CausalUnit(
                unit_id="stage_lift_%03d_%03d" % interval,
                kind="stage_phase_window",
                interval=interval,
                evidence={
                    "phase": "lift",
                    "target_object": primary,
                    "source": "object_z_motion_peak",
                    "candidate_span": list(candidate_span),
                },
            )
        )
    if motion_scores:
        t = max(motion_scores, key=lambda item: item[1])[0]
        interval = _expand_interval((t - 3, t + 4), 0, n_actions, pad)
        units.append(
            CausalUnit(
                unit_id="stage_transport_place_%03d_%03d" % interval,
                kind="stage_phase_window",
                interval=interval,
                evidence={
                    "phase": "transport_or_place",
                    "target_object": primary,
                    "source": "object_motion_peak",
                    "candidate_span": list(candidate_span),
                },
            )
        )
    for raw_interval in _gripper_transition_intervals(actions, start, end)[:2]:
        interval = _expand_interval(raw_interval, 0, n_actions, pad)
        phase = "grasp" if len(units) <= 1 else "release"
        units.append(
            CausalUnit(
                unit_id="stage_%s_%03d_%03d" % (phase, interval[0], interval[1]),
                kind="stage_phase_window",
                interval=interval,
                evidence={
                    "phase": phase,
                    "target_object": primary,
                    "source": "gripper_transition",
                    "candidate_span": list(candidate_span),
                },
            )
        )
    return units


def _target_object_units(
    snapshots: Sequence[StateSnapshot],
    signature: FailureSignature,
    start: int,
    end: int,
    n_actions: int,
    pad: int,
    candidate_span: Interval,
) -> List[CausalUnit]:
    units: List[CausalUnit] = []
    objects = _primary_affected_objects(signature)
    motion = (signature.evidence or {}).get("motion", {})
    non_target = motion.get("non_target_objects") if isinstance(motion, dict) else {}
    moved_distractors = []
    if isinstance(non_target, dict):
        moved_distractors = [
            str(obj)
            for obj, value in non_target.items()
            if isinstance(value, dict) and float(value.get("displacement", 0.0)) >= 0.03
        ]
    for idx, obj in enumerate(objects[:3]):
        interval = _object_motion_peak_interval(snapshots, obj, start, end)
        if interval is None:
            continue
        interval = _expand_interval(interval, 0, n_actions, pad)
        units.append(
            CausalUnit(
                unit_id="target_object_%02d_%03d_%03d" % (idx, interval[0], interval[1]),
                kind="target_object_hypothesis",
                interval=interval,
                evidence={
                    "target_object": obj,
                    "role": "intended_target",
                    "failed_goal_predicates": list(signature.failed_goal_predicates),
                    "candidate_span": list(candidate_span),
                },
            )
        )
    for idx, obj in enumerate(moved_distractors[:3]):
        interval = _object_motion_peak_interval(snapshots, obj, start, end)
        if interval is None:
            continue
        interval = _expand_interval(interval, 0, n_actions, pad)
        units.append(
            CausalUnit(
                unit_id="distractor_object_%02d_%03d_%03d" % (idx, interval[0], interval[1]),
                kind="target_object_hypothesis",
                interval=interval,
                evidence={
                    "target_object": obj,
                    "role": "moved_distractor",
                    "failure_type": signature.failure_type,
                    "candidate_span": list(candidate_span),
                },
            )
        )
    return units


def _language_phrase_units(
    signature: FailureSignature,
    task_language: str,
    start: int,
    end: int,
    n_actions: int,
    pad: int,
    candidate_span: Interval,
) -> List[CausalUnit]:
    units: List[CausalUnit] = []
    language = str(task_language or "")
    interval = _expand_interval((start, min(end, start + max(1, pad))), 0, n_actions, pad)
    phrases = []
    for obj in _primary_affected_objects(signature)[:4]:
        idx = language.lower().find(str(obj).lower())
        phrases.append(
            {
                "kind": "object_phrase",
                "phrase": str(obj),
                "char_span": None if idx < 0 else [int(idx), int(idx + len(str(obj)))],
                "found_in_original_prompt": idx >= 0,
            }
        )
    for pred in signature.failed_goal_predicates:
        for arg in str(pred).split()[1:]:
            if not arg.endswith("_region"):
                continue
            idx = language.lower().find(str(arg).lower())
            phrases.append(
                {
                    "kind": "region_phrase",
                    "phrase": str(arg),
                    "char_span": None if idx < 0 else [int(idx), int(idx + len(str(arg)))],
                    "found_in_original_prompt": idx >= 0,
                }
            )
    if phrases:
        units.append(
            CausalUnit(
                unit_id="language_phrase_%03d_%03d" % interval,
                kind="language_phrase",
                interval=interval,
                evidence={
                    "phrase_candidates": phrases,
                    "candidate_span": list(candidate_span),
                    "intervention_family": "phrase_delete_replace_disambiguate",
                    "physical_world_unchanged": True,
                },
            )
        )
    return units


def _visual_grounding_units(
    signature: FailureSignature,
    start: int,
    end: int,
    n_actions: int,
    pad: int,
    candidate_span: Interval,
) -> List[CausalUnit]:
    units: List[CausalUnit] = []
    interval = _expand_interval((start, min(end, start + max(1, pad))), 0, n_actions, pad)
    objects = _primary_affected_objects(signature)[:4]
    if not objects:
        return units
    for idx, obj in enumerate(objects):
        units.append(
            CausalUnit(
                unit_id="visual_grounding_%02d_%03d_%03d" % (idx, interval[0], interval[1]),
                kind="visual_grounding_mask",
                interval=interval,
                evidence={
                    "target_object": obj,
                    "mode": "highlight_target",
                    "candidate_span": list(candidate_span),
                    "physical_world_unchanged": True,
                    "requires_projection": True,
                },
            )
        )
    return units


def _contact_event_units(
    actions: np.ndarray,
    snapshots: Sequence[StateSnapshot],
    signature: FailureSignature,
    start: int,
    end: int,
    n_actions: int,
    pad: int,
    candidate_span: Interval,
) -> List[CausalUnit]:
    del actions
    units: List[CausalUnit] = []
    seen_contact_steps = [
        t
        for t in range(start, min(end, len(snapshots)))
        if getattr(snapshots[t], "contacts", ())
    ]
    if seen_contact_steps:
        t = int(seen_contact_steps[0])
        interval = _expand_interval((t - 2, t + 3), 0, n_actions, pad)
        records = getattr(snapshots[t], "contact_records", ()) or ()
        units.append(
            CausalUnit(
                unit_id="contact_%03d_%03d" % interval,
                kind="contact_event",
                interval=interval,
                evidence={
                    "contact_quality": "mujoco_contact",
                    "contacts": list(snapshots[t].contacts),
                    "contact_records": [_jsonable(record) for record in records],
                    "hard_contact_evidence": True,
                    "candidate_span": list(candidate_span),
                },
            )
        )
        return units

    objects = _primary_affected_objects(signature)
    if not objects:
        return units
    primary = objects[0]
    close_steps = []
    for t in range(start, min(end, len(snapshots))):
        snapshot = snapshots[t]
        if snapshot.eef_pos is None or primary not in snapshot.object_positions:
            continue
        dist = float(
            np.linalg.norm(
                np.asarray(snapshot.eef_pos, dtype=np.float64)
                - np.asarray(snapshot.object_positions[primary], dtype=np.float64)
            )
        )
        gripper_closed = bool(
            snapshot.gripper_qpos and float(np.mean(snapshot.gripper_qpos)) < 0.04
        )
        if dist <= 0.08 and gripper_closed:
            close_steps.append((t, dist))
    if close_steps:
        t = min(close_steps, key=lambda item: item[1])[0]
        interval = _expand_interval((t - 2, t + 3), 0, n_actions, pad)
        units.append(
            CausalUnit(
                unit_id="contact_proximity_%03d_%03d" % interval,
                kind="contact_event",
                interval=interval,
                evidence={
                    "contact_quality": "degraded_proximity",
                    "target_object": primary,
                    "source": "eef_object_proximity_plus_gripper_close",
                    "candidate_span": list(candidate_span),
                    "not_sufficient_alone_for_strong_visual_contact_evidence": True,
                },
            )
        )
    return units


def _state_anchor_units(
    signature: FailureSignature,
    start: int,
    end: int,
    n_actions: int,
    pad: int,
    candidate_span: Interval,
    chunk_size: int,
) -> List[CausalUnit]:
    del end
    anchor = signature.anchor_interval()
    anchor_start = anchor[0] if anchor[1] > anchor[0] else start
    anchors = sorted(
        {
            max(0, start),
            max(0, anchor_start - max(8, int(chunk_size) * 2)),
            max(0, anchor_start - max(16, int(chunk_size) * 4)),
        }
    )
    units: List[CausalUnit] = []
    for idx, s in enumerate(anchors):
        e = min(n_actions, max(s + 1, s + int(chunk_size)))
        interval = _expand_interval((s, e), 0, n_actions, pad)
        units.append(
            CausalUnit(
                unit_id="state_anchor_%02d_%03d_%03d" % (idx, interval[0], interval[1]),
                kind="state_anchor_unit",
                interval=interval,
                evidence={
                    "state_anchor_step": int(s),
                    "repair_semantics": "policy_replan_from_pre_state_only",
                    "state_oracle_only": False,
                    "object_teleport_oracle_allowed": False,
                    "candidate_span": list(candidate_span),
                },
            )
        )
    return units
