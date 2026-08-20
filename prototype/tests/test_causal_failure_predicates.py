from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "prototype" / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from causal_failure_predicates import (
    CausalUnit,
    CausalUnitResult,
    FailureSignature,
    GoalPredicate,
    StateSnapshot,
    build_causal_units,
    compare_failure_signatures,
    infer_failure_signature,
    make_causal_validation_result,
)


PREDICATES = (GoalPredicate(("on", "cup_1", "plate_region")),)


def _snapshot(
    t: int,
    cup_pos,
    eef_pos=(0.5, 0.5, 0.5),
    action=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    goal=False,
) -> StateSnapshot:
    return StateSnapshot(
        t=t,
        success=bool(goal),
        goal_truth={"On cup_1 plate_region": bool(goal)},
        object_positions={"cup_1": tuple(float(x) for x in cup_pos)},
        eef_pos=tuple(float(x) for x in eef_pos),
        gripper_qpos=(0.0, 0.0),
        action=tuple(float(x) for x in action),
    )


def test_wrong_placement_signature() -> None:
    snapshots = [
        _snapshot(0, (0.0, 0.0, 0.0), action=(0.4, 0.0, 0.0, 0, 0, 0, -1)),
        _snapshot(1, (0.03, 0.0, 0.0), action=(0.4, 0.0, 0.0, 0, 0, 0, -1)),
        _snapshot(2, (0.08, 0.0, 0.0), action=(0.4, 0.0, 0.0, 0, 0, 0, -1)),
    ]
    signature = infer_failure_signature(snapshots, PREDICATES, event_window=2)
    assert signature.failure_type == "wrong_placement"
    assert signature.failed_goal_predicates == ("On cup_1 plate_region",)
    assert signature.affected_objects == ("cup_1",)


def test_grasp_miss_signature() -> None:
    snapshots = [
        _snapshot(0, (0.0, 0.0, 0.0), eef_pos=(0.06, 0.0, 0.0)),
        _snapshot(1, (0.005, 0.0, 0.0), eef_pos=(0.03, 0.0, 0.0)),
        _snapshot(2, (0.006, 0.0, 0.0), eef_pos=(0.02, 0.0, 0.0)),
    ]
    signature = infer_failure_signature(snapshots, PREDICATES, event_window=2)
    assert signature.failure_type == "grasp_miss_no_transport"


def test_slip_signature() -> None:
    snapshots = [
        _snapshot(0, (0.0, 0.0, 0.0), action=(0, 0, 0, 0, 0, 0, -1)),
        _snapshot(1, (0.0, 0.0, 0.05), action=(0, 0, 0, 0, 0, 0, 1)),
        _snapshot(2, (0.005, 0.0, 0.0), action=(0, 0, 0, 0, 0, 0, -1)),
    ]
    signature = infer_failure_signature(snapshots, PREDICATES, event_window=2)
    assert signature.failure_type == "premature_release_or_slip"


def test_stagnation_is_goal_failure_with_stagnation_mechanism() -> None:
    snapshots = [
        _snapshot(0, (0.0, 0.0, 0.0), action=(0, 0, 0, 0, 0, 0, -1)),
        _snapshot(1, (0.0, 0.0, 0.0), action=(0, 0, 0, 0, 0, 0, -1)),
        _snapshot(2, (0.0, 0.0, 0.0), action=(0, 0, 0, 0, 0, 0, -1)),
    ]
    signature = infer_failure_signature(snapshots, PREDICATES, event_window=2)
    assert signature.failure_type == "unsatisfied_goal_predicates_at_timeout"
    assert signature.mechanism == "stagnation_timeout"


def test_same_failure_rejects_different_failure_type() -> None:
    reference = FailureSignature(
        failure_type="wrong_object",
        failed_goal_predicates=("On cup_1 plate_region",),
        affected_objects=("cup_1",),
        anchor_start=4,
        anchor_end=8,
        semantic_quality="full",
        confidence=0.9,
    )
    candidate = FailureSignature(
        failure_type="stagnation_timeout",
        failed_goal_predicates=("On cup_1 plate_region",),
        affected_objects=("cup_1",),
        anchor_start=4,
        anchor_end=8,
        semantic_quality="full",
        confidence=0.9,
    )
    result = compare_failure_signatures(reference, candidate)
    assert not result.same_failure
    assert not result.type_match


def test_causal_effect_marks_core_unit() -> None:
    unit = CausalUnit(
        unit_id="action_chunk_000",
        kind="action_chunk",
        interval=(10, 15),
    )
    unit_result = CausalUnitResult(
        unit=unit,
        base_same_failure_rate=1.0,
        ablated_same_failure_rate=0.2,
        causal_effect=0.8,
        is_causal_core=True,
        repair_pass=True,
        policy_strong_repair_pass=True,
        best_counterfactual={"strategy": "hold"},
    )
    validation = make_causal_validation_result(
        base_same_failure_rate=1.0,
        unit_results=[unit_result],
        ce_threshold=0.3,
    )
    assert validation.passed
    assert validation.causal_core_units[0].causal_effect >= 0.3


def test_destructive_ablation_is_necessity_not_repair_valid() -> None:
    unit = CausalUnit(
        unit_id="semantic_anchor_144_146",
        kind="goal_predicate_anchor",
        interval=(144, 146),
    )
    unit_result = CausalUnitResult(
        unit=unit,
        base_same_failure_rate=1.0,
        ablated_same_failure_rate=0.0,
        causal_effect=1.0,
        is_causal_core=False,
        is_necessity_core=True,
        repair_pass=False,
        best_counterfactual={
            "strategy": "hold",
            "destructive_ablation": True,
            "repair_pass": False,
        },
    )

    validation = make_causal_validation_result(
        base_same_failure_rate=1.0,
        unit_results=[unit_result],
        ce_threshold=0.3,
    )

    assert validation.same_failure_necessity_pass
    assert not validation.repair_valid_causal_pass
    assert not validation.passed
    assert len(validation.necessity_core_units) == 1
    assert len(validation.causal_core_units) == 0


def test_full_success_repair_pass_is_separate_flag() -> None:
    unit = CausalUnit(
        unit_id="semantic_anchor_10_15",
        kind="goal_predicate_anchor",
        interval=(10, 15),
    )
    unit_result = CausalUnitResult(
        unit=unit,
        base_same_failure_rate=1.0,
        ablated_same_failure_rate=0.0,
        causal_effect=1.0,
        is_causal_core=True,
        is_necessity_core=True,
        repair_pass=True,
        policy_strong_repair_pass=True,
        best_counterfactual={"strategy": "hold"},
        repair_evaluations=(
            {
                "source": "policy_replan_from_pre_state",
                "repair_pass": True,
                "evaluation": {"success": True},
                "repair_evidence": {"repair_pass": True, "success": True},
            },
        ),
    )

    validation = make_causal_validation_result(
        base_same_failure_rate=1.0,
        unit_results=[unit_result],
        ce_threshold=0.3,
    )

    assert validation.repair_valid_causal_pass
    assert validation.policy_strong_repair_valid_pass
    assert validation.full_success_repair_pass
    assert validation.full_success_policy_repair_pass
    assert validation.to_dict()["full_success_repair_pass"] is True


def test_demo_only_repair_is_not_policy_strong_pass() -> None:
    unit = CausalUnit(
        unit_id="semantic_anchor_demo_only",
        kind="goal_predicate_anchor",
        interval=(10, 15),
    )
    unit_result = CausalUnitResult(
        unit=unit,
        base_same_failure_rate=1.0,
        ablated_same_failure_rate=0.0,
        causal_effect=1.0,
        is_causal_core=False,
        is_necessity_core=True,
        repair_pass=False,
        demo_existence_repair_pass=True,
        repair_evidence={
            "repair_pass": False,
            "policy_repair_pass": False,
            "success_or_demo_repair_pass": True,
            "demo_existence_repair_pass": True,
        },
        repair_evaluations=(
            {
                "source": "success_or_demo_nn_repair",
                "repair_pass": True,
                "evaluation": {"success": True},
                "repair_evidence": {"repair_pass": True, "success": True},
            },
        ),
        best_counterfactual={"strategy": "hold"},
    )

    validation = make_causal_validation_result(
        base_same_failure_rate=1.0,
        unit_results=[unit_result],
        ce_threshold=0.3,
    )

    assert validation.same_failure_necessity_pass
    assert validation.demo_existence_repair_pass
    assert validation.full_success_demo_repair_pass
    assert not validation.policy_strong_repair_valid_pass
    assert not validation.repair_valid_causal_pass
    assert not validation.passed


def test_build_causal_units_expands_before_minimal_slice() -> None:
    actions = np.zeros((80, 7), dtype=np.float32)
    actions[30:34, 6] = 1.0
    snapshots = [
        _snapshot(t, (0.001 * t, 0.0, 0.0))
        for t in range(81)
    ]
    signature = FailureSignature(
        failure_type="wrong_placement",
        failed_goal_predicates=("On cup_1 plate_region",),
        affected_objects=("cup_1",),
        anchor_start=60,
        anchor_end=62,
        semantic_quality="full",
        confidence=0.9,
    )

    candidate = type("Candidate", (), {"span_start": 60, "span_end": 62})()
    units = build_causal_units(
        candidate, actions, snapshots, signature, chunk_size=5, context_before=30, max_units=20
    )

    assert any(unit.interval[0] < 60 for unit in units)
    assert any(unit.kind == "gripper_transition" for unit in units)


def test_multimodal_units_include_state_anchor_and_degraded_contact() -> None:
    actions = np.zeros((40, 7), dtype=np.float32)
    actions[10:13, 6] = 1.0
    snapshots = [
        _snapshot(
            t,
            (0.0 + 0.002 * max(0, t - 12), 0.0, 0.0),
            eef_pos=(0.03, 0.0, 0.0),
        )
        for t in range(41)
    ]
    signature = FailureSignature(
        failure_type="wrong_placement",
        failed_goal_predicates=("On cup_1 plate_region",),
        affected_objects=("cup_1",),
        anchor_start=16,
        anchor_end=20,
        semantic_quality="full",
        confidence=0.9,
    )

    candidate = type("Candidate", (), {"span_start": 18, "span_end": 20})()
    units = build_causal_units(
        candidate,
        actions,
        snapshots,
        signature,
        chunk_size=5,
        context_before=20,
        max_units=40,
    )

    assert any(unit.kind == "state_anchor_unit" for unit in units)
    contact_units = [unit for unit in units if unit.kind == "contact_event"]
    assert contact_units
    assert any(
        unit.evidence.get("contact_quality") == "degraded_proximity"
        for unit in contact_units
    )
    assert all(
        not unit.evidence.get("object_teleport_oracle_allowed", False)
        for unit in units
        if unit.kind == "state_anchor_unit"
    )
