from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1] / "code"
if str(CODE_ROOT) in sys.path:
    sys.path.remove(str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT))

from causal_failure_predicates import (
    CausalUnit,
    CausalUnitResult,
    FailureSignature,
    SameFailureResult,
    StateSnapshot,
    build_global_multimodal_units,
    build_k_minimal_causal_sets,
    contact_records_from_env,
    make_causal_validation_result,
)
from edd_types import CandidateSlice
from pi05_natural_failure_probe import (
    Pi05Rollout,
    ReplayEvaluation,
    _apply_visual_policy_intervention,
    _counterfactual_repair_evidence,
    _rule_language_intervention,
    _replay_cache_key,
    _same_failure_rate_bounds,
    _sequential_trial_stop_reason,
    _trial_repair_pass,
    _restage_replay_evaluation,
    _visual_policy_mask_intervention,
)
from risk_critic_export import _strict_causal_pass
from slice_review_export import (
    _coordinate_validation_for_rollout,
    _load_rollout_archive,
    _summarize_recorded_repair_evidence,
)


def _signature() -> FailureSignature:
    return FailureSignature(
        failure_type="wrong_object",
        failed_goal_predicates=("In white_yellow_mug_1 microwave_1_heating_region",),
        affected_objects=("white_yellow_mug_1",),
        anchor_start=10,
        anchor_end=20,
        semantic_quality="full",
        confidence=0.9,
    )


def _unit_result(unit: CausalUnit, **flags: bool) -> CausalUnitResult:
    repair_evaluations = []
    if flags.get("raw_policy_repair_pass"):
        repair_evaluations.append(
            {
                "source": "policy_replan_from_pre_state",
                "repair_pass": True,
                "evaluation": {"success": True},
                "repair_evidence": {"repair_pass": True, "success": True},
            }
        )
    if flags.get("language_phrase_repair_pass"):
        repair_evaluations.append(
            {
                "source": "policy_language_disambiguation_repair",
                "repair_pass": True,
                "evaluation": {"success": False},
                "repair_evidence": {"repair_pass": True},
                "language_intervention": {
                    "schema_version": "shed-cfs-language-phrase-intervention-v1",
                    "selected_phrase": {"phrase": "white_yellow_mug_1", "char_span": [8, 26]},
                    "prompt_diff": {"operation": "append_disambiguating_phrase_clauses"},
                },
            }
        )
    if flags.get("demo_existence_repair_pass"):
        repair_evaluations.append(
            {
                "source": "success_or_demo_nn_repair",
                "repair_pass": True,
                "evaluation": {"success": True},
                "repair_evidence": {"repair_pass": True, "success": True},
            }
        )
    policy_pass = bool(
        flags.get("raw_policy_repair_pass")
        or flags.get("language_phrase_repair_pass")
        or flags.get("visual_mask_repair_pass")
    )
    return CausalUnitResult(
        unit=unit,
        base_same_failure_rate=1.0,
        ablated_same_failure_rate=0.2,
        causal_effect=0.8,
        is_causal_core=True,
        is_necessity_core=True,
        repair_pass=policy_pass,
        best_counterfactual={"strategy": "hold"},
        repair_evaluations=tuple(repair_evaluations),
        policy_strong_repair_pass=policy_pass,
        demo_existence_repair_pass=bool(flags.get("demo_existence_repair_pass")),
        raw_policy_repair_pass=bool(flags.get("raw_policy_repair_pass")),
        language_phrase_repair_pass=bool(flags.get("language_phrase_repair_pass")),
        visual_mask_repair_pass=bool(flags.get("visual_mask_repair_pass")),
    )


def test_v4_top_k_keeps_three_layers_and_ranked_sources() -> None:
    unit = CausalUnit(
        unit_id="contact_010_020",
        kind="contact_event",
        interval=(10, 20),
        evidence={"contact_quality": "mujoco_contact", "hard_contact_evidence": True},
    )
    result = _unit_result(unit, raw_policy_repair_pass=True)
    validation = make_causal_validation_result(1.0, [result], ce_threshold=0.3)

    assert validation.raw_policy_repair_valid_pass
    assert not validation.language_phrase_repair_valid_pass
    assert validation.k_minimal_causal_sets[0]["bounded_minimal"]
    assert validation.k_minimal_causal_sets[0]["repair_sources"] == ["policy_raw"]
    assert validation.k_minimal_causal_sets[0]["drop_one_unit_checks"][0]["causal_effect"] == 0.8


def test_language_phrase_unit_is_part_of_global_multimodal_units() -> None:
    unit = CausalUnit(
        unit_id="action_010_015",
        kind="action_chunk",
        interval=(10, 15),
        evidence={},
    )
    result = _unit_result(unit, language_phrase_repair_pass=True)
    units = build_global_multimodal_units([result])
    kinds = {item["kind"] for item in units}
    assert "action_chunk" in kinds
    assert "language_phrase" in kinds

    sets = build_k_minimal_causal_sets(1.0, [result], top_k=5, ce_threshold=0.3)
    assert sets[0]["repair_sources"] == ["policy_language_phrase"]
    assert any(unit["kind"] == "language_phrase" for unit in sets[0]["units"])


def test_top_k_prefers_policy_strong_over_demo_existence() -> None:
    demo_unit = CausalUnit(
        unit_id="demo_short_010_011",
        kind="action_chunk",
        interval=(10, 11),
        evidence={},
    )
    raw_unit = CausalUnit(
        unit_id="raw_long_010_020",
        kind="action_chunk",
        interval=(10, 20),
        evidence={},
    )
    sets = build_k_minimal_causal_sets(
        1.0,
        [
            _unit_result(demo_unit, demo_existence_repair_pass=True),
            _unit_result(raw_unit, raw_policy_repair_pass=True),
        ],
        top_k=2,
        ce_threshold=0.3,
    )
    assert sets[0]["repair_sources"] == ["policy_raw"]
    assert sets[0]["policy_strong_repair_valid"]


def test_contact_records_from_mujoco_like_env_are_hard_evidence() -> None:
    class Contact:
        geom1 = 0
        geom2 = 1
        pos = np.array([0.1, 0.2, 0.3])
        dist = -0.004

    class Data:
        ncon = 1
        contact = [Contact()]

    class Model:
        geom_bodyid = [0, 1]

        def geom_id2name(self, idx: int) -> str:
            return ["gripper0_finger", "white_yellow_mug_1_collision"][idx]

        def body_id2name(self, idx: int) -> str:
            return ["gripper0", "white_yellow_mug_1"][idx]

    env = SimpleNamespace(sim=SimpleNamespace(data=Data(), model=Model()))
    records = contact_records_from_env(env)
    assert records[0]["contact_quality"] == "mujoco_contact"
    assert records[0]["body2"] == "white_yellow_mug_1"
    assert records[0]["distance"] == -0.004


def test_phrase_and_visual_interventions_are_explicit() -> None:
    signature = _signature()
    rollout = Pi05Rollout(
        task_suite_name="libero_10",
        task_id=1,
        init_state_id=15,
        task_language="Put the white_yellow_mug_1 into the microwave.",
        target_key="white_yellow_mug_1",
        target_key_trace=[],
        actions=np.zeros((30, 7), dtype=np.float32),
        states_before_action=[],
        snapshots=[
            StateSnapshot(
                t=0,
                success=False,
                goal_truth={},
                object_positions={"white_yellow_mug_1": (0.0, 0.5, 0.8)},
            )
        ],
        goal_predicates=tuple(),
        semantic_quality="full",
        failure_signature=signature,
        distance_trace=[0.0, 1.0],
        success=False,
        done_step=None,
    )
    unit = SimpleNamespace(
        unit_id="target_object_000",
        interval=(0, 5),
        evidence={"target_object": "white_yellow_mug_1"},
    )

    language = _rule_language_intervention(rollout, unit)
    assert language["schema_version"] == "shed-cfs-language-phrase-intervention-v1"
    assert language["selected_phrase"]["char_span"] is not None

    visual = _visual_policy_mask_intervention(
        SimpleNamespace(enable_visual_policy_mask=True, resize_size=224),
        rollout,
        unit,
    )
    assert visual["schema_version"] == "shed-cfs-visual-grounding-mask-v1"
    assert visual["applied_to_policy_input"]
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    changed = _apply_visual_policy_intervention(image, visual, "agent")
    assert int(changed.sum()) > 0


def test_v4_strict_pass_uses_separated_policy_sources() -> None:
    report = {
        "schema_version": "shed-cfs-causal-v4-global-multimodal",
        "policy_raw_repair_valid_pass": False,
        "policy_language_phrase_repair_valid_pass": True,
        "language_phrase_core_units": [
            {
                "unit": {"unit_id": "language_action_010_015"},
                "language_phrase_repair_pass": True,
                "repair_evidence": {"language_phrase_repair_pass": True},
            }
        ],
        "causal_validation": {},
    }
    assert _strict_causal_pass(report)


def test_replay_restage_keeps_cache_result_but_updates_stage() -> None:
    signature = _signature()
    evidence = SameFailureResult(
        same_failure=True,
        score=1.0,
        type_match=True,
        failed_predicate_jaccard=1.0,
        affected_object_jaccard=1.0,
        evidence_score=1.0,
        threshold=0.75,
        reasons=tuple(),
    )
    evaluation = ReplayEvaluation(
        candidate=CandidateSlice.from_window(10, 20, n_steps=50, level="old_stage"),
        same_failure=True,
        same_failure_rate=1.0,
        failure_rate=1.0,
        trials=5,
        success=False,
        start_distance=0.0,
        end_distance=1.0,
        distance_delta=1.0,
        steps=10,
        signature=signature,
        same_failure_evidence=evidence,
        planned_trials=5,
        executed_trials=5,
        same_failure_count=5,
        same_failure_rate_lower_bound=1.0,
        same_failure_rate_upper_bound=1.0,
        cache_key="abc",
    )
    restaged = _restage_replay_evaluation(
        evaluation, 10, 20, n_steps=50, stage_level="new_stage", from_cache=True
    )
    assert restaged.from_cache
    assert restaged.cache_key == "abc"
    assert restaged.candidate.level == "new_stage"
    assert evaluation.candidate.level == "old_stage"


def test_exact_trial_bounds_preserve_threshold_decision() -> None:
    assert _same_failure_rate_bounds(4, executed=4, planned=5) == (0.8, 1.0)
    assert _same_failure_rate_bounds(0, executed=2, planned=5) == (0.0, 0.6)


def test_causal_effect_early_stop_waits_until_ce_is_decidable() -> None:
    assert (
        _sequential_trial_stop_reason(
            0.0,
            0.6,
            0.8,
            objective="same_failure",
        )
        == "same_failure_threshold_impossible"
    )
    assert (
        _sequential_trial_stop_reason(
            0.0,
            0.6,
            0.8,
            objective="causal_effect",
            ce_reference_rate=0.8,
            ce_threshold=0.3,
        )
        is None
    )
    assert (
        _sequential_trial_stop_reason(
            0.0,
            0.4,
            0.8,
            objective="causal_effect",
            ce_reference_rate=0.8,
            ce_threshold=0.3,
        )
        == "causal_effect_threshold_already_met"
    )
    assert (
        _sequential_trial_stop_reason(
            0.6,
            1.0,
            0.8,
            objective="causal_effect",
            ce_reference_rate=0.8,
            ce_threshold=0.3,
        )
        == "causal_effect_threshold_impossible"
    )


def test_repair_valid_early_stop_is_exact_for_k5() -> None:
    assert (
        _sequential_trial_stop_reason(
            0.8,
            1.0,
            0.8,
            objective="repair_valid",
        )
        == "repair_valid_threshold_already_met"
    )
    assert (
        _sequential_trial_stop_reason(
            0.0,
            0.6,
            0.8,
            objective="repair_valid",
        )
        == "repair_valid_threshold_impossible"
    )
    assert (
        _sequential_trial_stop_reason(
            0.2,
            0.8,
            0.8,
            objective="repair_valid",
        )
        is None
    )


def test_repair_evidence_uses_aggregate_trial_outcomes() -> None:
    signature = _signature()
    evidence = SameFailureResult(
        same_failure=False,
        score=0.0,
        type_match=False,
        failed_predicate_jaccard=0.0,
        affected_object_jaccard=0.0,
        evidence_score=0.0,
        threshold=0.75,
        reasons=("different",),
    )
    evaluation = ReplayEvaluation(
        candidate=CandidateSlice.from_window(10, 20, n_steps=50, level="policy_repair"),
        same_failure=False,
        same_failure_rate=0.0,
        failure_rate=0.2,
        trials=5,
        success=False,
        start_distance=0.0,
        end_distance=0.0,
        distance_delta=0.0,
        steps=10,
        signature=signature,
        same_failure_evidence=evidence,
        planned_trials=5,
        executed_trials=5,
        trial_outcomes=tuple(
            {
                "success": i < 4,
                "failed_goal_predicates": []
                if i < 4
                else ["In white_yellow_mug_1 microwave_1_heating_region", "extra"],
                "failed_goal_count": 0 if i < 4 else 2,
                "goal_progress": 1 if i < 4 else 0,
            }
            for i in range(5)
        ),
    )
    repair = _counterfactual_repair_evidence(signature, evaluation)
    assert repair["repair_pass"]
    assert repair["repair_pass_rate"] == 0.8
    assert repair["success"]


def test_repair_evidence_does_not_promote_single_trial_to_k5_pass() -> None:
    signature = _signature()
    evidence = SameFailureResult(
        same_failure=False,
        score=0.0,
        type_match=False,
        failed_predicate_jaccard=0.0,
        affected_object_jaccard=0.0,
        evidence_score=0.0,
        threshold=0.75,
        reasons=("different",),
    )
    evaluation = ReplayEvaluation(
        candidate=CandidateSlice.from_window(10, 20, n_steps=50, level="policy_repair"),
        same_failure=False,
        same_failure_rate=0.0,
        failure_rate=0.0,
        trials=1,
        success=True,
        start_distance=0.0,
        end_distance=0.0,
        distance_delta=0.0,
        steps=10,
        signature=signature,
        same_failure_evidence=evidence,
        planned_trials=5,
        executed_trials=1,
        trial_outcomes=(
            {
                "success": True,
                "failed_goal_predicates": [],
                "failed_goal_count": 0,
                "goal_progress": 1,
                "affected_objects": [],
            },
        ),
    )
    repair = _counterfactual_repair_evidence(signature, evaluation)
    assert not repair["repair_pass"]
    assert repair["repair_pass_rate"] == 0.2


def test_trial_repair_pass_matches_non_worsening_improvement_rule() -> None:
    signature = _signature()
    assert _trial_repair_pass(
        signature,
        {
            "success": False,
            "failed_goal_predicates": [],
            "failed_goal_count": 0,
            "goal_progress": 1,
            "affected_objects": ["white_yellow_mug_1"],
        },
    )
    assert not _trial_repair_pass(
        signature,
        {
            "success": False,
            "failed_goal_predicates": [
                "In white_yellow_mug_1 microwave_1_heating_region",
                "extra_future_goal",
            ],
            "failed_goal_count": 2,
            "goal_progress": 0,
            "affected_objects": ["white_yellow_mug_1", "wrong_object"],
        },
    )


def test_repair_cache_key_ignores_report_end_but_same_failure_key_keeps_it() -> None:
    args = SimpleNamespace(
        continuation="recorded",
        accept_same_failure_rate=0.8,
        same_failure_threshold=0.75,
        replay_evaluation_timeout_seconds=0.0,
        event_window=24,
        replan_steps=5,
    )
    rollout = Pi05Rollout(
        task_suite_name="libero_10",
        task_id=1,
        init_state_id=2,
        task_language="task",
        target_key="object",
        target_key_trace=[],
        actions=np.zeros((50, 7), dtype=np.float32),
        states_before_action=[],
        snapshots=[],
        goal_predicates=tuple(),
        semantic_quality="full",
        failure_signature=_signature(),
        distance_trace=[0.0] * 51,
        success=False,
        done_step=None,
        reset_seed=7,
    )
    repair_key_a = _replay_cache_key(
        args,
        rollout,
        10,
        12,
        rollout.failure_signature,
        client=object(),
        action_replacements=None,
        external_actions=None,
        policy_from_step=10,
        prompt_override=None,
        visual_intervention=None,
        trials=5,
        early_stop_objective="repair_valid",
    )
    repair_key_b = _replay_cache_key(
        args,
        rollout,
        10,
        30,
        rollout.failure_signature,
        client=object(),
        action_replacements=None,
        external_actions=None,
        policy_from_step=10,
        prompt_override=None,
        visual_intervention=None,
        trials=5,
        early_stop_objective="repair_valid",
    )
    same_key_a = _replay_cache_key(
        args,
        rollout,
        10,
        12,
        rollout.failure_signature,
        client=None,
        action_replacements=None,
        external_actions=None,
        policy_from_step=None,
        prompt_override=None,
        visual_intervention=None,
        trials=5,
    )
    same_key_b = _replay_cache_key(
        args,
        rollout,
        10,
        30,
        rollout.failure_signature,
        client=None,
        action_replacements=None,
        external_actions=None,
        policy_from_step=None,
        prompt_override=None,
        visual_intervention=None,
        trials=5,
    )
    assert repair_key_a == repair_key_b
    assert same_key_a != same_key_b


def test_review_coordinate_validation_rejects_fresh_rollout_length_mismatch() -> None:
    case = SimpleNamespace(
        rollout_length=520,
        minimal_start=402,
        minimal_end=404,
        repair_replay_start=336,
        repair_replay_end=345,
    )
    rollout = SimpleNamespace(
        actions=np.zeros((278, 7), dtype=np.float32),
        states_before_action=[np.zeros(3, dtype=np.float32) for _ in range(279)],
    )
    validation = _coordinate_validation_for_rollout(case, rollout, (336, 345))

    assert validation["coordinate_mismatch"]
    assert not validation["strict_coordinate_match"]
    assert validation["report_action_length"] == 520
    assert validation["fresh_action_length"] == 278
    assert "fresh_rollout_length_differs_from_report" in validation["reasons"]
    destructive = {
        item["name"]: item for item in validation["interval_checks"]
    }["destructive_core_unit"]
    assert not destructive["in_bounds"]
    assert "end_exceeds_action_trace" in destructive["reasons"]


def test_recorded_repair_summary_counts_only_strict_coordinate_matches() -> None:
    case = SimpleNamespace(
        failed_goals=["goal_a", "goal_b"],
        full_success_repair_pass=True,
    )
    mismatch_meta = {
        "available": True,
        "success": True,
        "strict_source_aware_evidence": False,
        "coordinate_mismatch": True,
        "failure_signature": {"failed_goal_predicates": [], "failure_type": "success"},
        "video_path": "/tmp/mismatch.mp4",
    }
    strict_meta = {
        "available": True,
        "success": False,
        "strict_source_aware_evidence": True,
        "coordinate_mismatch": False,
        "failure_signature": {
            "failed_goal_predicates": ["goal_a"],
            "failure_type": "improved",
        },
        "video_path": "/tmp/strict.mp4",
    }

    summary = _summarize_recorded_repair_evidence(
        case,
        [
            {"trial": 0, "raw_policy_repair_replay": mismatch_meta},
            {"trial": 1, "raw_policy_repair_replay": strict_meta},
        ],
        [],
    )

    assert summary["observed_any_success"]
    assert not summary["any_success"]
    assert summary["any_improvement"]
    assert summary["coordinate_mismatch_count"] == 1
    assert summary["reported_vs_recorded_mismatch"]


def test_recorded_error_continuation_is_not_counted_as_repair_success() -> None:
    case = SimpleNamespace(
        failed_goals=["goal_a"],
        full_success_repair_pass=True,
    )
    recorded_error_meta = {
        "available": True,
        "variant": "recorded_error_continuation",
        "success": True,
        "counts_as_repair_evidence": False,
        "strict_source_aware_evidence": True,
        "coordinate_mismatch": False,
        "reason": "policy repair re-query disabled",
        "failure_signature": {"failed_goal_predicates": [], "failure_type": "success"},
        "video_path": "/tmp/recorded_error.mp4",
    }

    summary = _summarize_recorded_repair_evidence(
        case,
        [{"trial": 0, "raw_policy_repair_replay": recorded_error_meta}],
        [],
    )

    assert not summary["any_success"]
    assert not summary["observed_any_success"]
    assert summary["recorded_error_continuation_count"] == 1
    assert summary["non_repair_continuations"][0]["source_slot"] == "raw_policy"
    assert summary["reported_vs_recorded_mismatch"]


def test_review_loads_exact_rollout_archive(tmp_path) -> None:
    archive = tmp_path / "rollout_archive.npz"
    metadata = {
        "schema_version": "pi05-rollout-archive-v1",
        "task_suite_name": "libero_10",
        "task_id": 8,
        "init_state_id": 39,
        "task_language": "put both moka pots on the stove",
        "target_key": "moka_pot_1",
        "target_key_trace": ["moka_pot_1"],
        "distance_trace": [0.3, 0.2, 0.1],
        "success": False,
        "done_step": None,
        "reset_seed": 57,
        "length": 2,
        "failure_signature": {
            "failure_type": "wrong_placement",
            "failed_goal_predicates": ["On moka_pot_1 flat_stove_1_cook_region"],
            "affected_objects": ["moka_pot_1"],
            "anchor_window": [1, 2],
            "semantic_quality": "full",
            "confidence": 0.95,
        },
    }
    np.savez_compressed(
        archive,
        actions=np.ones((2, 7), dtype=np.float32),
        states_before_action=np.ones((2, 5), dtype=np.float64),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    case = SimpleNamespace(
        task_id=8,
        init_state_id=39,
        task_language=metadata["task_language"],
        report={
            "selected_failed_rollout": {"rollout_archive_path": str(archive)},
            "original_failure_signature": metadata["failure_signature"],
        },
    )

    rollout = _load_rollout_archive(case)

    assert rollout is not None
    assert rollout.length == 2
    assert rollout.loaded_from_archive
    assert rollout.failure_signature.failure_type == "wrong_placement"
    assert np.asarray(rollout.states_before_action).shape == (2, 5)


def test_hierarchical_pruning_trace_is_reported() -> None:
    validation = make_causal_validation_result(
        1.0,
        [],
        ce_threshold=0.3,
        hierarchical_pruning_trace=[
            {"group": "contact", "status": "pruned", "unit_ids": ["contact_001"]}
        ],
    )
    payload = validation.to_dict()
    assert payload["hierarchical_pruning_trace"][0]["group"] == "contact"
    assert payload["hierarchical_pruning_trace"][0]["status"] == "pruned"
