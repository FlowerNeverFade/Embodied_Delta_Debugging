from causal_failure_predicates import (
    GoalPredicate,
    StateSnapshot,
    compare_failure_signatures,
    infer_failure_signature,
)
from custom_tasks import long_breakfast_curriculum_v2 as task


class FakeEnv:
    def __init__(self, truth):
        self.truth = {tuple(k): bool(v) for k, v in truth.items()}

    def _eval_predicate(self, raw):
        return bool(self.truth.get(tuple(raw), False))


def _snapshot(t: int, goal_truth: dict) -> StateSnapshot:
    return StateSnapshot(
        t=t,
        success=False,
        goal_truth=goal_truth,
        object_positions={
            "white_yellow_mug_1": (0.0, 0.0, 0.0),
            "akita_black_bowl_1": (0.1, 0.0, 0.0),
        },
        eef_pos=(0.0, 0.0, 0.1),
    )


def test_curriculum_has_four_parseable_task_specs():
    suite = task.make_task_suite(task.SUITE_NAME)
    assert suite is not None
    assert suite.n_tasks == 4
    assert task.task_metadata()["layout_revision"] == "v2.1-clean-stage01"
    assert suite.get_task(0).language.startswith("Put the yellow and white mug")
    assert "open the microwave" not in suite.get_task(0).language.lower()
    for task_id in range(4):
        assert task.custom_bddl_path(task.SUITE_NAME, task_id).exists()
        assert task.task_spec(task_id).stages[0].key == "mug_in_microwave"


def test_l1_stage_oracle_does_not_credit_initial_open_as_progress():
    tracker = task.StageOracleTracker(task_id=0)
    env = FakeEnv({("open", "microwave_1"): True})
    truth = tracker.update(env)
    assert truth["Stage01 mug_in_microwave"] is False
    assert truth["Stage02 microwave_closed"] is False
    assert tracker.completed_count == 0

    env.truth[("in", "white_yellow_mug_1", "microwave_1_heating_region")] = True
    truth = tracker.update(env)
    assert truth["Stage01 mug_in_microwave"] is True
    assert truth["Stage02 microwave_closed"] is False


def test_stage_aware_target_uses_earliest_unfinished_stage():
    obs = {
        "robot0_eef_pos": [0.0, 0.0, 0.0],
        "white_yellow_mug_1_pos": [0.1, 0.0, 0.0],
        "akita_black_bowl_1_pos": [0.2, 0.0, 0.0],
    }
    snap = _snapshot(0, {"Stage01 mug_in_microwave": False})
    assert (
        task.target_key_for_snapshot(task.SUITE_NAME, 1, snap, obs)
        == "white_yellow_mug_1_pos"
    )

    snap = _snapshot(
        1,
        {
            "Stage01 mug_in_microwave": True,
            "Stage02 microwave_closed": True,
            "Stage03 bowl_in_bottom_drawer": False,
        },
    )
    assert (
        task.target_key_for_snapshot(task.SUITE_NAME, 1, snap, obs)
        == "akita_black_bowl_1_pos"
    )


def test_stage_aware_same_failure_keys_on_earliest_failed_stage():
    predicates = task.stage_predicates_for_suite(task.SUITE_NAME, 1)
    ref_truth = {
        "Stage01 mug_in_microwave": True,
        "Stage02 microwave_closed": False,
        "Stage03 bowl_in_bottom_drawer": False,
        "Stage04 bottom_drawer_closed": False,
        task.ORDER_VALID_LABEL: True,
    }
    cand_truth = dict(ref_truth)
    other_truth = {
        "Stage01 mug_in_microwave": True,
        "Stage02 microwave_closed": True,
        "Stage03 bowl_in_bottom_drawer": False,
        "Stage04 bottom_drawer_closed": False,
        task.ORDER_VALID_LABEL: True,
    }
    ref = infer_failure_signature([_snapshot(0, ref_truth), _snapshot(1, ref_truth)], predicates)
    cand = infer_failure_signature([_snapshot(0, cand_truth), _snapshot(1, cand_truth)], predicates)
    other = infer_failure_signature([_snapshot(0, other_truth), _snapshot(1, other_truth)], predicates)

    assert compare_failure_signatures(ref, cand).same_failure
    mismatch = compare_failure_signatures(ref, other)
    assert not mismatch.same_failure
    assert any("stage_mismatch" in reason for reason in mismatch.reasons)


def test_scripted_expert_is_repair_only_and_pick_place_gated():
    pick_snapshot = _snapshot(0, {"Stage01 mug_in_microwave": False})
    meta = task.expert_repair_metadata(task.SUITE_NAME, 0, pick_snapshot)
    assert meta["available"] is True
    assert meta["expert_repair_only"] is True
    assert meta["stage"]["repair_kind"] == "pick_place"

    close_snapshot = _snapshot(
        1,
        {
            "Stage01 mug_in_microwave": True,
            "Stage02 microwave_closed": False,
        },
    )
    meta = task.expert_repair_metadata(task.SUITE_NAME, 0, close_snapshot)
    assert meta["available"] is False
    assert meta["reason"] == "stage_repair_kind_not_implemented"


def test_initial_state_quality_rejects_tipped_primary_mug():
    obs = {
        "white_yellow_mug_1_pos": [0.0, 0.0, 0.90],
        "white_yellow_mug_1_quat": [0.45, 0.50, 0.55, -0.50],
        "moka_pot_1_pos": [0.2, 0.2, 0.9],
        "robot0_eef_pos": [0.0, 0.0, 1.0],
    }
    snap = _snapshot(0, {"Stage01 mug_in_microwave": False})
    quality = task.initial_state_quality(task.SUITE_NAME, 3, obs, env=None, snapshot=snap)
    assert quality["valid"] is False
    assert "primary_target_not_upright" in quality["reasons"]


def test_initial_state_quality_accepts_clean_stage01_mug():
    obs = {
        "white_yellow_mug_1_pos": [0.0, 0.0, 0.90],
        "white_yellow_mug_1_quat": [0.0, 0.0, 0.70710678, -0.70710678],
        "moka_pot_1_pos": [0.2, 0.2, 0.9],
        "robot0_eef_pos": [0.0, 0.0, 1.0],
    }
    snap = _snapshot(0, {"Stage01 mug_in_microwave": False})
    quality = task.initial_state_quality(task.SUITE_NAME, 3, obs, env=None, snapshot=snap)
    assert quality["valid"] is True
    assert quality["reasons"] == []
