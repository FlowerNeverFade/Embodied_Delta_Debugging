from causal_failure_predicates import GoalPredicate
from custom_tasks import long_horizon_breakfast_cleanup as task


class FakeEnv:
    def __init__(self, truth):
        self.truth = {tuple(k): bool(v) for k, v in truth.items()}

    def _eval_predicate(self, raw):
        return bool(self.truth.get(tuple(raw), False))


def test_stage_predicates_are_label_only():
    labels = [predicate.label for predicate in task.stage_predicates_for_suite(task.SUITE_NAME)]
    assert "Stage01 microwave_open" in labels
    assert "Stageorder valid_sequence_so_far" in labels
    assert all(predicate.args == () for predicate in task.stage_predicates_for_suite(task.SUITE_NAME))


def test_stage_oracle_advances_monotonically_one_stage_at_a_time():
    tracker = task.StageOracleTracker()
    env = FakeEnv({("open", "microwave_1"): True})
    truth = tracker.update(env)
    assert truth["Stage01 microwave_open"] is True
    assert truth["Stage02 mug_in_microwave"] is False
    assert truth["Stageorder valid_sequence_so_far"] is True

    env.truth[("in", "white_yellow_mug_1", "microwave_1_heating_region")] = True
    truth = tracker.update(env)
    assert truth["Stage01 microwave_open"] is True
    assert truth["Stage02 mug_in_microwave"] is True
    assert truth["Stage03 microwave_closed"] is False


def test_stage_oracle_marks_order_violation():
    tracker = task.StageOracleTracker()
    env = FakeEnv(
        {
            ("on", "moka_pot_1", "flat_stove_1_cook_region"): True,
            ("open", "microwave_1"): True,
        }
    )
    truth = tracker.update(env)
    assert truth["Stage01 microwave_open"] is True
    assert truth["Stage08 first_moka_pot_on_stove"] is False
    assert truth["Stageorder valid_sequence_so_far"] is False
    assert "moka_pot_on_stove_before_stove_turned_on" in tracker.order_violation_reasons


def test_custom_suite_metadata_is_available():
    suite = task.make_task_suite(task.SUITE_NAME)
    assert suite is not None
    assert suite.get_task(0).language.startswith("Open the microwave")
    assert len(suite.get_task_init_states(0)) == task.DEFAULT_NUM_INIT_STATES
    assert task.custom_bddl_path(task.SUITE_NAME, 0).exists()
