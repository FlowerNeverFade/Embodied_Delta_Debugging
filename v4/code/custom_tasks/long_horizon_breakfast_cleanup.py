from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from causal_failure_predicates import GoalPredicate, StateSnapshot


SUITE_NAME = "libero_long_breakfast_cleanup"
TASK_NAME = "KITCHEN_LONG_HORIZON_BREAKFAST_CLEANUP"
TASK_LANGUAGE = (
    "Open the microwave, put the yellow and white mug inside it, close the microwave, "
    "open the bottom drawer, put the black bowl inside it, close the drawer, turn on "
    "the stove, put both moka pots on the stove, then turn the stove off."
)
BDDL_PATH = (
    Path(__file__).resolve().parent
    / "long_horizon_breakfast_cleanup"
    / f"{TASK_NAME}.bddl"
)
DEFAULT_NUM_INIT_STATES = 50
DEFAULT_MAX_STEPS = 900
ORDER_VALID_LABEL = "Stageorder valid_sequence_so_far"


@dataclass(frozen=True)
class CustomTask:
    name: str
    language: str
    problem: str
    problem_folder: str
    bddl_file: str
    init_states_file: str


@dataclass(frozen=True)
class StageDefinition:
    index: int
    key: str
    label: str
    predicate: Tuple[str, ...]
    description: str


STAGES: Tuple[StageDefinition, ...] = (
    StageDefinition(
        1,
        "microwave_open",
        "Stage01 microwave_open",
        ("open", "microwave_1"),
        "Microwave is open before inserting the mug.",
    ),
    StageDefinition(
        2,
        "mug_in_microwave",
        "Stage02 mug_in_microwave",
        ("in", "white_yellow_mug_1", "microwave_1_heating_region"),
        "Yellow-white mug is inside the microwave heating region.",
    ),
    StageDefinition(
        3,
        "microwave_closed",
        "Stage03 microwave_closed",
        ("close", "microwave_1"),
        "Microwave is closed after the mug is inside.",
    ),
    StageDefinition(
        4,
        "bottom_drawer_open",
        "Stage04 bottom_drawer_open",
        ("open", "white_cabinet_1_bottom_region"),
        "Bottom drawer is open before inserting the black bowl.",
    ),
    StageDefinition(
        5,
        "bowl_in_bottom_drawer",
        "Stage05 bowl_in_bottom_drawer",
        ("in", "akita_black_bowl_1", "white_cabinet_1_bottom_region"),
        "Black bowl is inside the cabinet bottom drawer.",
    ),
    StageDefinition(
        6,
        "bottom_drawer_closed",
        "Stage06 bottom_drawer_closed",
        ("close", "white_cabinet_1_bottom_region"),
        "Bottom drawer is closed after the bowl is inside.",
    ),
    StageDefinition(
        7,
        "stove_turned_on",
        "Stage07 stove_turned_on",
        ("turnon", "flat_stove_1"),
        "Stove is turned on after storage subtasks are finished.",
    ),
    StageDefinition(
        8,
        "first_moka_pot_on_stove",
        "Stage08 first_moka_pot_on_stove",
        ("custom_any_on", "moka_pot_1", "moka_pot_2", "flat_stove_1_cook_region"),
        "At least one moka pot is on the stove cook region.",
    ),
    StageDefinition(
        9,
        "both_moka_pots_on_stove",
        "Stage09 both_moka_pots_on_stove",
        ("custom_both_on", "moka_pot_1", "moka_pot_2", "flat_stove_1_cook_region"),
        "Both moka pots are on the stove cook region.",
    ),
    StageDefinition(
        10,
        "stove_turned_off_after_pots",
        "Stage10 stove_turned_off_after_pots",
        ("turnoff", "flat_stove_1"),
        "Stove is turned off after both moka pots are placed.",
    ),
)

STAGE_PREDICATES: Tuple[GoalPredicate, ...] = tuple(
    GoalPredicate((stage.label,)) for stage in STAGES
) + (GoalPredicate((ORDER_VALID_LABEL,)),)


class LongBreakfastTaskSuite:
    name = SUITE_NAME
    n_tasks = 1

    def __init__(self, num_init_states: int = DEFAULT_NUM_INIT_STATES):
        self.tasks = [
            CustomTask(
                name=TASK_NAME,
                language=TASK_LANGUAGE,
                problem="Libero",
                problem_folder="custom_tasks/long_horizon_breakfast_cleanup",
                bddl_file=BDDL_PATH.name,
                init_states_file="generated_from_bddl_reset",
            )
        ]
        self.num_init_states = int(num_init_states)

    def get_task(self, i: int) -> CustomTask:
        if int(i) != 0:
            raise IndexError(f"{SUITE_NAME} has one task, got task_id={i}")
        return self.tasks[0]

    def get_task_init_states(self, i: int) -> List[None]:
        self.get_task(i)
        return [None for _ in range(self.num_init_states)]


class StageOracleTracker:
    """Monotonic stage-progress oracle for the custom long-horizon task."""

    def __init__(self, initial_goal_truth: Optional[Dict[str, bool]] = None):
        initial_goal_truth = initial_goal_truth or {}
        self.completed = {
            stage.label: bool(initial_goal_truth.get(stage.label, False))
            for stage in STAGES
        }
        self.order_valid = bool(initial_goal_truth.get(ORDER_VALID_LABEL, True))
        self.order_violation_reasons: List[str] = []

    @classmethod
    def from_snapshot(cls, snapshot: Optional[StateSnapshot]) -> "StageOracleTracker":
        return cls(None if snapshot is None else snapshot.goal_truth)

    @property
    def completed_count(self) -> int:
        return sum(1 for stage in STAGES if self.completed.get(stage.label, False))

    def update(self, env) -> Dict[str, bool]:
        instant = instantaneous_stage_truth(env)
        self._detect_order_violations(instant)

        # Advance at most one stage per environment step. This preserves a
        # meaningful temporal trace when multiple predicates happen to be true
        # at the same state, e.g. an initially-open drawer.
        for stage in STAGES:
            if self.completed.get(stage.label, False):
                continue
            if all(self.completed.get(prev.label, False) for prev in STAGES[: stage.index - 1]):
                if bool(instant.get(stage.label, False)):
                    self.completed[stage.label] = True
            break

        truth = {stage.label: bool(self.completed.get(stage.label, False)) for stage in STAGES}
        truth[ORDER_VALID_LABEL] = bool(self.order_valid)
        return truth

    def _detect_order_violations(self, instant: Dict[str, bool]) -> None:
        stage = self.completed
        violations: List[Tuple[str, bool]] = [
            (
                "microwave_closed_before_mug_inserted",
                instant.get("Stage03 microwave_closed", False)
                and not stage.get("Stage02 mug_in_microwave", False),
            ),
            (
                "drawer_closed_before_bowl_inserted",
                instant.get("Stage06 bottom_drawer_closed", False)
                and not stage.get("Stage05 bowl_in_bottom_drawer", False),
            ),
            (
                "moka_pot_on_stove_before_stove_turned_on",
                instant.get("Stage08 first_moka_pot_on_stove", False)
                and not stage.get("Stage07 stove_turned_on", False),
            ),
            (
                "stove_turned_off_before_both_moka_pots_placed",
                stage.get("Stage07 stove_turned_on", False)
                and not stage.get("Stage09 both_moka_pots_on_stove", False)
                and instant.get("Stage10 stove_turned_off_after_pots", False),
            ),
        ]
        for reason, active in violations:
            if active and reason not in self.order_violation_reasons:
                self.order_violation_reasons.append(reason)
                self.order_valid = False


def is_custom_suite(task_suite_name: str) -> bool:
    return str(task_suite_name).lower() == SUITE_NAME


def max_steps_for_suite(task_suite_name: str) -> Optional[int]:
    return DEFAULT_MAX_STEPS if is_custom_suite(task_suite_name) else None


def make_task_suite(task_suite_name: str) -> Optional[LongBreakfastTaskSuite]:
    if not is_custom_suite(task_suite_name):
        return None
    return LongBreakfastTaskSuite()


def custom_bddl_path(task_suite_name: str, task_id: int) -> Optional[Path]:
    if not is_custom_suite(task_suite_name):
        return None
    if int(task_id) != 0:
        raise IndexError(f"{SUITE_NAME} has one task, got task_id={task_id}")
    return BDDL_PATH


def deterministic_reset_seed(base_seed: int, init_state_id: int) -> int:
    return int(base_seed) + 1009 * int(init_state_id)


def stage_predicates_for_suite(task_suite_name: str) -> Tuple[GoalPredicate, ...]:
    return STAGE_PREDICATES if is_custom_suite(task_suite_name) else tuple()


def make_stage_tracker(task_suite_name: str, snapshot: Optional[StateSnapshot] = None):
    if not is_custom_suite(task_suite_name):
        return None
    return StageOracleTracker.from_snapshot(snapshot)


def augment_snapshot_with_stage_truth(
    task_suite_name: str,
    snapshot: StateSnapshot,
    tracker: Optional[StageOracleTracker],
    env,
) -> StateSnapshot:
    if tracker is None or not is_custom_suite(task_suite_name):
        return snapshot
    merged = dict(snapshot.goal_truth)
    merged.update(tracker.update(env))
    return replace(snapshot, goal_truth=merged)


def stage_trace_from_snapshots(snapshots: Sequence[StateSnapshot]) -> dict:
    labels = [stage.label for stage in STAGES] + [ORDER_VALID_LABEL]
    truth_by_t = [
        {label: bool(snapshot.goal_truth.get(label, False)) for label in labels}
        for snapshot in snapshots
    ]
    progress_counts = [
        sum(1 for stage in STAGES if row.get(stage.label, False))
        for row in truth_by_t
    ]
    final_truth = truth_by_t[-1] if truth_by_t else {}
    first_completed = {}
    for label in labels:
        first_completed[label] = next(
            (i for i, row in enumerate(truth_by_t) if bool(row.get(label, False))),
            None,
        )
    return {
        "schema_version": "long-breakfast-stage-oracle-v1",
        "task_suite_name": SUITE_NAME,
        "stage_order": [
            {
                "index": stage.index,
                "label": stage.label,
                "predicate": list(stage.predicate),
                "description": stage.description,
            }
            for stage in STAGES
        ],
        "final_truth": final_truth,
        "progress_counts": progress_counts,
        "first_completed_step": first_completed,
        "num_steps_recorded": len(truth_by_t),
    }


def task_metadata() -> dict:
    return {
        "schema_version": "libero-custom-long-task-v1",
        "task_suite_name": SUITE_NAME,
        "task_name": TASK_NAME,
        "task_language": TASK_LANGUAGE,
        "bddl_path": str(BDDL_PATH),
        "default_max_steps": int(DEFAULT_MAX_STEPS),
        "stage_oracle": stage_trace_from_snapshots([])["stage_order"],
        "final_goals": [
            "In white_yellow_mug_1 microwave_1_heating_region",
            "Close microwave_1",
            "In akita_black_bowl_1 white_cabinet_1_bottom_region",
            "Close white_cabinet_1_bottom_region",
            "On moka_pot_1 flat_stove_1_cook_region",
            "On moka_pot_2 flat_stove_1_cook_region",
            "Turnoff flat_stove_1",
        ],
    }


def instantaneous_stage_truth(env) -> Dict[str, bool]:
    return {
        stage.label: _eval_stage_predicate(env, stage.predicate)
        for stage in STAGES
    }


def _base_env(env):
    return getattr(env, "env", env)


def _eval_stage_predicate(env, predicate: Tuple[str, ...]) -> bool:
    head = predicate[0]
    if head == "custom_any_on":
        return any(_eval_stage_predicate(env, ("on", obj, predicate[3])) for obj in predicate[1:3])
    if head == "custom_both_on":
        return all(_eval_stage_predicate(env, ("on", obj, predicate[3])) for obj in predicate[1:3])
    base = _base_env(env)
    try:
        return bool(base._eval_predicate(list(predicate)))
    except Exception:
        return False
