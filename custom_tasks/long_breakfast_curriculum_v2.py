from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from causal_failure_predicates import GoalPredicate, StateSnapshot


PROJECT_ROOT = Path("/root/autodl-tmp/research/Embodied_Delta_Debugging")
SUITE_NAME = "libero_long_breakfast_curriculum_v2"
LAYOUT_REVISION = "v2.1-clean-stage01"
TASK_FOLDER = PROJECT_ROOT / "custom_tasks" / "long_breakfast_curriculum_v2"
DEFAULT_NUM_INIT_STATES = 50
DEFAULT_MAX_STEPS = 820
ORDER_VALID_LABEL = "Stageorder valid_sequence_so_far"
WHITE_YELLOW_MUG_STABLE_QUAT_XYZW = np.asarray(
    [0.0, 0.0, np.sqrt(0.5), -np.sqrt(0.5)], dtype=np.float64
)


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
    target_objects: Tuple[str, ...] = ()
    target_region: str = ""
    repair_kind: str = ""


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    name: str
    language: str
    bddl_file: str
    stages: Tuple[StageDefinition, ...]
    final_goals: Tuple[str, ...]
    max_steps: int
    starts_with_stove_on: bool = False

    @property
    def bddl_path(self) -> Path:
        return TASK_FOLDER / self.bddl_file


def _stage(
    index: int,
    key: str,
    predicate: Tuple[str, ...],
    description: str,
    target_objects: Sequence[str] = (),
    target_region: str = "",
    repair_kind: str = "",
) -> StageDefinition:
    label = "Stage%02d %s" % (index, key)
    return StageDefinition(
        index=index,
        key=key,
        label=label,
        predicate=tuple(predicate),
        description=description,
        target_objects=tuple(target_objects),
        target_region=target_region,
        repair_kind=repair_kind,
    )


L1_STAGES = (
    _stage(
        1,
        "mug_in_microwave",
        ("in", "white_yellow_mug_1", "microwave_1_heating_region"),
        "Yellow-white mug is inside the microwave heating region.",
        ("white_yellow_mug_1",),
        "microwave_1_heating_region",
        "pick_place",
    ),
    _stage(
        2,
        "microwave_closed",
        ("close", "microwave_1"),
        "Microwave is closed after the mug is inside.",
        ("microwave_1",),
        "",
        "articulation_close",
    ),
)

L2_STAGES = L1_STAGES + (
    _stage(
        3,
        "bowl_in_bottom_drawer",
        ("in", "akita_black_bowl_1", "white_cabinet_1_bottom_region"),
        "Black bowl is inside the cabinet bottom drawer.",
        ("akita_black_bowl_1",),
        "white_cabinet_1_bottom_region",
        "pick_place",
    ),
    _stage(
        4,
        "bottom_drawer_closed",
        ("close", "white_cabinet_1_bottom_region"),
        "Bottom drawer is closed after the bowl is inside.",
        ("white_cabinet_1",),
        "",
        "articulation_close",
    ),
)

L3_STAGES = L2_STAGES + (
    _stage(
        5,
        "first_moka_pot_on_stove",
        ("custom_any_on", "moka_pot_1", "moka_pot_2", "flat_stove_1_cook_region"),
        "At least one moka pot is on the already-on stove cook region.",
        ("moka_pot_1", "moka_pot_2"),
        "flat_stove_1_cook_region",
        "pick_place",
    ),
    _stage(
        6,
        "both_moka_pots_on_stove",
        ("custom_both_on", "moka_pot_1", "moka_pot_2", "flat_stove_1_cook_region"),
        "Both moka pots are on the already-on stove cook region.",
        ("moka_pot_1", "moka_pot_2"),
        "flat_stove_1_cook_region",
        "pick_place",
    ),
)

L4_STAGES = L2_STAGES + (
    _stage(
        5,
        "stove_turned_on",
        ("turnon", "flat_stove_1"),
        "Stove is turned on after the storage subtasks are finished.",
        ("flat_stove_1",),
        "",
        "articulation_toggle",
    ),
    _stage(
        6,
        "first_moka_pot_on_stove",
        ("custom_any_on", "moka_pot_1", "moka_pot_2", "flat_stove_1_cook_region"),
        "At least one moka pot is on the stove cook region after turn-on.",
        ("moka_pot_1", "moka_pot_2"),
        "flat_stove_1_cook_region",
        "pick_place",
    ),
    _stage(
        7,
        "both_moka_pots_on_stove",
        ("custom_both_on", "moka_pot_1", "moka_pot_2", "flat_stove_1_cook_region"),
        "Both moka pots are on the stove cook region.",
        ("moka_pot_1", "moka_pot_2"),
        "flat_stove_1_cook_region",
        "pick_place",
    ),
    _stage(
        8,
        "stove_turned_off_after_pots",
        ("turnoff", "flat_stove_1"),
        "Stove is turned off after both moka pots are placed.",
        ("flat_stove_1",),
        "",
        "articulation_toggle",
    ),
)


TASK_SPECS: Tuple[TaskSpec, ...] = (
    TaskSpec(
        task_id=0,
        name="KITCHEN_BREAKFAST_L1_PUT_MUG_IN_MICROWAVE_AND_CLOSE_IT",
        language=(
            "Put the yellow and white mug into the microwave and close the microwave."
        ),
        bddl_file="KITCHEN_BREAKFAST_L1_PUT_MUG_IN_MICROWAVE_AND_CLOSE_IT.bddl",
        stages=L1_STAGES,
        final_goals=(
            "In white_yellow_mug_1 microwave_1_heating_region",
            "Close microwave_1",
        ),
        max_steps=260,
    ),
    TaskSpec(
        task_id=1,
        name="KITCHEN_BREAKFAST_L2_MUG_MICROWAVE_BOWL_DRAWER",
        language=(
            "Put the yellow and white mug into the microwave and close the microwave, "
            "then put the black bowl into the bottom drawer and close the drawer."
        ),
        bddl_file="KITCHEN_BREAKFAST_L2_MUG_MICROWAVE_BOWL_DRAWER.bddl",
        stages=L2_STAGES,
        final_goals=(
            "In white_yellow_mug_1 microwave_1_heating_region",
            "Close microwave_1",
            "In akita_black_bowl_1 white_cabinet_1_bottom_region",
            "Close white_cabinet_1_bottom_region",
        ),
        max_steps=440,
    ),
    TaskSpec(
        task_id=2,
        name="KITCHEN_BREAKFAST_L3_STORAGE_AND_MOKA_ON_ALREADY_ON_STOVE",
        language=(
            "Put the yellow and white mug into the microwave and close the microwave, "
            "put the black bowl into the bottom drawer and close the drawer, then put "
            "both moka pots on the already-on stove."
        ),
        bddl_file="KITCHEN_BREAKFAST_L3_STORAGE_AND_MOKA_ON_ALREADY_ON_STOVE.bddl",
        stages=L3_STAGES,
        final_goals=(
            "In white_yellow_mug_1 microwave_1_heating_region",
            "Close microwave_1",
            "In akita_black_bowl_1 white_cabinet_1_bottom_region",
            "Close white_cabinet_1_bottom_region",
            "On moka_pot_1 flat_stove_1_cook_region",
            "On moka_pot_2 flat_stove_1_cook_region",
        ),
        max_steps=660,
        starts_with_stove_on=True,
    ),
    TaskSpec(
        task_id=3,
        name="KITCHEN_BREAKFAST_L4_FULL_STORAGE_STOVE_MOKA_TURNOFF",
        language=(
            "Put the yellow and white mug into the microwave and close the microwave, "
            "put the black bowl into the bottom drawer and close the drawer, turn on "
            "the stove, put both moka pots on the stove, then turn the stove off."
        ),
        bddl_file="KITCHEN_BREAKFAST_L4_FULL_STORAGE_STOVE_MOKA_TURNOFF.bddl",
        stages=L4_STAGES,
        final_goals=(
            "In white_yellow_mug_1 microwave_1_heating_region",
            "Close microwave_1",
            "In akita_black_bowl_1 white_cabinet_1_bottom_region",
            "Close white_cabinet_1_bottom_region",
            "On moka_pot_1 flat_stove_1_cook_region",
            "On moka_pot_2 flat_stove_1_cook_region",
            "Turnoff flat_stove_1",
        ),
        max_steps=820,
    ),
)


class LongBreakfastCurriculumV2TaskSuite:
    name = SUITE_NAME
    n_tasks = len(TASK_SPECS)

    def __init__(self, num_init_states: int = DEFAULT_NUM_INIT_STATES):
        self.num_init_states = int(num_init_states)
        self.tasks = [
            CustomTask(
                name=spec.name,
                language=spec.language,
                problem="Libero",
                problem_folder="custom_tasks/long_breakfast_curriculum_v2",
                bddl_file=spec.bddl_file,
                init_states_file="generated_from_bddl_reset",
            )
            for spec in TASK_SPECS
        ]

    def get_task(self, i: int) -> CustomTask:
        return self.tasks[int(i)]

    def get_task_init_states(self, i: int) -> List[None]:
        self.get_task(i)
        return [None for _ in range(self.num_init_states)]


class StageOracleTracker:
    """Monotonic stage tracker for one curriculum level."""

    def __init__(
        self,
        task_id: int = 0,
        initial_goal_truth: Optional[Dict[str, bool]] = None,
    ):
        self.task_id = int(task_id)
        self.spec = task_spec(self.task_id)
        initial_goal_truth = initial_goal_truth or {}
        self.completed = {
            stage.label: bool(initial_goal_truth.get(stage.label, False))
            for stage in self.spec.stages
        }
        self.order_valid = bool(initial_goal_truth.get(ORDER_VALID_LABEL, True))
        self.order_violation_reasons: List[str] = []

    @classmethod
    def from_snapshot(
        cls,
        task_id: int = 0,
        snapshot: Optional[StateSnapshot] = None,
    ) -> "StageOracleTracker":
        return cls(task_id, None if snapshot is None else snapshot.goal_truth)

    @property
    def completed_count(self) -> int:
        return sum(
            1 for stage in self.spec.stages if self.completed.get(stage.label, False)
        )

    def update(self, env) -> Dict[str, bool]:
        instant = instantaneous_stage_truth(env, self.task_id)
        self._detect_order_violations(instant)
        for stage in self.spec.stages:
            if self.completed.get(stage.label, False):
                continue
            if all(
                self.completed.get(prev.label, False)
                for prev in self.spec.stages[: stage.index - 1]
            ) and bool(instant.get(stage.label, False)):
                self.completed[stage.label] = True
            break
        truth = {
            stage.label: bool(self.completed.get(stage.label, False))
            for stage in self.spec.stages
        }
        truth[ORDER_VALID_LABEL] = bool(self.order_valid)
        return truth

    def _detect_order_violations(self, instant: Dict[str, bool]) -> None:
        labels = {stage.key: stage.label for stage in self.spec.stages}

        def done(key: str) -> bool:
            label = labels.get(key)
            return bool(label and self.completed.get(label, False))

        def active(key: str) -> bool:
            label = labels.get(key)
            return bool(label and instant.get(label, False))

        checks = [
            (
                "microwave_closed_before_mug_inserted",
                active("microwave_closed") and not done("mug_in_microwave"),
            ),
            (
                "drawer_closed_before_bowl_inserted",
                active("bottom_drawer_closed") and not done("bowl_in_bottom_drawer"),
            ),
            (
                "moka_pot_on_stove_before_storage_done",
                active("first_moka_pot_on_stove")
                and ("bowl_in_bottom_drawer" in labels)
                and not done("bottom_drawer_closed"),
            ),
            (
                "moka_pot_on_stove_before_stove_turned_on",
                active("first_moka_pot_on_stove")
                and ("stove_turned_on" in labels)
                and not done("stove_turned_on"),
            ),
            (
                "stove_turned_off_before_both_moka_pots_placed",
                active("stove_turned_off_after_pots")
                and done("stove_turned_on")
                and not done("both_moka_pots_on_stove"),
            ),
        ]
        for reason, violation in checks:
            if violation and reason not in self.order_violation_reasons:
                self.order_violation_reasons.append(reason)
                self.order_valid = False


def is_custom_suite(task_suite_name: str) -> bool:
    return str(task_suite_name).lower() == SUITE_NAME


def task_spec(task_id: int) -> TaskSpec:
    task_id = int(task_id)
    if task_id < 0 or task_id >= len(TASK_SPECS):
        raise IndexError(f"{SUITE_NAME} has {len(TASK_SPECS)} tasks, got task_id={task_id}")
    return TASK_SPECS[task_id]


def make_task_suite(task_suite_name: str) -> Optional[LongBreakfastCurriculumV2TaskSuite]:
    if not is_custom_suite(task_suite_name):
        return None
    return LongBreakfastCurriculumV2TaskSuite()


def custom_bddl_path(task_suite_name: str, task_id: int) -> Optional[Path]:
    if not is_custom_suite(task_suite_name):
        return None
    return task_spec(task_id).bddl_path


def deterministic_reset_seed(base_seed: int, init_state_id: int) -> int:
    return int(base_seed) + 1009 * int(init_state_id)


def max_steps_for_suite(task_suite_name: str) -> Optional[int]:
    return DEFAULT_MAX_STEPS if is_custom_suite(task_suite_name) else None


def max_steps_for_task(task_suite_name: str, task_id: int) -> Optional[int]:
    return task_spec(task_id).max_steps if is_custom_suite(task_suite_name) else None


def stage_predicates_for_suite(
    task_suite_name: str, task_id: int = 0
) -> Tuple[GoalPredicate, ...]:
    if not is_custom_suite(task_suite_name):
        return tuple()
    return tuple(GoalPredicate((stage.label,)) for stage in task_spec(task_id).stages) + (
        GoalPredicate((ORDER_VALID_LABEL,)),
    )


def make_stage_tracker(
    task_suite_name: str,
    task_id: int = 0,
    snapshot: Optional[StateSnapshot] = None,
):
    if not is_custom_suite(task_suite_name):
        return None
    return StageOracleTracker.from_snapshot(task_id, snapshot)


def augment_snapshot_with_stage_truth(
    task_suite_name: str,
    snapshot: StateSnapshot,
    tracker: Optional[StageOracleTracker],
    env,
    task_id: int = 0,
) -> StateSnapshot:
    if tracker is None or not is_custom_suite(task_suite_name):
        return snapshot
    merged = dict(snapshot.goal_truth)
    merged.update(tracker.update(env))
    return replace(snapshot, goal_truth=merged)


def stage_trace_from_snapshots(
    snapshots: Sequence[StateSnapshot],
    task_id: int = 0,
) -> dict:
    spec = task_spec(task_id)
    labels = [stage.label for stage in spec.stages] + [ORDER_VALID_LABEL]
    truth_by_t = [
        {label: bool(snapshot.goal_truth.get(label, False)) for label in labels}
        for snapshot in snapshots
    ]
    progress_counts = [
        sum(1 for stage in spec.stages if row.get(stage.label, False))
        for row in truth_by_t
    ]
    final_truth = truth_by_t[-1] if truth_by_t else {}
    first_completed = {
        label: next(
            (i for i, row in enumerate(truth_by_t) if bool(row.get(label, False))),
            None,
        )
        for label in labels
    }
    earliest_failed = _earliest_failed_stage_from_truth(spec, final_truth)
    return {
        "schema_version": "long-breakfast-curriculum-v2-stage-oracle-v1",
        "task_suite_name": SUITE_NAME,
        "task_id": int(task_id),
        "task_name": spec.name,
        "stage_order": [
            {
                "index": stage.index,
                "key": stage.key,
                "label": stage.label,
                "predicate": list(stage.predicate),
                "description": stage.description,
                "target_objects": list(stage.target_objects),
                "target_region": stage.target_region,
                "repair_kind": stage.repair_kind,
            }
            for stage in spec.stages
        ],
        "final_truth": final_truth,
        "progress_counts": progress_counts,
        "stage_progress_count": int(progress_counts[-1]) if progress_counts else 0,
        "earliest_failed_stage": None if earliest_failed is None else _stage_public(earliest_failed),
        "first_completed_step": first_completed,
        "stage_first_completed_step": first_completed,
        "num_steps_recorded": len(truth_by_t),
    }


def stage_summary_from_snapshots(
    task_suite_name: str,
    snapshots: Sequence[StateSnapshot],
    task_id: int = 0,
) -> Optional[dict]:
    if not is_custom_suite(task_suite_name):
        return None
    trace = stage_trace_from_snapshots(snapshots, task_id)
    return {
        "earliest_failed_stage": trace.get("earliest_failed_stage"),
        "stage_progress_count": trace.get("stage_progress_count"),
        "stage_first_completed_step": trace.get("stage_first_completed_step"),
    }


def target_key_for_snapshot(
    task_suite_name: str,
    task_id: int,
    snapshot: Optional[StateSnapshot],
    obs: dict,
    task_language: str = "",
) -> Optional[str]:
    if not is_custom_suite(task_suite_name):
        return None
    stage = earliest_unfinished_stage(task_id, snapshot.goal_truth if snapshot else {})
    if stage is None:
        return None
    candidates = list(stage.target_objects)
    if stage.key == "both_moka_pots_on_stove" and snapshot is not None:
        positions = snapshot.object_positions
        stove_hint = _site_or_fixture_pos(obs, "flat_stove_1")
        if stove_hint is not None:
            ranked = []
            for obj in candidates:
                if obj in positions:
                    dist = float(
                        np.linalg.norm(
                            np.asarray(positions[obj], dtype=np.float64)
                            - np.asarray(stove_hint, dtype=np.float64)
                        )
                    )
                    ranked.append((dist, obj))
            if ranked:
                candidates = [obj for _dist, obj in sorted(ranked, reverse=True)]
    return _first_existing_pos_key(obs, candidates)


def task_metadata(task_suite_name: str = SUITE_NAME, task_id: Optional[int] = None) -> dict:
    specs = TASK_SPECS if task_id is None else (task_spec(task_id),)
    return {
        "schema_version": "libero-custom-long-breakfast-curriculum-v2.1",
        "task_suite_name": SUITE_NAME,
        "layout_revision": LAYOUT_REVISION,
        "default_max_steps": int(DEFAULT_MAX_STEPS),
        "num_tasks": len(TASK_SPECS),
        "initial_state_quality": {
            "stage01_primary_target": "white_yellow_mug_1",
            "target_quat_reference_xyzw": [
                float(x) for x in WHITE_YELLOW_MUG_STABLE_QUAT_XYZW
            ],
            "target_quat_alignment_threshold": 0.93,
            "target_clearance_xy_threshold": 0.10,
        },
        "tasks": [
            {
                "task_id": spec.task_id,
                "task_name": spec.name,
                "task_language": spec.language,
                "bddl_path": str(spec.bddl_path),
                "max_steps": int(spec.max_steps),
                "starts_with_stove_on": bool(spec.starts_with_stove_on),
                "stage_oracle": stage_trace_from_snapshots([], spec.task_id)["stage_order"],
                "final_goals": list(spec.final_goals),
            }
            for spec in specs
        ],
    }


def instantaneous_stage_truth(env, task_id: int = 0) -> Dict[str, bool]:
    return {
        stage.label: _eval_stage_predicate(env, stage.predicate)
        for stage in task_spec(task_id).stages
    }


def earliest_unfinished_stage(
    task_id: int, goal_truth: Optional[Dict[str, bool]] = None
) -> Optional[StageDefinition]:
    truth = goal_truth or {}
    for stage in task_spec(task_id).stages:
        if not bool(truth.get(stage.label, False)):
            return stage
    return None


def initial_state_quality(
    task_suite_name: str,
    task_id: int,
    obs: dict,
    env,
    snapshot: Optional[StateSnapshot] = None,
) -> dict:
    if not is_custom_suite(task_suite_name):
        return {"valid": True, "reasons": [], "semantic_quality": "not_curriculum_v2"}
    spec = task_spec(task_id)
    stage = earliest_unfinished_stage(task_id, snapshot.goal_truth if snapshot else {})
    target_objects = list(stage.target_objects if stage else spec.stages[0].target_objects)
    primary = target_objects[0] if target_objects else ""
    reasons: List[str] = []
    warnings: List[str] = []
    metrics: Dict[str, object] = {
        "task_id": int(task_id),
        "task_name": spec.name,
        "layout_revision": LAYOUT_REVISION,
        "checked_stage": None if stage is None else _stage_public(stage),
        "primary_target": primary,
    }

    if not primary:
        return {
            "valid": True,
            "reasons": [],
            "warnings": warnings,
            "metrics": metrics,
            "semantic_quality": "no_primary_target",
        }
    pos = _obs_pos(obs, primary)
    if pos is None:
        reasons.append("missing_primary_target_position")
    else:
        metrics["primary_target_pos"] = [float(x) for x in pos[:3]]
        metrics["primary_target_z"] = float(pos[2])
        if primary == "white_yellow_mug_1":
            if float(pos[2]) < 0.88:
                reasons.append("primary_target_below_table_or_unstable")
            if float(pos[2]) > 0.945:
                warnings.append("primary_target_high_after_settle")

    quat_key = f"{primary}_quat"
    quat = obs.get(quat_key) if isinstance(obs, dict) else None
    if primary == "white_yellow_mug_1":
        if quat is None:
            warnings.append("missing_primary_target_quat")
        else:
            arr = np.asarray(quat, dtype=np.float64).reshape(-1)[:4]
            norm = float(np.linalg.norm(arr))
            if norm <= 1e-8:
                reasons.append("primary_target_bad_quat")
            else:
                arr = arr / norm
                ref = WHITE_YELLOW_MUG_STABLE_QUAT_XYZW
                alignment = float(abs(np.dot(arr, ref)))
                metrics["primary_target_quat_xyzw"] = [float(x) for x in arr]
                metrics["primary_target_quat_alignment"] = alignment
                if alignment < 0.93:
                    reasons.append("primary_target_not_upright")

    if pos is not None:
        nearest = None
        ignored = {
            primary,
            "robot0_eef",
            "robot0_gripper",
            "microwave_1",
            "white_cabinet_1",
            "flat_stove_1",
        }
        for key, value in (obs or {}).items():
            if not key.endswith("_pos") or "_to_" in key or key.startswith("robot"):
                continue
            obj = key[:-4]
            if obj in ignored:
                continue
            other = np.asarray(value, dtype=np.float64).reshape(-1)
            if other.shape[0] < 3:
                continue
            dist_xy = float(np.linalg.norm((other[:2] - pos[:2])))
            if nearest is None or dist_xy < nearest[1]:
                nearest = (obj, dist_xy)
        if nearest is not None:
            metrics["nearest_object_to_primary_xy"] = {
                "object": nearest[0],
                "distance": float(nearest[1]),
            }
            if nearest[1] < 0.10:
                reasons.append("primary_target_too_close_to_%s" % nearest[0])
            elif nearest[1] < 0.13:
                warnings.append("primary_target_low_clearance_to_%s" % nearest[0])

    return {
        "valid": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "metrics": metrics,
        "semantic_quality": "full",
    }


def expert_repair_metadata(task_suite_name: str, task_id: int, snapshot: StateSnapshot) -> dict:
    if not is_custom_suite(task_suite_name):
        return {"available": False, "reason": "not_curriculum_v2"}
    stage = earliest_unfinished_stage(task_id, snapshot.goal_truth)
    if stage is None:
        return {"available": False, "reason": "all_stages_complete"}
    if stage.repair_kind != "pick_place":
        return {
            "available": False,
            "reason": "stage_repair_kind_not_implemented",
            "stage": _stage_public(stage),
            "expert_repair_only": True,
        }
    return {
        "available": True,
        "source": "scripted_stage_pick_place_expert",
        "expert_repair_only": True,
        "stage": _stage_public(stage),
        "object_candidates": list(stage.target_objects),
        "target_region": stage.target_region,
    }


def expert_action(
    controller_state: dict,
    env,
    obs: dict,
    task_id: int,
    snapshot: StateSnapshot,
) -> Optional[np.ndarray]:
    """Online scripted pick-place action for repair validation only."""
    stage = earliest_unfinished_stage(task_id, snapshot.goal_truth)
    if stage is None or stage.repair_kind != "pick_place":
        return None
    target_obj = _choose_pick_object(stage, obs)
    if target_obj is None:
        return None
    target_region_pos = _region_pos(env, stage.target_region)
    if target_region_pos is None:
        return None
    object_pos = _obs_pos(obs, target_obj)
    eef_pos = _obs_pos(obs, "robot0_eef")
    if object_pos is None or eef_pos is None:
        return None

    phase = str(controller_state.get("phase") or "approach")
    ticks = int(controller_state.get("ticks") or 0)
    controller_state["target_object"] = target_obj
    controller_state["target_region"] = stage.target_region
    controller_state["stage_key"] = stage.key

    above_obj = np.asarray(object_pos, dtype=np.float64) + np.array([0.0, 0.0, 0.12])
    grasp_pos = np.asarray(object_pos, dtype=np.float64) + np.array([0.0, 0.0, 0.025])
    above_region = np.asarray(target_region_pos, dtype=np.float64) + np.array([0.0, 0.0, 0.14])
    place_pos = np.asarray(target_region_pos, dtype=np.float64) + np.array([0.0, 0.0, 0.055])

    if phase == "approach":
        action = _move_action(eef_pos, above_obj, gripper=-1.0)
        if _near(eef_pos, above_obj, 0.025) or ticks > 45:
            controller_state.update(phase="descend", ticks=0)
        else:
            controller_state["ticks"] = ticks + 1
        return action
    if phase == "descend":
        action = _move_action(eef_pos, grasp_pos, gripper=-1.0)
        if _near(eef_pos, grasp_pos, 0.018) or ticks > 35:
            controller_state.update(phase="close", ticks=0)
        else:
            controller_state["ticks"] = ticks + 1
        return action
    if phase == "close":
        controller_state["ticks"] = ticks + 1
        if ticks > 12:
            controller_state.update(phase="lift", ticks=0)
        return _move_action(eef_pos, grasp_pos, gripper=1.0)
    if phase == "lift":
        action = _move_action(eef_pos, above_obj, gripper=1.0)
        if _near(eef_pos, above_obj, 0.03) or ticks > 45:
            controller_state.update(phase="transfer", ticks=0)
        else:
            controller_state["ticks"] = ticks + 1
        return action
    if phase == "transfer":
        action = _move_action(eef_pos, above_region, gripper=1.0)
        if _near(eef_pos, above_region, 0.035) or ticks > 75:
            controller_state.update(phase="lower", ticks=0)
        else:
            controller_state["ticks"] = ticks + 1
        return action
    if phase == "lower":
        action = _move_action(eef_pos, place_pos, gripper=1.0)
        if _near(eef_pos, place_pos, 0.025) or ticks > 40:
            controller_state.update(phase="release", ticks=0)
        else:
            controller_state["ticks"] = ticks + 1
        return action
    if phase == "release":
        controller_state["ticks"] = ticks + 1
        if ticks > 12:
            controller_state.update(phase="retreat", ticks=0)
        return _move_action(eef_pos, place_pos, gripper=-1.0)
    if phase == "retreat":
        action = _move_action(eef_pos, above_region, gripper=-1.0)
        if _near(eef_pos, above_region, 0.04) or ticks > 35:
            controller_state.update(phase="done", ticks=0)
        else:
            controller_state["ticks"] = ticks + 1
        return action
    return None


def _stage_public(stage: StageDefinition) -> dict:
    return {
        "index": int(stage.index),
        "key": stage.key,
        "label": stage.label,
        "description": stage.description,
        "predicate": list(stage.predicate),
        "target_objects": list(stage.target_objects),
        "target_region": stage.target_region,
        "repair_kind": stage.repair_kind,
    }


def _earliest_failed_stage_from_truth(
    spec: TaskSpec, final_truth: Dict[str, bool]
) -> Optional[StageDefinition]:
    for stage in spec.stages:
        if not bool(final_truth.get(stage.label, False)):
            return stage
    if not bool(final_truth.get(ORDER_VALID_LABEL, True)):
        return StageDefinition(
            index=len(spec.stages) + 1,
            key="order_violation",
            label=ORDER_VALID_LABEL,
            predicate=(ORDER_VALID_LABEL,),
            description="Stage order was violated.",
        )
    return None


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


def _obs_pos(obs: dict, name: str) -> Optional[np.ndarray]:
    key = name if name.endswith("_pos") else f"{name}_pos"
    value = obs.get(key)
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] < 3:
        return None
    return arr[:3]


def _first_existing_pos_key(obs: dict, object_names: Sequence[str]) -> Optional[str]:
    for name in object_names:
        key = f"{name}_pos"
        if key in obs:
            return key
    return None


def _site_or_fixture_pos(obs: dict, name: str) -> Optional[np.ndarray]:
    pos = _obs_pos(obs, name)
    if pos is not None:
        return pos
    return None


def _region_pos(env, region_name: str) -> Optional[np.ndarray]:
    if not region_name:
        return None
    base = _base_env(env)
    state = getattr(base, "object_states_dict", {}).get(region_name)
    if state is None:
        return None
    try:
        return np.asarray(state.get_geom_state()["pos"], dtype=np.float64)[:3]
    except Exception:
        return None


def _choose_pick_object(stage: StageDefinition, obs: dict) -> Optional[str]:
    candidates = [obj for obj in stage.target_objects if f"{obj}_pos" in obs]
    if not candidates:
        return None
    eef = _obs_pos(obs, "robot0_eef")
    if eef is None or len(candidates) == 1:
        return candidates[0]
    return min(
        candidates,
        key=lambda obj: float(np.linalg.norm(eef - np.asarray(obs[f"{obj}_pos"], dtype=np.float64))),
    )


def _near(current: np.ndarray, target: np.ndarray, threshold: float) -> bool:
    return float(np.linalg.norm(np.asarray(current) - np.asarray(target))) <= float(threshold)


def _move_action(current: np.ndarray, target: np.ndarray, gripper: float) -> np.ndarray:
    delta = np.asarray(target, dtype=np.float64) - np.asarray(current, dtype=np.float64)
    xyz = np.clip(delta * 8.0, -1.0, 1.0)
    return np.asarray([xyz[0], xyz[1], xyz[2], 0.0, 0.0, 0.0, gripper], dtype=np.float32)
