from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from causal_failure_predicates import eval_goal_truth, get_goal_predicates
from custom_tasks import long_horizon_breakfast_cleanup as long_task
from pi05_natural_failure_probe import _make_env


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "long_breakfast_cleanup_oracle_feasibility.json"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that the custom long breakfast cleanup BDDL goals are jointly satisfiable."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--camera-size", type=int, default=128)
    return parser.parse_args(argv)


def _set_free_joint_xyz(sim, joint_name: str, xyz) -> None:
    qpos = np.asarray(sim.data.get_joint_qpos(joint_name)).copy()
    qpos[:3] = np.asarray(xyz, dtype=np.float64)[:3]
    sim.data.set_joint_qpos(joint_name, qpos)


def _refresh(base) -> None:
    base.sim.forward()
    base._post_process()
    base._update_observables(force=True)
    base._check_success()


def _record_stage(base, tracker, label: str) -> dict:
    truth = tracker.update(base)
    return {
        "label": label,
        "stage_truth": truth,
        "completed_stage_count": sum(
            1 for key, value in truth.items() if key.startswith("Stage") and key != "Stageorder valid_sequence_so_far" and value
        ),
        "order_valid": bool(truth.get("Stageorder valid_sequence_so_far", False)),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    env, _suite, task = _make_env(args, long_task.SUITE_NAME, 0)
    try:
        env.seed(long_task.deterministic_reset_seed(args.seed, 0))
        env.reset()
        base = env.env
        sim = base.sim

        # Set articulation states first because region sites attached to the
        # drawer move with its joint.
        tracker = long_task.StageOracleTracker()
        stage_reference = [_record_stage(base, tracker, "initial_open_microwave")]

        heating_pos = base.object_states_dict["microwave_1_heating_region"].get_geom_state()["pos"]
        _set_free_joint_xyz(sim, "white_yellow_mug_1_joint0", heating_pos)
        _refresh(base)
        stage_reference.append(_record_stage(base, tracker, "place_mug_in_microwave"))

        sim.data.set_joint_qpos("microwave_1_microjoint", 0.0)
        _refresh(base)
        stage_reference.append(_record_stage(base, tracker, "close_microwave"))

        # The bottom drawer starts open in the BDDL init state; after the first
        # three stages are complete, this should count as Stage04.
        stage_reference.append(_record_stage(base, tracker, "observe_bottom_drawer_open"))

        bottom_pos = base.object_states_dict["white_cabinet_1_bottom_region"].get_geom_state()["pos"]
        _set_free_joint_xyz(sim, "akita_black_bowl_1_joint0", bottom_pos)
        _refresh(base)
        stage_reference.append(_record_stage(base, tracker, "place_bowl_in_bottom_drawer"))

        sim.data.set_joint_qpos("white_cabinet_1_bottom_level", 0.003)
        _refresh(base)
        stage_reference.append(_record_stage(base, tracker, "close_bottom_drawer"))

        sim.data.set_joint_qpos("flat_stove_1_button", 0.8)
        _refresh(base)
        stage_reference.append(_record_stage(base, tracker, "turn_on_stove"))

        cook_pos = base.object_states_dict["flat_stove_1_cook_region"].get_geom_state()["pos"]
        _set_free_joint_xyz(sim, "moka_pot_1_joint0", np.asarray(cook_pos) + [0.0, -0.025, 0.06])
        _refresh(base)
        stage_reference.append(_record_stage(base, tracker, "place_first_moka_pot"))

        _set_free_joint_xyz(sim, "moka_pot_2_joint0", np.asarray(cook_pos) + [0.0, 0.025, 0.06])
        _refresh(base)
        stage_reference.append(_record_stage(base, tracker, "place_second_moka_pot"))

        sim.data.set_joint_qpos("flat_stove_1_button", -0.003)
        _refresh(base)
        stage_reference.append(_record_stage(base, tracker, "turn_off_stove"))

        heating_pos = base.object_states_dict["microwave_1_heating_region"].get_geom_state()["pos"]
        bottom_pos = base.object_states_dict["white_cabinet_1_bottom_region"].get_geom_state()["pos"]
        cook_pos = base.object_states_dict["flat_stove_1_cook_region"].get_geom_state()["pos"]

        _set_free_joint_xyz(sim, "white_yellow_mug_1_joint0", heating_pos)
        _set_free_joint_xyz(sim, "akita_black_bowl_1_joint0", bottom_pos)
        _set_free_joint_xyz(sim, "moka_pot_1_joint0", np.asarray(cook_pos) + [0.0, -0.025, 0.06])
        _set_free_joint_xyz(sim, "moka_pot_2_joint0", np.asarray(cook_pos) + [0.0, 0.025, 0.06])
        _refresh(base)

        predicates = get_goal_predicates(env)
        truth = eval_goal_truth(env, predicates)
        payload = {
            "schema_version": "long-breakfast-oracle-feasibility-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "custom_task": long_task.task_metadata(),
            "task_language": task.language,
            "method": "simulator_state_oracle_not_robot_demo",
            "all_bddl_goals_satisfied": bool(truth) and all(bool(v) for v in truth.values()),
            "goal_truth": truth,
            "stage_reference": stage_reference,
            "stage_reference_all_complete": bool(
                stage_reference
                and stage_reference[-1]["completed_stage_count"] == len(long_task.STAGES)
                and stage_reference[-1]["order_valid"]
            ),
            "joint_settings": {
                "microwave_1_microjoint": 0.0,
                "white_cabinet_1_bottom_level": 0.003,
                "flat_stove_1_button": -0.003,
            },
            "placement_sites": {
                "microwave_1_heating_region": [float(x) for x in heating_pos],
                "white_cabinet_1_bottom_region": [float(x) for x in bottom_pos],
                "flat_stove_1_cook_region": [float(x) for x in cook_pos],
            },
            "limitations": [
                "This only proves the BDDL final state is jointly satisfiable in the simulator.",
                "It is not a robot-controller demonstration and is not used as Repair SFT data.",
            ],
        }
    finally:
        env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not payload["all_bddl_goals_satisfied"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
