from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from causal_failure_predicates import eval_goal_truth, get_goal_predicates
from custom_tasks import long_breakfast_curriculum_v2 as curriculum
from pi05_natural_failure_probe import _make_env


DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/research/Embodied_Delta_Debugging/outputs/long_breakfast_curriculum_v2_oracle_feasibility.json"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check curriculum-v2 final goals with simulator state oracle."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task-ids", default="0,1,2,3")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--camera-size", type=int, default=128)
    return parser.parse_args(argv)


def _parse_ids(text: str) -> list[int]:
    return [int(x) for x in str(text).split(",") if x.strip()]


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
            1
            for key, value in truth.items()
            if key.startswith("Stage")
            and key != curriculum.ORDER_VALID_LABEL
            and value
        ),
        "order_valid": bool(truth.get(curriculum.ORDER_VALID_LABEL, False)),
    }


def _site(base, key: str) -> np.ndarray:
    return np.asarray(base.object_states_dict[key].get_geom_state()["pos"], dtype=np.float64)


def _satisfy_task(base, task_id: int) -> list[dict]:
    sim = base.sim
    tracker = curriculum.StageOracleTracker(task_id)
    reference = [_record_stage(base, tracker, "initial")]

    heating_pos = _site(base, "microwave_1_heating_region")
    _set_free_joint_xyz(sim, "white_yellow_mug_1_joint0", heating_pos)
    _refresh(base)
    reference.append(_record_stage(base, tracker, "place_mug_in_microwave"))

    sim.data.set_joint_qpos("microwave_1_microjoint", 0.0)
    _refresh(base)
    reference.append(_record_stage(base, tracker, "close_microwave"))

    if task_id >= 1:
        bottom_pos = _site(base, "white_cabinet_1_bottom_region")
        _set_free_joint_xyz(sim, "akita_black_bowl_1_joint0", bottom_pos)
        _refresh(base)
        reference.append(_record_stage(base, tracker, "place_bowl_in_bottom_drawer"))

        sim.data.set_joint_qpos("white_cabinet_1_bottom_level", 0.003)
        _refresh(base)
        bottom_pos = _site(base, "white_cabinet_1_bottom_region")
        _set_free_joint_xyz(sim, "akita_black_bowl_1_joint0", bottom_pos)
        _refresh(base)
        reference.append(_record_stage(base, tracker, "close_bottom_drawer"))

    if task_id == 3:
        sim.data.set_joint_qpos("flat_stove_1_button", 0.8)
        _refresh(base)
        reference.append(_record_stage(base, tracker, "turn_on_stove"))

    if task_id >= 2:
        cook_pos = _site(base, "flat_stove_1_cook_region")
        _set_free_joint_xyz(sim, "moka_pot_1_joint0", cook_pos + [0.0, -0.025, 0.06])
        _refresh(base)
        reference.append(_record_stage(base, tracker, "place_first_moka_pot"))
        _set_free_joint_xyz(sim, "moka_pot_2_joint0", cook_pos + [0.0, 0.025, 0.06])
        _refresh(base)
        reference.append(_record_stage(base, tracker, "place_second_moka_pot"))

    if task_id == 3:
        sim.data.set_joint_qpos("flat_stove_1_button", -0.003)
        _refresh(base)
        reference.append(_record_stage(base, tracker, "turn_off_stove"))

    return reference


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    cases = []
    for task_id in _parse_ids(args.task_ids):
        env, _suite, task = _make_env(args, curriculum.SUITE_NAME, task_id)
        try:
            env.seed(curriculum.deterministic_reset_seed(args.seed, 0))
            env.reset()
            base = env.env
            stage_reference = _satisfy_task(base, task_id)
            predicates = get_goal_predicates(env)
            truth = eval_goal_truth(env, predicates)
            spec = curriculum.task_spec(task_id)
            cases.append(
                {
                    "task_id": int(task_id),
                    "task_language": task.language,
                    "all_bddl_goals_satisfied": bool(truth) and all(bool(v) for v in truth.values()),
                    "goal_truth": truth,
                    "stage_reference": stage_reference,
                    "stage_reference_all_complete": bool(
                        stage_reference
                        and stage_reference[-1]["completed_stage_count"] == len(spec.stages)
                        and stage_reference[-1]["order_valid"]
                    ),
                }
            )
        finally:
            env.close()
    payload = {
        "schema_version": "long-breakfast-curriculum-v2-oracle-feasibility-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "custom_task": curriculum.task_metadata(),
        "method": "simulator_state_oracle_not_robot_demo",
        "cases": cases,
        "all_cases_feasible": all(
            case["all_bddl_goals_satisfied"] and case["stage_reference_all_complete"]
            for case in cases
        ),
        "limitations": [
            "This proves final BDDL states are jointly satisfiable in simulation.",
            "It is not a policy success claim and is not used as Repair SFT action data.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not payload["all_cases_feasible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
