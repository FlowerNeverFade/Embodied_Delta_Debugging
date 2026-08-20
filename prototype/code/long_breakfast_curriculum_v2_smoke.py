from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from causal_failure_predicates import get_goal_predicates, semantic_quality_for_env
from custom_tasks import long_breakfast_curriculum_v2 as curriculum
from pi05_natural_failure_probe import LIBERO_DUMMY_ACTION, _make_env, _semantic_snapshot


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "long_breakfast_curriculum_v2_smoke.json"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test curriculum-v2 LIBERO tasks.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task-ids", default="0,1,2,3")
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument("--dummy-steps", type=int, default=5)
    return parser.parse_args(argv)


def _parse_ids(text: str) -> list[int]:
    return [int(x) for x in str(text).split(",") if x.strip()]


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    cases = []
    for task_id in _parse_ids(args.task_ids):
        env, suite, task = _make_env(args, curriculum.SUITE_NAME, task_id)
        try:
            env.seed(curriculum.deterministic_reset_seed(args.seed, args.init_state_id))
            obs = env.reset()
            predicates = get_goal_predicates(env) + curriculum.stage_predicates_for_suite(
                curriculum.SUITE_NAME, task_id
            )
            tracker = curriculum.make_stage_tracker(curriculum.SUITE_NAME, task_id)
            snapshots = [
                _semantic_snapshot(
                    curriculum.SUITE_NAME,
                    0,
                    obs,
                    env=env,
                    action=None,
                    success=False,
                    predicates=predicates,
                    stage_tracker=tracker,
                    task_id=task_id,
                )
            ]
            done = False
            for step in range(max(0, int(args.dummy_steps))):
                obs, _reward, done, _info = env.step(LIBERO_DUMMY_ACTION)
                snapshots.append(
                    _semantic_snapshot(
                        curriculum.SUITE_NAME,
                        step + 1,
                        obs,
                        env=env,
                        action=LIBERO_DUMMY_ACTION,
                        success=bool(done),
                        predicates=predicates,
                        stage_tracker=tracker,
                        task_id=task_id,
                    )
                )
                if done:
                    break
            trace = curriculum.stage_trace_from_snapshots(snapshots, task_id)
            cases.append(
                {
                    "task_id": int(task_id),
                    "task_language": task.language,
                    "bddl_path": str(curriculum.custom_bddl_path(curriculum.SUITE_NAME, task_id)),
                    "semantic_quality": semantic_quality_for_env(env),
                    "num_predicates": int(len(predicates)),
                    "initial_goal_truth": snapshots[0].goal_truth,
                    "final_goal_truth": snapshots[-1].goal_truth,
                    "stage_oracle_trace": trace,
                    "initial_core_goal_already_satisfied": bool(
                        trace.get("stage_progress_count", 0) >= len(curriculum.task_spec(task_id).stages)
                    ),
                    "obs_keys_sample": list(obs.keys())[:32],
                }
            )
        finally:
            env.close()
    payload = {
        "schema_version": "long-breakfast-curriculum-v2-smoke-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "custom_task": curriculum.task_metadata(),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
