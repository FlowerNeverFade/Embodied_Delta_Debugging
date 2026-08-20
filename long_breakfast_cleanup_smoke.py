from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from causal_failure_predicates import get_goal_predicates, semantic_quality_for_env
from custom_tasks import long_horizon_breakfast_cleanup as long_task
from pi05_natural_failure_probe import (
    LIBERO_DUMMY_ACTION,
    _make_env,
    _semantic_snapshot,
)


DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/research/Embodied_Delta_Debugging/outputs/long_breakfast_cleanup_smoke.json"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the custom LIBERO long-horizon breakfast cleanup task."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument("--dummy-steps", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    env, suite, task = _make_env(args, long_task.SUITE_NAME, 0)
    try:
        env.seed(long_task.deterministic_reset_seed(args.seed, args.init_state_id))
        obs = env.reset()
        predicates = get_goal_predicates(env) + long_task.stage_predicates_for_suite(
            long_task.SUITE_NAME
        )
        tracker = long_task.make_stage_tracker(long_task.SUITE_NAME)
        snapshots = [
            _semantic_snapshot(
                long_task.SUITE_NAME,
                0,
                obs,
                env=env,
                action=None,
                success=False,
                predicates=predicates,
                stage_tracker=tracker,
            )
        ]
        done = False
        reward = 0.0
        info = {}
        for step in range(max(0, int(args.dummy_steps))):
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            snapshots.append(
                _semantic_snapshot(
                    long_task.SUITE_NAME,
                    step + 1,
                    obs,
                    env=env,
                    action=LIBERO_DUMMY_ACTION,
                    success=bool(done),
                    predicates=predicates,
                    stage_tracker=tracker,
                )
            )
            if done:
                break

        payload = {
            "schema_version": "long-breakfast-cleanup-smoke-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "custom_task": long_task.task_metadata(),
            "task_language": task.language,
            "init_state_id": int(args.init_state_id),
            "seed": int(args.seed),
            "dummy_steps_requested": int(args.dummy_steps),
            "dummy_steps_executed": int(len(snapshots) - 1),
            "done": bool(done),
            "reward": float(reward),
            "info": {str(k): str(v) for k, v in (info or {}).items()},
            "semantic_quality": semantic_quality_for_env(env),
            "num_predicates": int(len(predicates)),
            "bddl_goal_predicates": [p.to_dict() for p in get_goal_predicates(env)],
            "stage_oracle_trace": long_task.stage_trace_from_snapshots(snapshots),
            "initial_goal_truth": snapshots[0].goal_truth,
            "final_goal_truth": snapshots[-1].goal_truth,
            "obs_keys_sample": list(obs.keys())[:32],
        }
    finally:
        env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
