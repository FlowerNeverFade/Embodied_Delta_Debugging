from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "libero_smoke.json"


def run_smoke(args: argparse.Namespace) -> dict:
    # Keep the smoke test self-contained and friendly to headless servers.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(args.gpu_id))

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    init_state_id = min(args.init_state_id, len(initial_states) - 1)
    bddl_file = (
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )

    env = None
    observations = []
    rewards = []
    dones = []
    try:
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=args.camera_size,
            camera_widths=args.camera_size,
        )
        env.seed(args.seed)
        env.reset()
        obs = env.set_init_state(initial_states[init_state_id])
        dummy_action = [0.0] * 6 + [-1.0]
        for step in range(args.steps):
            if step == args.corrupt_step:
                action = [0.25, -0.25, 0.15, 0.0, 0.0, 0.0, 1.0]
            else:
                action = dummy_action
            obs, reward, done, info = env.step(action)
            observations.append(
                {
                    "step": int(step),
                    "eef_pos": np.asarray(obs.get("robot0_eef_pos", [])).tolist(),
                    "gripper_qpos": np.asarray(obs.get("robot0_gripper_qpos", [])).tolist(),
                }
            )
            rewards.append(float(reward))
            dones.append(bool(done))
            if done:
                break
        status = "passed"
        error = None
    except Exception as exc:  # pragma: no cover - depends on local MuJoCo stack
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            env.close()

    return {
        "status": status,
        "error": error,
        "task_suite_name": args.task_suite_name,
        "task_id": int(args.task_id),
        "task_language": getattr(task, "language", None) if "task" in locals() else None,
        "bddl_file": str(bddl_file) if "bddl_file" in locals() else None,
        "init_state_id": int(args.init_state_id),
        "steps_requested": int(args.steps),
        "steps_completed": len(observations),
        "reward_sum": float(sum(rewards)),
        "done_any": bool(any(dones)),
        "sample_observations": observations[:3] + observations[-3:] if len(observations) > 6 else observations,
        "env": {
            "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
            "MUJOCO_EGL_DEVICE_ID": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
        },
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LIBERO reset/set_init_state/step smoke test.")
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--corrupt-step", type=int, default=15)
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    result = run_smoke(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
