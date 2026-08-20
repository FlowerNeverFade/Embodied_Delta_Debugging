# Embodied Delta Debugging Prototype

This directory contains a first-pass SHED-CFS prototype for testing whether
failure-inducing trajectory minimization is feasible on local LIBERO data.

## What It Validates

- Reads the local LeRobot-format LIBERO dataset:
  `/root/autodl-tmp/research/VLA_SKILL/datasets/HuggingFaceVLA_libero`
- Injects controlled action-chunk failures into successful demonstrations.
- Runs stochastic hierarchical delta debugging over action chunks, then refines
  to a frame-level causal failure slice.
- Emits minimal reproducible embodied bug reports as JSON.
- Runs a LIBERO simulator smoke test for `reset`, `set_init_state`, and `step`.

## Commands

Use the OpenPI venv for offline parquet work:

```bash
cd /root/autodl-tmp/research/Embodied_Delta_Debugging
/root/autodl-tmp/research/openpi/.venv/bin/python data_probe.py --episode 0
/root/autodl-tmp/research/openpi/.venv/bin/python test/test_shed_minimizer.py
/root/autodl-tmp/research/openpi/.venv/bin/python run_offline_probe.py --num-episodes 20
```

Use the LIBERO Python 3.8 env for simulator smoke tests:

```bash
cd /root/autodl-tmp/research/Embodied_Delta_Debugging
/root/autodl-tmp/envs/libero38/bin/python libero_replay_smoke.py --steps 30
```

Run a non-injected real-simulator failure probe:

```bash
cd /root/autodl-tmp/research/Embodied_Delta_Debugging
/root/autodl-tmp/envs/libero38/bin/python real_sim_failure_probe.py --max-steps 40
```

Run a natural `pi05_libero` rollout failure probe. Start the OpenPI policy
server in one shell:

```bash
cd /root/autodl-tmp/research/openpi
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY \
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/autodl-tmp/research/openpi/.venv/bin/python scripts/serve_policy.py \
  --port 8000 policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=/root/autodl-tmp/research/VLA_SKILL/model/pi05_libero
```

Then run the LIBERO client probe in another shell:

```bash
cd /root/autodl-tmp/research/Embodied_Delta_Debugging
/root/autodl-tmp/envs/libero38/bin/python pi05_natural_failure_probe.py
```

If future work needs downloads or dependency installs, run them without proxy
environment variables, for example:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY <command>
```

## Current Outputs

- `outputs/offline_probe/summary.json`: 20-episode aggregate result.
- `outputs/offline_probe/*_report.json`: per-episode embodied bug reports.
- `outputs/libero_smoke.json`: simulator smoke-test result.
- `outputs/real_sim_failure_probe*.json`: non-injected simulator failure replay results.
- `outputs/pi05_natural_failure_probe.json`: natural pi05_libero rollout result.
- `test/outputs/test_shed_minimizer.json`: synthetic unit-test result.

## Interpretation

This v0 is deliberately conservative. It proves the minimization machinery can
recover injected failure windows from real LIBERO trajectories and that the
local LIBERO simulator can run. The `real_sim_failure_probe.py` script goes one
step further by collecting a non-injected failure in the simulator and verifying
candidate slices with MuJoCo state-reset replay. It still does not prove natural
VLA failure causality; the next step is to collect real failed rollouts from a
VLA policy and replace the simple geometric failure predicate with richer
same-failure labels.
