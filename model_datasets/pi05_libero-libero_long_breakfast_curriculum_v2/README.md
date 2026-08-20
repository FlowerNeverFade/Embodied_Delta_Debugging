# pi05_libero / libero_long_breakfast_curriculum_v2

Isolated SHED-CFS runs for the curriculum-v2 custom LIBERO breakfast tasks.

This suite replaces the legacy one-shot long task with four levels:

- L1: mug into microwave, close microwave
- L2: L1 plus bowl into bottom drawer, close drawer
- L3: L2 plus both moka pots on an already-on stove
- L4: L2 plus turn on stove, place both moka pots, turn stove off

Outputs stay under this model-dataset folder and are not mixed with legacy long-task or `libero_10` runs.

This variant uses the OpenPI `pi05_libero` checkpoint at
`/root/autodl-tmp/research/VLA_SKILL/model/pi05_libero`.

Current task layout revision: `v2.1-clean-stage01`.

For custom curriculum runs, the probe retries reset seeds and rejects invalid initial
states before policy rollout. In particular, a sideways / unstable
`white_yellow_mug_1`, a target below the tabletop, or a Stage01 target too close to
other objects is reported as `invalid_init` and is excluded from event search,
minimization, Risk Critic export, and Repair SFT admission. This prevents bad reset
physics from being counted as model failure.

For v2 reports, Risk Critic positives are intentionally strict:
`same_failure_necessity_pass` or `repair_valid_causal_pass` is required. A slice
that merely reproduces the same semantic failure, but has no necessity core and no
repair-valid counterfactual, is not exported as a positive training sample.

Useful commands:

```bash
./run_smoke.sh
./run_pilot_video_eval.sh
./start_pilot_video_eval_multigpu.sh
./status_pilot_video_eval_multigpu.sh
```

The scripted expert is marked `expert_repair_only=true`: it is used only for repair counterfactuals / candidate Repair SFT pairs, not as a policy success claim.
