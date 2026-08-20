# pi0fast-libero / libero_long_breakfast_curriculum_v2

Isolated SHED-CFS runs for the curriculum-v2 custom LIBERO breakfast tasks.

This suite replaces the legacy one-shot long task with four levels:

- L1: mug into microwave, close microwave
- L2: L1 plus bowl into bottom drawer, close drawer
- L3: L2 plus both moka pots on an already-on stove
- L4: L2 plus turn on stove, place both moka pots, turn stove off

Outputs stay under this model-dataset folder and are not mixed with legacy long-task or `libero_10` runs.

Useful commands:

```bash
./run_smoke.sh
./run_pilot_video_eval.sh
./start_pilot_video_eval_multigpu.sh
./status_pilot_video_eval_multigpu.sh
```

The scripted expert is marked `expert_repair_only=true`: it is used only for repair counterfactuals / candidate Repair SFT pairs, not as a policy success claim.
