# pi0fast-libero / libero_long_breakfast_cleanup

Isolated SHED-CFS runs for the custom LIBERO long-horizon breakfast cleanup task.

The task uses project-local BDDL and a stage oracle in `custom_tasks/long_horizon_breakfast_cleanup.py`.
Outputs stay under this model-dataset folder and are not mixed with `libero_10` reports.

Useful commands:

```bash
./run_smoke.sh
./start_ultra_video_eval.sh
./status_ultra_video_eval.sh
```

