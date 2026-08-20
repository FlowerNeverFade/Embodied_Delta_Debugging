# Embodied Delta Debugging

SHED-CFS research code for finding and validating minimal causal slices in
failed embodied-AI trajectories. The project contains the original prototype,
the frozen causal-v3 snapshot, and the causal-v4 implementation.

## Repository Contents

- Root Python modules and `test/`: the initial prototype and smoke tests.
- `model_datasets/`: experiment launch and process-management scripts (without
  model data, local configuration, or generated outputs).
- `v3/code_snapshot/`: the archived causal-v3 implementation.
- `v4/code/`, `v4/test/`, and `v4/scripts/`: the causal-v4 implementation,
  tests, and launch scripts.
- `*.md`, `*.docx`, `1.png`, and `1.pptx`: research notes and presentation
  artifacts.

Large downloaded assets and generated results are intentionally not tracked in
Git. The local workspace contains model checkpoints, datasets, videos, replay
outputs, caches, logs, and other artifacts that are unsuitable for a normal
GitHub repository. Recreate those assets with the project-specific download
scripts, or provide their paths through the command-line options documented in
the scripts.

## Quick Checks

Run the lightweight tests from an environment that provides the project
dependencies:

```bash
python -m py_compile *.py
python -m pytest -q test
```

For the causal-v4 snapshot:

```bash
PYTHONPATH=v4/code python -m py_compile v4/code/*.py
PYTHONPATH=v4/code python -m pytest -q v4/test
```

The simulator and policy-server probes require external LIBERO/OpenPI/VLA
installations and locally downloaded checkpoints; they are not bundled here.

## Notes

Several legacy scripts retain absolute paths from the original research
machine. Treat those defaults as examples and override them with the scripts'
CLI arguments or adapt them to the local environment before running.

No software license has been specified for this research snapshot yet.
