# Embodied Delta Debugging

SHED-CFS research code for finding and validating minimal causal slices in
failed embodied-AI trajectories.

## Start Here

| Area | Purpose |
| --- | --- |
| [`v4/`](v4/) | Current causal-v4 implementation and tests |
| [`v3/`](v3/) | Frozen causal-v3 archive |
| [`prototype/`](prototype/) | Original prototype, utilities, and tests |
| [`docs/`](docs/) | Research reports, slides, and diagrams |
| [`model_datasets/`](model_datasets/) | Experiment launch/status scripts only |

The large local assets under `dataset/`, `model/`, `model_datasets/*/outputs/`,
and `outputs/` are intentionally excluded from Git. They contain downloaded
datasets, model checkpoints, videos, replay results, caches, and machine-local
configuration. Recreate or mount them separately before running simulator or
policy experiments.

## Quick Checks

Prototype unit tests:

```bash
PYTHONPATH=prototype/code python -m pytest -q prototype/tests
```

Current v4 unit tests:

```bash
PYTHONPATH=v4/code:$OPENPI_CLIENT_SRC python -m pytest -q v4/test
```

Syntax-only checks that do not need simulator assets:

```bash
python -m compileall -q prototype/code v4/code
find prototype v3 v4 -name '*.sh' -exec bash -n {} +
```

The simulator and policy-server probes require external LIBERO/OpenPI/VLA
installations and locally downloaded checkpoints. See the README in each
implementation directory for details.

## Project Notes

The prototype and v3 snapshot retain historical research defaults where they
refer to external datasets or environments. v4 resolves its project code from
the repository layout and accepts `EDD_PROJECT_ROOT` for a custom checkout.

No software license has been specified for this research snapshot yet.
