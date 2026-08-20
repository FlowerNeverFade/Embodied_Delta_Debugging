# Prototype

The first SHED-CFS implementation validates trajectory minimization with
controlled failures and local LIBERO/VLABench data.

## Layout

- `code/`: importable Python modules and custom task definitions.
- `tests/`: unit tests for the minimizer, predicates, exports, and task oracles.
- `scripts/`: setup, download, and experiment helper scripts.

## Checks

```bash
PYTHONPATH=prototype/code python -m pytest -q prototype/tests
python -m compileall -q prototype/code
```

The data and simulator probes expect external datasets and environments. Use
their `--dataset-root`, `--output`, and environment-specific options when the
defaults do not match the local machine.
