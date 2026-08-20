# SHED-CFS Causal-v4 Workspace

This directory contains the isolated causal-v4 implementation. It does not modify
the frozen `v3/` snapshot and it uses separate output directories and ports.

## What Changed

- Schema: `shed-cfs-causal-v4-global-multimodal`.
- Report layers are explicit:
  - `minimal_same_failure_slice`
  - `causal_core_units`
  - `repair_replay_context`
- Repair evidence is ranked separately:
  - `policy_raw_repair_valid_pass`
  - `policy_language_phrase_repair_valid_pass`
  - `policy_visual_mask_repair_valid_pass`
  - `demo_existence_repair_pass`
- Candidate units now include action chunks, gripper transitions, object motion,
  goal predicate transitions, contact events, language phrases, visual grounding
  masks, and state anchors.
- Contact evidence records MuJoCo geom/body/contact metadata when available.
- v4 outputs top-k bounded-minimal causal sets in `k_minimal_causal_sets`.

## Layout

- `code/`: runnable v4 code copied from v3 and upgraded in place.
- `scripts/`: v4 launch scripts using ports `8060-8062`.
- `test/`: v4-specific unit tests.
- `outputs/`: local-only links/indexes to large result directories; excluded from Git.
- `docs/`: local-only link to the original research report; the tracked copy is in the top-level `docs/` directory.

## Quick Checks

```bash
PYTHONPATH=/root/autodl-tmp/research/Embodied_Delta_Debugging/v4/code \
  /root/autodl-tmp/envs/libero38/bin/python -m py_compile \
  /root/autodl-tmp/research/Embodied_Delta_Debugging/v4/code/*.py

PYTHONPATH=/root/autodl-tmp/research/Embodied_Delta_Debugging/v4/code \
  /root/autodl-tmp/envs/libero38/bin/python -m pytest -q \
  /root/autodl-tmp/research/Embodied_Delta_Debugging/v4/test
```

## First Runs

- Gold regression K=5:
  `bash v4/scripts/run_v4_gold_regression_k5.sh`
- Targeted K=1 pilot on three GPUs:
  `bash v4/scripts/run_v4_targeted_k1_multigpu.sh`

Both scripts explicitly unset proxy environment variables before launching model
or simulator processes.

## Cost Knobs

- Replay cache is enabled by default and reuses physically equivalent replays
  across unit/group validations while restaging the report entry.
- Exact sequential trial pruning is enabled by default: K=5 stops only when the
  threshold pass/fail decision can no longer change.
- Hierarchical causal pruning is enabled by default: multimodal groups are
  tested first, and children are skipped only when the group CE is below the
  threshold. The report records `hierarchical_pruning_trace`.
- `--defer-source-repair` marks demo/success-NN repair as deferred during
  candidate mining; K=5 confirmation/review should rerun without it.
- `--causal-max-units`: caps how many multimodal units are ablated/repaired.
- `--causal-ablation-trials`: repeats each destructive intervention for CE.
- `--causal-ablation-strategies`: choose destructive variants; `hold` is fastest.
- `--disable-source-repair`: skips scripted/demo/success-NN repair and only tests policy repair.
- `--skip-source-repair-if-policy-pass`: keeps demo repair only for units not already repaired by policy.
- `--stop-after-first-repair-valid-core`: stops once a repair-valid core is found.

Recommended pattern: K=1 candidate search uses `hold`, `--causal-max-units 18`,
and `--defer-source-repair`; K=5 confirmation re-enables fuller strategies,
`--causal-ablation-trials 5`, and source repair for shortlisted candidates.
