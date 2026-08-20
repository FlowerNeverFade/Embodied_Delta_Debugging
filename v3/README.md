# SHED-CFS Causal-v3 Workspace

This directory freezes the current causal-v3 implementation and indexes the main v3 artifacts.

## Layout

- `code_snapshot/`: copied snapshot of the current v3-related Python modules.
- `scripts/`: copied snapshot of the v3 runner scripts.
- `outputs/`: local-only links/indexes to large v3 result directories; excluded from Git.
- `docs/`: local-only link to the original research report; the tracked copy is in the top-level `docs/` directory.

## Important Outputs

- `outputs/slice_review_k5_confirmed_repaired/`
  - Repaired K=5 human review pack with quadriptych videos.
  - Includes fixed `task08_init46_seed17` repair replay context.
- `outputs/confirm_k5_no_timeout_setsid/`
  - K=5 confirmation rerun with `--case-timeout-seconds 0`.
  - Contains no-timeout confirmations such as `task09_init01_seed27` and `task08_init21_seed17`.
- `outputs/targeted_k1/`
  - Earlier targeted K=1 v3 candidate search.
- `outputs/more_full_success_k1/`
  - Broader K=1 v3 candidate search.

## Notes

This is an archival/index workspace for v3. The heavy outputs remain in
`model_datasets/pi0fast-libero-libero_10/outputs/` and are referenced locally by symlink.
Future causal-v4 work should be created in a separate sibling directory, `v4/`, rather than
modifying this v3 snapshot in place.
