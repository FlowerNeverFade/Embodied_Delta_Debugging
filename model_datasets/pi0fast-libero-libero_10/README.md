# pi0fast-libero / libero_10

Isolated SHED-CFS Risk Critic runs for `/root/autodl-tmp/research/VLA_SKILL/model/pi0fast-libero`.

`policy_overlay/` is run-local OpenPI compatibility glue: it symlinks the original `model.safetensors` and reuses local LIBERO normalization stats. The source checkpoint is not modified.
