# pi05_base-libero_10

Independent run folder for `pi05_base` on `libero_10`.

Outputs stay under `./outputs`; this folder uses policy port `8011`.

`pi05_base` is a local PyTorch/safetensors checkpoint. The OpenPI venv needs
the local `transformers_replace` patch before serving this model. This folder
also sets `PYTORCH_DEVICE=cuda` and `PYTORCH_COMPILE_MODE=none` in
`config.env`, because the first `max-autotune` compile on Blackwell is too slow
for batch validation.

```bash
./run_smoke.sh
./start_background.sh
./status.sh
./stop.sh
```
