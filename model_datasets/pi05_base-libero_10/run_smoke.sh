#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RUN_DIR"
./run_foreground.sh --task-ids 8 --init-state-ids 2 --seeds 7 --max-cases 1 --no-resume
