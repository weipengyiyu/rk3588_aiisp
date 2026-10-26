#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec python3 "$ROOT/scripts/run_pipeline.py" \
  --mode rknn-native --model "$ROOT/models/fast-pack16/bayer_gaussian.rknn" \
  --tile 488 --tile-height 392 --bayer-pack 2 --cpu-threads 4 --halo 1 --post-filter edge \
  --edge-strength 0.75 --edge-threshold 96 --performance \
  --output-dir "$ROOT/run-fast" "$@"
