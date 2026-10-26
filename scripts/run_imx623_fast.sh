#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec python3 "$ROOT/scripts/run_pipeline.py" \
  --mode rknn-native --model "$ROOT/models/fast-rect/bayer_gaussian.rknn" \
  --tile 976 --tile-height 784 --halo 1 --post-filter edge \
  --edge-strength 0.75 --edge-threshold 96 --performance \
  --output-dir "$ROOT/run-fast" "$@"
