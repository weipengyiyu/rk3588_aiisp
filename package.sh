#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SDK_ROOT="$(cd "$ROOT/../.." && pwd)"
DEST="${1:-$ROOT/dist/rk3588-aiisp}"
test -x "$ROOT/build-aarch64/rk3588_aiisp"
test -f "$ROOT/models/bayer_gaussian.rknn"
mkdir -p "$DEST/bin" "$DEST/lib" "$DEST/models" "$DEST/scripts" "$DEST/docs"
cp "$ROOT/build-aarch64/rk3588_aiisp" "$ROOT/build-aarch64/test_raw" "$ROOT/build-aarch64/test_pipeline" "$DEST/bin/"
cp "$SDK_ROOT/external/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so" "$DEST/lib/"
cp "$ROOT/models/bayer_gaussian.rknn" "$ROOT/models/baseline.json" "$DEST/models/"
cp "$ROOT/scripts/run_pipeline.py" "$ROOT/scripts/compare_raw.py" "$DEST/scripts/"
cp "$ROOT/scripts/run_imx623_fast.sh" "$DEST/scripts/"
if [[ -d "$ROOT/models/fast-rect" ]]; then
  cp -a "$ROOT/models/fast-rect" "$DEST/models/"
fi
cp "$ROOT/README.md" "$DEST/"
cp -a "$ROOT/docs/." "$DEST/docs/"
(cd "$DEST" && find bin models lib scripts docs -type f -print0 | sort -z | xargs -0 sha256sum > manifest.sha256 && sha256sum README.md >> manifest.sha256)
printf 'Bundle: %s\n' "$DEST"
