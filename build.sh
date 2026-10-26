#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SDK_ROOT="$(cd "$ROOT/../.." && pwd)"
BUILD="${1:-$ROOT/build-aarch64}"
CXX="${CXX:-aarch64-linux-gnu-g++}"
python3 "$ROOT/scripts/check_sdk_abi.py" "$SDK_ROOT"
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DCMAKE_CXX_COMPILER="$CXX" -DSDK_ROOT="$SDK_ROOT" -DBUILD_TESTING=ON
cmake --build "$BUILD" --parallel 4
file "$BUILD/rk3588_aiisp"
