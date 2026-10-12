#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

cmake_args=(
  -S cpp
  -B cpp/build
  -DCMAKE_BUILD_TYPE=Release
  -DVISION_BUILD_TESTS=ON
)

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  cmake_args+=("-DCMAKE_PREFIX_PATH=${CONDA_PREFIX}")
fi

cmake "${cmake_args[@]}"
cmake --build cpp/build -j"$(nproc)"
ctest --test-dir cpp/build --output-on-failure
