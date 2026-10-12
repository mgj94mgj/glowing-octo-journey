#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -x cpp/build/vision ]]; then
  echo "C++ runtime is not built; run ./build_cpp.sh first" >&2
  exit 1
fi

exec cpp/build/vision "$@"
