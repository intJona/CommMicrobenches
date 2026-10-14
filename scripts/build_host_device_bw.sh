#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DPCPP="${DPCPP:-/home/intel/dpcpp_nightly/bin/dpclang++}"
"$DPCPP" -std=c++17 -O2 -fsycl -fsycl-targets=spir64 \
    -I/usr/include/level_zero \
    src/host_device_bw_bench.cpp -o src/host_device_bw_bench -lze_loader

echo "Built: $PROJECT_ROOT/src/host_device_bw_bench"
