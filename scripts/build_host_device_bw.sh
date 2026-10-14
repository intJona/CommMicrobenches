#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DPCPP="${DPCPP:-/home/intel/dpcpp_nightly/bin/dpclang++}"
"$DPCPP" -std=c++17 -O2 -fsycl -fsycl-targets=spir64 \
    src/host_device_bw_bench.cpp -o src/host_device_bw_bench

echo "Built: $PROJECT_ROOT/src/host_device_bw_bench"
