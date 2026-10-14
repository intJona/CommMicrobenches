#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT/src/CommBench/sweep"

export OMPI_CXX="${OMPI_CXX:-/home/intel/dpcpp_nightly/bin/dpclang++}"
make clean
make

echo "Built: $PROJECT_ROOT/src/CommBench/sweep/sweep"
