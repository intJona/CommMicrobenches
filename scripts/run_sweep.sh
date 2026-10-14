#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source scripts/lib_storage.sh
RESULTS_ROOT="$(resolve_results_root)"

SWEEP_DIR="$PROJECT_ROOT/src/CommBench/sweep"
BINARY="$SWEEP_DIR/sweep"
if [[ ! -x "$BINARY" ]]; then
    echo "Missing $BINARY. Run scripts/build.sh first." >&2
    exit 1
fi

CONFIG_NAME="${CONFIG_NAME:-unnamed}"
RUNS="${RUNS:-1}"
if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || [[ "$RUNS" -lt 1 ]]; then
    echo "RUNS must be a positive integer, got '$RUNS'" >&2
    exit 1
fi

# dpcpp_nightly's libsycl isn't on the default loader path.
export LD_LIBRARY_PATH="/home/intel/dpcpp_nightly/lib:${LD_LIBRARY_PATH:-}"

for ((run = 1; run <= RUNS; run++)); do
    RUN_ID="$(date +%Y%m%d-%H%M%S)_commbench_${CONFIG_NAME}"
    OUTPUT_DIR="$RESULTS_ROOT/$RUN_ID"
    mkdir -p "$OUTPUT_DIR"

    {
        echo "timestamp_utc=$(date --utc +%FT%TZ)"
        echo "config_name=$CONFIG_NAME"
        printf 'command=mpirun -np 2 %q %q\n' "$SWEEP_DIR/gpu_bind.sh" "$BINARY"
        lspci -d 8086:e223 2>&1
        xpu-smi discovery 2>&1 || true
    } >"$OUTPUT_DIR/environment.txt"

    echo "Running CommBench P2P sweep for configuration: $CONFIG_NAME (run $run/$RUNS)"
    (cd "$SWEEP_DIR" && mpirun -np 2 ./gpu_bind.sh ./sweep) 2>&1 | tee "$OUTPUT_DIR/commbench_sweep_4KB_to_256MB.txt"

    echo "Saved: $OUTPUT_DIR"
done
