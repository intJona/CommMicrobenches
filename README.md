# PCIe CommBench Benchmarks

Reproducible Intel Arc GPU-to-GPU PCIe peer-to-peer microbenchmarks for this workstation,
using [CommBench](https://github.com/merthidayetoglu/CommBench) (Stanford/ANL/ORNL's
multi-vendor GPU communication micro-benchmarking tool) as the engine. This is a sibling to
[PCIe NCCL Benchmarks](../PCIe%20NCCL%20Benchmarks/README.md) and
[PCIe ZePeer Benchmarks](../PCIe%20ZePeer%20Benchmarks/README.md) -- same goal (compare PCIe
configurations), third independent data point: a custom CommBench-based sweep
(`src/CommBench/sweep/main.cpp`) that measures genuine Level-Zero IPC device-to-device P2P
copy bandwidth/latency, alongside a manual GPU-to-host-to-GPU "staged" baseline (P2P disabled)
for comparison, across 2 MPI ranks (one per GPU).

`src/CommBench/` contains only the files needed to build the sweep (`commbench.h`, `comm.h`,
`util.h`, `LICENSE` copied from upstream CommBench, Apache-2.0) plus our own
`sweep/main.cpp` -- this is a standalone copy, not a clone of the full CommBench repo, so
building/running here never touches the original checkout under
`~/pcie package bandwidth sweep/CommBench`.

## Why a custom sweep instead of CommBench's stock examples

CommBench's built-in IPC library reconstructs a `ze_ipc_mem_handle_t` by stuffing a raw fd
(stolen cross-process via `pidfd_getfd`) into the handle's first bytes and zeroing the rest.
On this driver (libze-intel-gpu 25.31) that produces `ZE_RESULT_ERROR_INVALID_ARGUMENT` --
the handle's remaining metadata bytes matter here, not just the fd. `sweep/main.cpp` does the
IPC handle exchange by hand instead: the *entire* handle blob from `zeMemGetIpcHandle` is
preserved, and only the embedded fd is swapped for a cross-process-valid one over a Unix
domain socket with `SCM_RIGHTS` (the standard way to hand a GPU dma-buf/drm fd to another
process), rather than CommBench's `pidfd_getfd` trick (which also needs Yama
`ptrace_scope=0`).

Correctness (not just speed) is verified for both legs (`p2p` and `staged`) at every size
before trusting any timing number.

## First build

```bash
cd "/home/intel/PCIe CommBench Benchmarks"
./scripts/build.sh
```

Builds `src/CommBench/sweep/sweep` via `mpicxx` + `dpclang++` (oneAPI DPC++ SYCL compiler,
`/home/intel/dpcpp_nightly`) targeting `spir64`. Override the compiler with
`OMPI_CXX=<path> ./scripts/build.sh` if needed.

## Sweep

```bash
./scripts/run_sweep.sh
```

Runs the sweep with `mpirun -np 2` (rank 0 = GPU0 sender, rank 1 = GPU1 receiver, pinned via
`gpu_bind.sh`/`ZE_AFFINITY_MASK`) across message sizes 4 KB -> 256 MB, 5 warmup + 20 measured
iterations per size/leg. Environment variables:

- `CONFIG_NAME` -- label for this PCIe configuration (default `unnamed`)
- `RUNS` -- repeat count (default `1`)
- `COMMBENCH_RESULTS_DIR` -- results dir override (see storage config below)

Each run writes `environment.txt` (timestamp, config name, command, `lspci`/`xpu-smi`
snapshot) and `commbench_sweep_4KB_to_256MB.txt` (raw stdout) to a timestamped subfolder.

## Storage configuration

Same precedence pattern as the sibling projects: `$COMMBENCH_RESULTS_DIR` env var >
`.storage-config` (gitignored, copy from `.storage-config.example`) > `./results`.

## Dashboard

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

Compares bandwidth/latency across saved runs/configurations, with a toggle for which
statistic (min/median/avg across iterations) to plot, and separate series for the `p2p`
(genuine Level-Zero IPC) and `staged` (P2P-disabled baseline) legs.
