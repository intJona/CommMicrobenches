# CommBench P2P Sweep

GPU-to-GPU PCIe peer-to-peer bandwidth/latency sweep between two Intel GPUs, built on
[CommBench](https://github.com/merthidayetoglu/CommBench). Use it to compare PCIe
configurations (topology, lane width, ACS, switch vs. direct attach, ...).

Two MPI ranks, one per GPU, send messages from 4 KB to 256 MB. Each size is measured over
several transfer paths so P2P can be compared against a host-staged baseline.

## Why this exists

- **Why not stock CommBench?** Its built-in Level Zero IPC code rebuilds the IPC handle from a
  stolen file descriptor and zeroes the rest. On Intel driver `libze-intel-gpu` 25.31 that fails
  with `ZE_RESULT_ERROR_INVALID_ARGUMENT` (per the comment in `sweep/main.cpp`; not re-tested
  on other driver versions). `sweep/main.cpp` instead keeps the full handle and passes only the
  fd between processes over a Unix socket (`SCM_RIGHTS`), which also avoids needing
  `ptrace_scope=0`.
- **Why a P2P-off baseline?** To show what P2P actually buys, the sweep also copies
  GPU → host → GPU without any peer mapping.
- **Why correctness checks?** Every leg verifies the received data at every size before its
  timing is trusted.
- **Why the wrapper scripts?** `scripts/run_sweep.sh` pins one rank per GPU, repeats runs,
  saves each run in its own timestamped folder with a hardware snapshot, and picks the results
  location, so runs from different PCIe configurations stay comparable.

`src/CommBench/` is a minimal standalone copy of the upstream headers (Apache-2.0) plus our
`sweep/main.cpp`. The full upstream repo is not needed.

## Requirements

- Linux, 2 Intel GPUs with a Level Zero driver
- Open MPI (`mpirun`, `mpicxx`)
- Intel oneAPI DPC++/SYCL compiler (`dpclang++`) with its runtime libraries
- Level Zero development headers (`libze-dev`)

The DPC++ install location is hard-coded in `src/CommBench/sweep/Makefile`,
`scripts/build.sh` and `scripts/run_sweep.sh`. Update it there for your system.

## Build (once)

```bash
./scripts/build.sh
```

## Run a sweep

```bash
CONFIG_NAME=my_config RUNS=5 ./scripts/run_sweep.sh
```

| Variable | Default | Description |
|---|---|---|
| `CONFIG_NAME` | `unnamed` | Label for the physical PCIe configuration under test |
| `RUNS` | `1` | How many times to repeat the sweep |
| `COMMBENCH_RESULTS_DIR` | `./results` | Where results are written (see [Storage](#storage)) |

Each run creates a timestamped folder containing:

- `environment.txt` — timestamp, config name, command, `lspci` / `xpu-smi` snapshot
- `commbench_sweep_4KB_to_256MB.txt` — raw sweep output

Settings fixed in `sweep/main.cpp`: 5 warm-up and 20 timed iterations per size and leg.
Rank 0 sends from GPU 0 and rank 1 receives on GPU 1 (pinned by `gpu_bind.sh`).

### What is measured

| Dashboard trace | Path |
|---|---|
| `p2p` | Direct GPU → GPU via Level Zero IPC, issued on the dedicated copy engine; shown by default |
| `p2p_sycl` | Same transfer via a plain SYCL copy |
| `staged` | GPU → host → GPU in overlapping chunks (P2P-off baseline) |

The raw sweep output also contains other legs (non-pipelined staged copy, P2P pull). The
dashboard ignores them.

## Dashboard

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

Compares bandwidth and latency across saved runs and configurations.
The dashboard can run on a different machine from the benchmark, as long as both point
at the same results directory.

## Storage

Results location, highest priority first:

1. `COMMBENCH_RESULTS_DIR` environment variable
2. `.storage-config` (gitignored; copy from `.storage-config.example`)
3. `./results`
