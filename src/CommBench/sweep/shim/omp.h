// Minimal omp.h shim: CommBench only needs omp_get_wtime() for host-side timing,
// not real OpenMP offload, so avoid pulling in clang's full OpenMP/SPIR-V toolchain.
#pragma once
#include <time.h>

static inline double omp_get_wtime(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
