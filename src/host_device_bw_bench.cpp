// Standalone single-GPU host<->device bandwidth probe, independent of CommBench/MPI.
//
// Purpose: the CommBench sweep's "staged" leg (GPU->host->GPU) combines two PCIe hops plus
// an inter-process MPI copy, and its ~6.3GB/s result could come from either hop being the
// bottleneck, or from MPI/synchronization overhead. This tool isolates a SINGLE host<->device
// hop on ONE GPU, with no MPI/IPC in the picture, as an independent reference number.
//
// It also reports both host wall-clock time (std::chrono around queue.wait()) AND device-side
// SYCL profiling timestamps (command_start/command_end, sourced from the Level-Zero event
// timestamps, immune to host dispatch/polling overhead) for the SAME copy, so the two can be
// compared directly to quantify how much (if any) host-side overhead is inflating the
// wall-clock numbers used elsewhere in these benchmarks.
//
// Build: ./scripts/build_host_device_bw.sh
// Run:   ./src/host_device_bw_bench [device_index]

#include <sycl/sycl.hpp>
#include <vector>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <algorithm>

static const size_t MIN_BYTES = 4ULL * 1024;
static const size_t MAX_BYTES = 256ULL * 1024 * 1024;
static const int WARMUP = 5;
static const int ITERS = 20;

static double median_of(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// One direction's full sweep. `to_device`: true = host->device, false = device->host.
static void sweep(sycl::queue &q, bool to_device) {
  const char *label = to_device ? "H->D" : "D->H";
  char *dev_buf = sycl::malloc_device<char>(MAX_BYTES, q);
  char *host_buf = sycl::malloc_host<char>(MAX_BYTES, q);
  std::memset(host_buf, 0x5A, MAX_BYTES);

  for (size_t bytes = MIN_BYTES; bytes <= MAX_BYTES; bytes *= 2) {
    for (int w = 0; w < WARMUP; w++) {
      if (to_device) q.memcpy(dev_buf, host_buf, bytes).wait();
      else q.memcpy(host_buf, dev_buf, bytes).wait();
    }

    std::vector<double> host_times, device_times;
    host_times.reserve(ITERS);
    device_times.reserve(ITERS);
    for (int i = 0; i < ITERS; i++) {
      auto t0 = std::chrono::high_resolution_clock::now();
      sycl::event ev = to_device ? q.memcpy(dev_buf, host_buf, bytes)
                                  : q.memcpy(host_buf, dev_buf, bytes);
      ev.wait();
      auto t1 = std::chrono::high_resolution_clock::now();
      host_times.push_back(std::chrono::duration<double>(t1 - t0).count());

      uint64_t start_ns = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
      uint64_t end_ns = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
      device_times.push_back((double)(end_ns - start_ns) * 1e-9);
    }

    double host_med = median_of(host_times);
    double dev_med = median_of(device_times);
    printf("%-4s %12zu bytes : host_wallclock=%8.3f GB/s (%.2f us)  device_profiled=%8.3f GB/s (%.2f us)  overhead=%.2f us\n",
           label, bytes,
           bytes / host_med / 1e9, host_med * 1e6,
           bytes / dev_med / 1e9, dev_med * 1e6,
           (host_med - dev_med) * 1e6);
  }

  sycl::free(dev_buf, q);
  sycl::free(host_buf, q);
}

int main(int argc, char *argv[]) {
  int device_index = argc > 1 ? std::atoi(argv[1]) : 0;
  auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  if (device_index >= (int)devices.size()) {
    fprintf(stderr, "device index %d out of range (found %zu GPUs)\n", device_index, devices.size());
    return 1;
  }
  sycl::queue q(devices[device_index], sycl::property::queue::enable_profiling{});
  printf("Device: %s\n\n", devices[device_index].get_info<sycl::info::device::name>().c_str());

  printf("==================== Host -> Device ====================\n");
  sweep(q, true);
  printf("\n==================== Device -> Host ====================\n");
  sweep(q, false);
  return 0;
}
