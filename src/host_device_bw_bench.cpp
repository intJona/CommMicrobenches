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
#include <level_zero/ze_api.h>
#include <vector>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <algorithm>

static const size_t MIN_BYTES = 4ULL * 1024;
static const size_t MAX_BYTES = 256ULL * 1024 * 1024;
static const int WARMUP = 5;
static const int ITERS = 20;

#define CHECK_ZE(call) do { \
    ze_result_t _r = (call); \
    if (_r != ZE_RESULT_SUCCESS) { \
      fprintf(stderr, "%s failed: 0x%x (%s:%d)\n", #call, _r, __FILE__, __LINE__); \
      std::exit(1); \
    } \
  } while (0)

static double median_of(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// One direction's full sweep. `to_device`: true = host->device, false = device->host.
static void sweep(sycl::queue &q, bool to_device) {
  const char *label = to_device ? "H->D" : "D->H";
  char *dev_buf = sycl::malloc_device<char>(MAX_BYTES, q);
  char *host_buf = sycl::malloc_host<char>(MAX_BYTES, q);
  char *host_check = sycl::malloc_host<char>(MAX_BYTES, q);
  for (size_t i = 0; i < MAX_BYTES; i++) host_buf[i] = (char)((i * 2654435761ULL) & 0xFF);

  for (size_t bytes = MIN_BYTES; bytes <= MAX_BYTES; bytes *= 2) {
    for (int w = 0; w < WARMUP; w++) {
      if (to_device) q.memcpy(dev_buf, host_buf, bytes).wait();
      else q.memcpy(host_buf, dev_buf, bytes).wait();
    }

    // Correctness check before trusting any timing: round-trip through dev_buf and compare.
    q.memcpy(dev_buf, host_buf, bytes).wait();
    q.memcpy(host_check, dev_buf, bytes).wait();
    if (memcmp(host_check, host_buf, bytes) != 0) {
      printf("%-4s %12zu bytes : CORRECTNESS CHECK FAILED -- skipping timing\n", label, bytes);
      continue;
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
  sycl::free(host_check, q);
}

// Picks a command-queue-group ordinal for raw Level-Zero copies. Prefers a group that
// supports COPY but NOT COMPUTE (a dedicated copy/blitter engine, distinct from the
// general compute engine group SYCL's queue.memcpy() implicitly uses) to test whether
// routing the copy through a different hardware engine changes the bandwidth ceiling.
static uint32_t pick_copy_group_ordinal(ze_device_handle_t dev, bool &is_dedicated) {
  uint32_t count = 0;
  CHECK_ZE(zeDeviceGetCommandQueueGroupProperties(dev, &count, nullptr));
  std::vector<ze_command_queue_group_properties_t> props(count);
  for (auto &p : props) p = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES, nullptr, 0, 0, 0};
  CHECK_ZE(zeDeviceGetCommandQueueGroupProperties(dev, &count, props.data()));

  int fallback = -1;
  for (uint32_t i = 0; i < count; i++) {
    bool copy = props[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY;
    bool compute = props[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE;
    if (copy && !compute) {
      is_dedicated = true;
      return i;
    }
    if (copy && fallback < 0) fallback = (int)i;
  }
  is_dedicated = false;
  return fallback >= 0 ? (uint32_t)fallback : 0;
}

// Same sweep as above, but issues the copy via a raw Level-Zero immediate command list
// instead of SYCL's queue.memcpy(), to test whether the SYCL runtime abstraction itself
// (already confirmed to cost ~15GB/s vs raw L0 for GPU<->GPU P2P, see ze_peer comparison)
// is also responsible for the host<->device D->H bandwidth ceiling.
static void sweep_raw_l0(ze_context_handle_t ctx, ze_device_handle_t dev, bool to_device) {
  const char *label = to_device ? "H->D" : "D->H";
  bool dedicated = false;
  uint32_t ordinal = pick_copy_group_ordinal(dev, dedicated);
  printf("(raw L0 using command queue group ordinal %u, %s copy engine)\n",
         ordinal, dedicated ? "dedicated" : "shared compute+copy");

  ze_command_list_desc_t cl_desc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, ordinal, 0};
  ze_command_queue_desc_t cq_desc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, ordinal, 0, 0,
                                      ZE_COMMAND_QUEUE_MODE_DEFAULT, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_list_handle_t cl;
  CHECK_ZE(zeCommandListCreateImmediate(ctx, dev, &cq_desc, &cl));

  ze_device_mem_alloc_desc_t dev_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  ze_host_mem_alloc_desc_t host_desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
  void *dev_buf = nullptr, *host_buf = nullptr, *host_check = nullptr;
  CHECK_ZE(zeMemAllocDevice(ctx, &dev_desc, MAX_BYTES, 64, dev, &dev_buf));
  CHECK_ZE(zeMemAllocHost(ctx, &host_desc, MAX_BYTES, 64, &host_buf));
  CHECK_ZE(zeMemAllocHost(ctx, &host_desc, MAX_BYTES, 64, &host_check));
  char *hb = (char *)host_buf;
  for (size_t i = 0; i < MAX_BYTES; i++) hb[i] = (char)((i * 2654435761ULL) & 0xFF);

  for (size_t bytes = MIN_BYTES; bytes <= MAX_BYTES; bytes *= 2) {
    void *dst0 = to_device ? dev_buf : host_buf;
    void *src0 = to_device ? host_buf : dev_buf;
    for (int w = 0; w < WARMUP; w++) {
      CHECK_ZE(zeCommandListAppendMemoryCopy(cl, dst0, src0, bytes, nullptr, 0, nullptr));
      CHECK_ZE(zeCommandListHostSynchronize(cl, UINT64_MAX));
    }

    CHECK_ZE(zeCommandListAppendMemoryCopy(cl, dev_buf, host_buf, bytes, nullptr, 0, nullptr));
    CHECK_ZE(zeCommandListHostSynchronize(cl, UINT64_MAX));
    CHECK_ZE(zeCommandListAppendMemoryCopy(cl, host_check, dev_buf, bytes, nullptr, 0, nullptr));
    CHECK_ZE(zeCommandListHostSynchronize(cl, UINT64_MAX));
    if (memcmp(host_check, host_buf, bytes) != 0) {
      printf("%-4s %12zu bytes : CORRECTNESS CHECK FAILED -- skipping timing\n", label, bytes);
      continue;
    }

    std::vector<double> times;
    times.reserve(ITERS);
    for (int i = 0; i < ITERS; i++) {
      auto t0 = std::chrono::high_resolution_clock::now();
      CHECK_ZE(zeCommandListAppendMemoryCopy(cl, dst0, src0, bytes, nullptr, 0, nullptr));
      CHECK_ZE(zeCommandListHostSynchronize(cl, UINT64_MAX));
      auto t1 = std::chrono::high_resolution_clock::now();
      times.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    double med = median_of(times);
    printf("%-4s %12zu bytes : raw_l0=%8.3f GB/s (%.2f us)\n",
           label, bytes, bytes / med / 1e9, med * 1e6);
  }

  zeMemFree(ctx, dev_buf);
  zeMemFree(ctx, host_buf);
  zeMemFree(ctx, host_check);
  zeCommandListDestroy(cl);
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

  auto zeContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
  auto zeDevice = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());

  printf("\n================ Raw Level-Zero: Host -> Device ================\n");
  sweep_raw_l0(zeContext, zeDevice, true);
  printf("\n================ Raw Level-Zero: Device -> Host ================\n");
  sweep_raw_l0(zeContext, zeDevice, false);
  return 0;
}
