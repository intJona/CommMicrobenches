// P2P bandwidth/latency sweep between 2 Intel GPUs (sweep: 4KB -> 256MB).
//
// Rank 0 = GPU0 (sender), Rank 1 = GPU1 (receiver). Device binding is external via
// ZE_AFFINITY_MASK (see gpu_bind.sh) since the oneAPI port's sycl::queue uses gpu_selector_v.
//
// NOTE on CommBench's built-in IPC library: it reconstructs a ze_ipc_mem_handle_t by stuffing a
// raw fd (stolen cross-process via pidfd_getfd) into the handle's first bytes and zeroing the
// rest. On this driver (libze-intel-gpu 25.31) that produced ZE_RESULT_ERROR_INVALID_ARGUMENT --
// the handle's remaining metadata bytes matter here, not just the fd. So the IPC handle exchange
// below is done by hand instead: the *entire* handle blob from zeMemGetIpcHandle is preserved,
// and only the embedded fd is swapped for a cross-process-valid one, using a UNIX domain socket +
// SCM_RIGHTS (the standard, ptrace-independent way to hand a GPU dma-buf/drm fd to another
// process) rather than CommBench's pidfd_getfd trick (which also needs Yama ptrace_scope=0 and
// was not relied on here).
//
// Two legs are measured at each size in the sweep:
//   1. "p2p"    - genuine device-to-device transfer via Level-Zero IPC import + SYCL USM memcpy.
//   2. "staged" - manual GPU0 -> host -> GPU1 copy (device/host USM + MPI message for the
//                 inter-process hop), used as the "P2P disabled" baseline since it never uses a
//                 Level-Zero peer IPC mapping.
//
// Correctness (not just speed) is verified for both legs before trusting any timing number.

#define PORT_ONEAPI
#define CAP_ZE
#include "../commbench.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <sys/prctl.h>
#include <unistd.h>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <algorithm>

using namespace CommBench;

static const size_t MIN_BYTES = 4ULL * 1024;              // 4 KB
static const size_t MAX_BYTES = 256ULL * 1024 * 1024;     // 256 MB
static const int WARMUP = 5;
static const int ITERS = 20;
static const char *SOCK_PATH = "/tmp/p2psweep_ipc.sock";

static void fill_pattern(std::vector<char> &buf) {
  for (size_t i = 0; i < buf.size(); i++)
    buf[i] = (char)((i * 2654435761ULL) & 0xFF);
}

static double median_of(std::vector<double> &v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// Picks a command-queue-group ordinal with COPY support, preferring a dedicated copy/blitter
// engine (COPY but not COMPUTE) over the shared compute+copy group that SYCL's queue.memcpy()
// implicitly uses. Root-caused via host_device_bw_bench: routing through the dedicated copy
// engine via raw Level-Zero closes the ~32GB/s SYCL-abstraction ceiling up to ~55-57GB/s.
static uint32_t pick_copy_group_ordinal(ze_device_handle_t dev, bool &is_dedicated) {
  uint32_t count = 0;
  zeDeviceGetCommandQueueGroupProperties(dev, &count, nullptr);
  std::vector<ze_command_queue_group_properties_t> props(count);
  for (auto &p : props) p = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES, nullptr, 0, 0, 0};
  zeDeviceGetCommandQueueGroupProperties(dev, &count, props.data());

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

// Receiver side: accept one connection, hand over fd (SCM_RIGHTS) + the rest of the handle bytes.
static void serve_ipc_handle(ze_ipc_mem_handle_t &handle) {
  int listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  unlink(SOCK_PATH);
  struct sockaddr_un addr = {};
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, SOCK_PATH, sizeof(addr.sun_path) - 1);
  bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr));
  listen(listen_fd, 1);

  MPI_Barrier(MPI_COMM_WORLD); // tell sender the socket is ready to accept

  int conn_fd = accept(listen_fd, nullptr, nullptr);

  int embedded_fd;
  memcpy(&embedded_fd, handle.data, sizeof(int));

  struct msghdr msg = {};
  struct iovec iov;
  iov.iov_base = handle.data + sizeof(int);
  iov.iov_len = ZE_MAX_IPC_HANDLE_SIZE - sizeof(int);
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  char cmsgbuf[CMSG_SPACE(sizeof(int))];
  msg.msg_control = cmsgbuf;
  msg.msg_controllen = sizeof(cmsgbuf);
  struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  memcpy(CMSG_DATA(cmsg), &embedded_fd, sizeof(int));
  msg.msg_controllen = cmsg->cmsg_len;

  sendmsg(conn_fd, &msg, 0);
  close(conn_fd);
  close(listen_fd);
}

// Sender side: connect, receive fd (SCM_RIGHTS, valid in this process) + rest of handle bytes.
static ze_ipc_mem_handle_t fetch_ipc_handle() {
  MPI_Barrier(MPI_COMM_WORLD); // wait for receiver's listen() to be posted

  int sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  struct sockaddr_un addr = {};
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, SOCK_PATH, sizeof(addr.sun_path) - 1);
  while (connect(sock_fd, (struct sockaddr *)&addr, sizeof(addr)) != 0)
    usleep(10000); // receiver may not have called accept() yet

  ze_ipc_mem_handle_t handle = {};
  struct msghdr msg = {};
  struct iovec iov;
  iov.iov_base = handle.data + sizeof(int);
  iov.iov_len = ZE_MAX_IPC_HANDLE_SIZE - sizeof(int);
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  char cmsgbuf[CMSG_SPACE(sizeof(int))];
  msg.msg_control = cmsgbuf;
  msg.msg_controllen = sizeof(cmsgbuf);

  recvmsg(sock_fd, &msg, 0);
  struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
  int local_fd;
  memcpy(&local_fd, CMSG_DATA(cmsg), sizeof(int));
  memcpy(handle.data, &local_fd, sizeof(int));

  close(sock_fd);
  return handle;
}

int main(int argc, char *argv[]) {
  CommBench::init();

  if (numproc != 2) {
    if (myid == printid)
      printf("This test requires exactly 2 MPI ranks (one per GPU), got %d\n", numproc);
    MPI_Finalize();
    return 1;
  }

  const int sender = 0;
  const int recver = 1;

  char *dev_send = nullptr;
  char *dev_recv = nullptr;
  char *remote_recv = nullptr; // sender's imported view of the receiver's dev_recv
  char *remote_send = nullptr; // recver's imported view of the sender's dev_send (for the pull leg)
  char *host_stage = sycl::malloc_host<char>(MAX_BYTES, CommBench::q);

  auto zeContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(CommBench::q.get_context());
  auto zeDevice = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(CommBench::q.get_device());

  // Dedicated-copy-engine immediate command list for raw Level-Zero copies (both ranks use their
  // own local device's engine: sender for the P2P push leg, both ranks for the no-P2P staged leg's
  // host<->device copies, so it's a fair apples-to-apples comparison against the 'p2p' leg above).
  ze_command_list_handle_t copy_cl = nullptr;
  {
    bool dedicated = false;
    uint32_t ordinal = pick_copy_group_ordinal(zeDevice, dedicated);
    printf("rank %d: raw L0 using command queue group ordinal %u (%s copy engine)\n",
           myid, ordinal, dedicated ? "dedicated" : "shared compute+copy");
    ze_command_queue_desc_t cq_desc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, ordinal, 0, 0,
                                        ZE_COMMAND_QUEUE_MODE_DEFAULT, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    zeCommandListCreateImmediate(zeContext, zeDevice, &cq_desc, &copy_cl);
  }


  if (myid == sender)
    dev_send = sycl::malloc_device<char>(MAX_BYTES, CommBench::q);
  if (myid == recver)
    dev_recv = sycl::malloc_device<char>(MAX_BYTES, CommBench::q);

  // One-time IPC handle exchange for the whole sweep (same underlying allocation, varying count).
  if (myid == recver) {
    ze_ipc_mem_handle_t handle;
    ze_result_t res = zeMemGetIpcHandle(zeContext, dev_recv, &handle);
    if (res != ZE_RESULT_SUCCESS)
      printf("rank %d: zeMemGetIpcHandle failed: 0x%x\n", myid, res);
    serve_ipc_handle(handle);
  }
  if (myid == sender) {
    ze_ipc_mem_handle_t handle = fetch_ipc_handle();
    ze_result_t res = zeMemOpenIpcHandle(zeContext, zeDevice, handle, 0, (void **)&remote_recv);
    if (res != ZE_RESULT_SUCCESS)
      printf("rank %d: zeMemOpenIpcHandle failed: 0x%x\n", myid, res);
    else
      printf("rank %d: P2P IPC mapping established OK (remote_recv=%p)\n", myid, (void *)remote_recv);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Reciprocal mapping: sender exports dev_send so the receiver can issue a PULL copy
  // (read remote over PCIe, write local) instead of the sender PUSHing (read local, write remote).
  if (myid == sender) {
    ze_ipc_mem_handle_t handle;
    ze_result_t res = zeMemGetIpcHandle(zeContext, dev_send, &handle);
    if (res != ZE_RESULT_SUCCESS)
      printf("rank %d: zeMemGetIpcHandle (dev_send) failed: 0x%x\n", myid, res);
    serve_ipc_handle(handle);
  }
  if (myid == recver) {
    ze_ipc_mem_handle_t handle = fetch_ipc_handle();
    ze_result_t res = zeMemOpenIpcHandle(zeContext, zeDevice, handle, 0, (void **)&remote_send);
    if (res != ZE_RESULT_SUCCESS)
      printf("rank %d: zeMemOpenIpcHandle (remote_send) failed: 0x%x\n", myid, res);
    else
      printf("rank %d: P2P PULL IPC mapping established OK (remote_send=%p)\n", myid, (void *)remote_send);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  std::vector<char> host_pattern(MAX_BYTES);
  fill_pattern(host_pattern);
  std::vector<char> host_result(MAX_BYTES);

  if (myid == printid)
    printf("\n==================== P2P SWEEP: 4KB -> 256MB ====================\n\n");

  for (size_t bytes = MIN_BYTES; bytes <= MAX_BYTES; bytes *= 2) {

    // ---------------- Leg 1: genuine device-to-device P2P (Level-Zero IPC) ----------------
    {
      bool ok = true;
      if (myid == sender)
        CommBench::q.memcpy(dev_send, host_pattern.data(), bytes).wait();
      MPI_Barrier(MPI_COMM_WORLD);

      if (myid == sender)
        CommBench::q.memcpy(remote_recv, dev_send, bytes).wait();
      MPI_Barrier(MPI_COMM_WORLD);

      if (myid == recver) {
        CommBench::q.memcpy(host_result.data(), dev_recv, bytes).wait();
        ok = (memcmp(host_result.data(), host_pattern.data(), bytes) == 0);
      }
      MPI_Bcast(&ok, 1, MPI_C_BOOL, recver, MPI_COMM_WORLD);

      if (!ok) {
        if (myid == printid)
          printf("size %10zu bytes: P2P (IPC) leg FAILED CORRECTNESS CHECK -- skipping timing\n", bytes);
      } else {
        std::vector<double> times(ITERS);
        for (int iter = -WARMUP; iter < ITERS; iter++) {
          MPI_Barrier(MPI_COMM_WORLD);
          double t0 = MPI_Wtime();
          if (myid == sender)
            CommBench::q.memcpy(remote_recv, dev_send, bytes).wait();
          double elapsed = MPI_Wtime() - t0;
          MPI_Bcast(&elapsed, 1, MPI_DOUBLE, sender, MPI_COMM_WORLD);
          if (iter >= 0)
            times[iter] = elapsed;
        }
        if (myid == printid) {
          double minT = *std::min_element(times.begin(), times.end());
          double medT = median_of(times);
          double avgT = 0;
          for (double t : times) avgT += t;
          avgT /= times.size();
          printf("---- size %10zu bytes : P2P (IPC, Level-Zero) ----\n", bytes);
          printf("minTime: %.4e us, %.4e GB/s\n", minT * 1e6, bytes / minT / 1e9);
          printf("medTime: %.4e us, %.4e GB/s\n", medT * 1e6, bytes / medT / 1e9);
          printf("avgTime: %.4e us, %.4e GB/s\n\n", avgT * 1e6, bytes / avgT / 1e9);
        }
      }
    }

    // ---------------- Leg 1b: P2P PULL direction (receiver issues the read, not the sender) ----
    // NOTE: on this driver, reading a peer's memory through a zeMemOpenIpcHandle-imported
    // pointer silently returns stale/zero data (Level-Zero reports SUCCESS at every step,
    // confirmed by also bypassing SYCL and issuing the copy via raw zeCommandListAppendMemoryCopy
    // directly) -- only the PUSH direction (sender writes to the imported remote pointer) works.
    // The correctness check below is expected to fail every time; it's kept so the leg is
    // automatically re-tested (and would start reporting real numbers) if a future driver fixes this.
    {
      bool ok = true;
      if (myid == sender)
        CommBench::q.memcpy(dev_send, host_pattern.data(), bytes).wait();
      MPI_Barrier(MPI_COMM_WORLD);

      if (myid == recver)
        CommBench::q.memcpy(dev_recv, remote_send, bytes).wait();
      MPI_Barrier(MPI_COMM_WORLD);

      if (myid == recver) {
        CommBench::q.memcpy(host_result.data(), dev_recv, bytes).wait();
        ok = (memcmp(host_result.data(), host_pattern.data(), bytes) == 0);
      }
      MPI_Bcast(&ok, 1, MPI_C_BOOL, recver, MPI_COMM_WORLD);

      if (!ok) {
        if (myid == printid && bytes == MIN_BYTES)
          printf("size %10zu bytes (and all larger): P2P PULL (IPC) leg FAILED CORRECTNESS CHECK -- "
                 "reading a peer's memory via IPC import doesn't work on this driver, skipping timing for all sizes\n\n", bytes);
      } else {
        std::vector<double> times(ITERS);
        for (int iter = -WARMUP; iter < ITERS; iter++) {
          MPI_Barrier(MPI_COMM_WORLD);
          double t0 = MPI_Wtime();
          if (myid == recver)
            CommBench::q.memcpy(dev_recv, remote_send, bytes).wait();
          double elapsed = MPI_Wtime() - t0;
          MPI_Bcast(&elapsed, 1, MPI_DOUBLE, recver, MPI_COMM_WORLD);
          if (iter >= 0)
            times[iter] = elapsed;
        }
        if (myid == printid) {
          double minT = *std::min_element(times.begin(), times.end());
          double medT = median_of(times);
          double avgT = 0;
          for (double t : times) avgT += t;
          avgT /= times.size();
          printf("---- size %10zu bytes : P2P PULL (IPC, Level-Zero) ----\n", bytes);
          printf("minTime: %.4e us, %.4e GB/s\n", minT * 1e6, bytes / minT / 1e9);
          printf("medTime: %.4e us, %.4e GB/s\n", medT * 1e6, bytes / medT / 1e9);
          printf("avgTime: %.4e us, %.4e GB/s\n\n", avgT * 1e6, bytes / avgT / 1e9);
        }
      }
    }

    // ---------------- Leg 1c: P2P push via raw Level-Zero on the dedicated copy engine --------
    // Same transfer as Leg 1 (sender writes dev_send -> remote_recv), but issued through
    // zeCommandListAppendMemoryCopy on a dedicated-copy-engine immediate command list instead
    // of SYCL's queue.memcpy(), to bypass the SYCL-runtime abstraction tax confirmed in
    // host_device_bw_bench (SYCL plateaus ~32GB/s, raw L0 dedicated copy engine reaches ~55-57GB/s).
    {
      bool ok = true;
      if (myid == sender)
        CommBench::q.memcpy(dev_send, host_pattern.data(), bytes).wait();
      MPI_Barrier(MPI_COMM_WORLD);

      if (myid == sender) {
        zeCommandListAppendMemoryCopy(copy_cl, remote_recv, dev_send, bytes, nullptr, 0, nullptr);
        zeCommandListHostSynchronize(copy_cl, UINT64_MAX);
      }
      MPI_Barrier(MPI_COMM_WORLD);

      if (myid == recver) {
        CommBench::q.memcpy(host_result.data(), dev_recv, bytes).wait();
        ok = (memcmp(host_result.data(), host_pattern.data(), bytes) == 0);
      }
      MPI_Bcast(&ok, 1, MPI_C_BOOL, recver, MPI_COMM_WORLD);

      if (!ok) {
        if (myid == printid)
          printf("size %10zu bytes: P2P (raw L0) leg FAILED CORRECTNESS CHECK -- skipping timing\n", bytes);
      } else {
        std::vector<double> times(ITERS);
        for (int iter = -WARMUP; iter < ITERS; iter++) {
          MPI_Barrier(MPI_COMM_WORLD);
          double t0 = MPI_Wtime();
          if (myid == sender) {
            zeCommandListAppendMemoryCopy(copy_cl, remote_recv, dev_send, bytes, nullptr, 0, nullptr);
            zeCommandListHostSynchronize(copy_cl, UINT64_MAX);
          }
          double elapsed = MPI_Wtime() - t0;
          MPI_Bcast(&elapsed, 1, MPI_DOUBLE, sender, MPI_COMM_WORLD);
          if (iter >= 0)
            times[iter] = elapsed;
        }
        if (myid == printid) {
          double minT = *std::min_element(times.begin(), times.end());
          double medT = median_of(times);
          double avgT = 0;
          for (double t : times) avgT += t;
          avgT /= times.size();
          printf("---- size %10zu bytes : P2P (raw Level-Zero, dedicated copy engine) ----\n", bytes);
          printf("minTime: %.4e us, %.4e GB/s\n", minT * 1e6, bytes / minT / 1e9);
          printf("medTime: %.4e us, %.4e GB/s\n", medT * 1e6, bytes / medT / 1e9);
          printf("avgTime: %.4e us, %.4e GB/s\n\n", avgT * 1e6, bytes / avgT / 1e9);
        }
      }
    }

    // ---------------- Leg 2: staged GPU0 -> host -> GPU1 ("P2P disabled" baseline), raw L0 ------
    // Same host-staged mechanism as before, but the host<->device copies now go through raw
    // Level-Zero on each rank's own dedicated copy engine instead of SYCL queue.memcpy() -- an
    // apples-to-apples comparison against the 'p2p' leg above (both bypass the SYCL abstraction
    // tax; the only remaining variable is P2P-over-PCIe vs round-tripping through host RAM).
    {
      bool ok = true;
      if (myid == sender)
        CommBench::q.memcpy(dev_send, host_pattern.data(), bytes).wait();

      // Untimed correctness pass.
      if (myid == sender) {
        zeCommandListAppendMemoryCopy(copy_cl, host_stage, dev_send, bytes, nullptr, 0, nullptr);
        zeCommandListHostSynchronize(copy_cl, UINT64_MAX);
        MPI_Send(host_stage, bytes, MPI_BYTE, recver, 0, MPI_COMM_WORLD);
      } else {
        MPI_Recv(host_stage, bytes, MPI_BYTE, sender, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        zeCommandListAppendMemoryCopy(copy_cl, dev_recv, host_stage, bytes, nullptr, 0, nullptr);
        zeCommandListHostSynchronize(copy_cl, UINT64_MAX);
        CommBench::q.memcpy(host_result.data(), dev_recv, bytes).wait();
        ok = (memcmp(host_result.data(), host_pattern.data(), bytes) == 0);
      }
      MPI_Bcast(&ok, 1, MPI_C_BOOL, recver, MPI_COMM_WORLD);

      if (!ok) {
        if (myid == printid)
          printf("size %10zu bytes: staged leg FAILED CORRECTNESS CHECK -- skipping timing\n\n", bytes);
      } else {
        std::vector<double> times(ITERS);
        for (int iter = -WARMUP; iter < ITERS; iter++) {
          MPI_Barrier(MPI_COMM_WORLD);
          double t0 = MPI_Wtime();
          if (myid == sender) {
            zeCommandListAppendMemoryCopy(copy_cl, host_stage, dev_send, bytes, nullptr, 0, nullptr);
            zeCommandListHostSynchronize(copy_cl, UINT64_MAX);
            MPI_Send(host_stage, bytes, MPI_BYTE, recver, 1, MPI_COMM_WORLD);
            int ack;
            MPI_Recv(&ack, 1, MPI_INT, recver, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          } else {
            MPI_Recv(host_stage, bytes, MPI_BYTE, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            zeCommandListAppendMemoryCopy(copy_cl, dev_recv, host_stage, bytes, nullptr, 0, nullptr);
            zeCommandListHostSynchronize(copy_cl, UINT64_MAX);
            int ack = 1;
            MPI_Send(&ack, 1, MPI_INT, sender, 2, MPI_COMM_WORLD);
          }
          double elapsed = MPI_Wtime() - t0;
          MPI_Bcast(&elapsed, 1, MPI_DOUBLE, sender, MPI_COMM_WORLD);
          if (iter >= 0)
            times[iter] = elapsed;
        }
        if (myid == printid) {
          double minT = *std::min_element(times.begin(), times.end());
          double medT = median_of(times);
          double avgT = 0;
          for (double t : times) avgT += t;
          avgT /= times.size();
          printf("---- size %10zu bytes : staged (GPU->host->GPU, P2P disabled) ----\n", bytes);
          printf("minTime: %.4e us, %.4e GB/s\n", minT * 1e6, bytes / minT / 1e9);
          printf("medTime: %.4e us, %.4e GB/s\n", medT * 1e6, bytes / medT / 1e9);
          printf("avgTime: %.4e us, %.4e GB/s\n\n", avgT * 1e6, bytes / avgT / 1e9);
        }
      }
    }
  }

  if (myid == sender && dev_send) sycl::free(dev_send, CommBench::q);
  if (myid == recver && dev_recv) sycl::free(dev_recv, CommBench::q);
  sycl::free(host_stage, CommBench::q);
  if (copy_cl) zeCommandListDestroy(copy_cl);

  MPI_Finalize();
  return 0;
}
