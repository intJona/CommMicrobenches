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
  char *host_stage = sycl::malloc_host<char>(MAX_BYTES, CommBench::q);

  auto zeContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(CommBench::q.get_context());
  auto zeDevice = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(CommBench::q.get_device());

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

    // ---------------- Leg 2: staged GPU0 -> host -> GPU1 ("P2P disabled" baseline) ----------
    {
      bool ok = true;
      if (myid == sender)
        CommBench::q.memcpy(dev_send, host_pattern.data(), bytes).wait();

      // Untimed correctness pass.
      if (myid == sender) {
        CommBench::q.memcpy(host_stage, dev_send, bytes).wait();
        MPI_Send(host_stage, bytes, MPI_BYTE, recver, 0, MPI_COMM_WORLD);
      } else {
        MPI_Recv(host_stage, bytes, MPI_BYTE, sender, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        CommBench::q.memcpy(dev_recv, host_stage, bytes).wait();
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
            CommBench::q.memcpy(host_stage, dev_send, bytes).wait();
            MPI_Send(host_stage, bytes, MPI_BYTE, recver, 1, MPI_COMM_WORLD);
            int ack;
            MPI_Recv(&ack, 1, MPI_INT, recver, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          } else {
            MPI_Recv(host_stage, bytes, MPI_BYTE, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CommBench::q.memcpy(dev_recv, host_stage, bytes).wait();
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

  MPI_Finalize();
  return 0;
}
