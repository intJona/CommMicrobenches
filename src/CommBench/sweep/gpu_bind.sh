#!/bin/bash
# Pin each MPI rank to its own Intel GPU by PCI order (rank 0 -> GPU0, rank 1 -> GPU1).
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ZE_AFFINITY_MASK=$OMPI_COMM_WORLD_LOCAL_RANK
exec "$@"
