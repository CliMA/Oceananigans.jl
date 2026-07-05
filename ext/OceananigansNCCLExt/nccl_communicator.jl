# Keep every NCCL communicator we create alive for the lifetime of the process.
#
# `NCCL.Communicator` installs a finalizer that calls `ncclCommDestroy`, which is itself a
# blocking collective. Julia's GC runs at uncoordinated times across MPI ranks, so if a
# communicator is allowed to become garbage, GC on one rank can fire `ncclCommDestroy` while
# the other ranks are inside an unrelated NCCL collective (`ncclCommInitRank` during the
# creation of another communicator, or a `SendRecv` during a halo fill) — and the job
# deadlocks. Rooting the communicators here means their finalizers never run at an
# uncoordinated point; they are released together at process teardown instead.
#
# A normal simulation creates a single, long-lived communicator and never hits this. The
# hazard is specific to creating several communicators in one process (e.g. a test suite that
# sweeps decompositions), which is exactly where the deadlock was observed.
const _PERSISTENT_NCCL_COMMUNICATORS = Any[]

"""
    create_nccl_comm_from_mpi(mpi_subcomm)

Create an NCCL communicator that mirrors the given MPI subcommunicator.
Rank 0 within the subcommunicator generates a UniqueID and broadcasts it
to all ranks via MPI, then all ranks join the NCCL communicator.

Returns `nothing` if the subcommunicator has only 1 rank (no communication needed).
"""
function create_nccl_comm_from_mpi(mpi_subcomm)
    nranks = MPI.Comm_size(mpi_subcomm)
    nranks == 1 && return nothing

    # The UniqueID broadcast and `ncclCommInitRank` (inside `NCCL.Communicator`) are blocking
    # collectives that every rank must enter together. Julia's GC is uncoordinated across ranks;
    # if it fires here it can run a stale `NCCL.Communicator` finalizer — which calls
    # `ncclCommDestroy`, itself a blocking collective — on one rank while the others are mid
    # `ncclCommInitRank`, deadlocking the job. Disable GC across the collective creation so no
    # finalizer can interleave with it. A single long-lived communicator never hits this; it
    # bites code that creates several communicators in one process (e.g. the test suite).
    gc_was_enabled = GC.enable(false)
    try
        my_rank = MPI.Comm_rank(mpi_subcomm)

        # UniqueID.internal is NTuple{128, Cchar}
        id_bytes = Vector{Cchar}(undef, 128)
        
        # Rank 0 creates UniqueID, broadcasts raw bytes via MPI
        if my_rank == 0
            nccl_id = NCCL.UniqueID()
            id_ref = Ref(nccl_id.internal)
            unsafe_copyto!(pointer(id_bytes),
                           Ptr{Cchar}(pointer_from_objref(id_ref)), 128)
        end

        MPI.Bcast!(id_bytes, mpi_subcomm; root=0)

        # Reconstruct UniqueID on all ranks.
        nccl_internal = ntuple(i -> id_bytes[i], Val(128))
        nccl_id_all = NCCL.UniqueID(nccl_internal)

        nccl_comm = NCCL.Communicator(nranks, my_rank; unique_id=nccl_id_all)
        push!(_PERSISTENT_NCCL_COMMUNICATORS, nccl_comm)  # root it; never GC-finalized mid-collective
        return nccl_comm
    finally
        GC.enable(gc_was_enabled)
    end
end
