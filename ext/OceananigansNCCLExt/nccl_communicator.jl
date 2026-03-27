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

    my_rank = MPI.Comm_rank(mpi_subcomm)

    # Rank 0 creates UniqueID, broadcasts raw bytes via MPI
    if my_rank == 0
        nccl_id = NCCL.UniqueID()
        # UniqueID.internal is NTuple{128, Cchar} (Int8)
        id_bytes = Vector{UInt8}(undef, 128)
        id_ref = Ref(nccl_id.internal)
        unsafe_copyto!(pointer(id_bytes),
                       Ptr{UInt8}(pointer_from_objref(id_ref)), 128)
    else
        id_bytes = Vector{UInt8}(undef, 128)
    end

    MPI.Bcast!(id_bytes, mpi_subcomm; root=0)

    # Reconstruct UniqueID on all ranks
    # ncclUniqueId.internal is NTuple{128, Cchar} (Int8), so reinterpret UInt8→Int8
    nccl_internal = ntuple(i -> reinterpret(Int8, id_bytes[i]), Val(128))
    nccl_id_all = NCCL.UniqueID(nccl_internal)

    return NCCL.Communicator(nranks, my_rank; unique_id=nccl_id_all)
end
