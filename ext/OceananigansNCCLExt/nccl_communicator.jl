"""
    create_nccl_comm_from_mpi(mpi_subcomm)

Create an NCCL communicator that mirrors the given MPI subcommunicator.

Returns `nothing` if the subcommunicator has only 1 rank (no communication needed).
"""
function create_nccl_comm_from_mpi(mpi_subcomm)
    MPI.Comm_size(mpi_subcomm) == 1 && return nothing
    return NCCL.Communicator(mpi_subcomm)
end
