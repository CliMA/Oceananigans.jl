#####
##### NCCL subcommunicator cache for TransposableField transposes
#####
##### The TransposableField creates MPI subcommunicators (comms.xy, comms.yz)
##### via MPI.Comm_split. We create matching NCCL communicators lazily and
##### cache them in a global dictionary keyed by the MPI comm.
#####

const _nccl_subcomm_cache = Dict{MPI.Comm, NCCL.Communicator}()

function _get_nccl_subcomm(mpi_subcomm)
    get!(_nccl_subcomm_cache, mpi_subcomm) do
        create_nccl_comm_from_mpi(mpi_subcomm)
    end
end

# TEMPORARILY DISABLED: NCCL dispatch for transposes
# Falls back to base alltoall_transpose! (CPU staging in nccl_distributed.jl)
# to isolate whether the NCCL transpose or something else causes NHM garbage.
#
# function DC.transpose_y_to_x!(arch::NCCLDistributedArch, pf::DC.TransposableField) ...
# function DC.transpose_x_to_y!(arch::NCCLDistributedArch, pf::DC.TransposableField) ...
# function DC.transpose_z_to_y!(arch::NCCLDistributedArch, pf::DC.TransposableField) ...
# function DC.transpose_y_to_z!(arch::NCCLDistributedArch, pf::DC.TransposableField) ...

# Standalone NCCL transpose implementations

"""
    nccl_alltoall!(buffer, counts, nccl_comm)

Replace MPI Alltoall with NCCL grouped Send/Recv.
No `sync_device!` needed — NCCL ops are GPU-stream-native.
"""
function nccl_alltoall!(buffer, counts, nccl_comm)
    T = eltype(buffer.send)
    count_per_rank = T <: Complex ? 2 * counts[1] : counts[1]
    datatype = NCCL.ncclDataType_t(T)
    NCCL.LibNCCL.ncclAlltoAll(pointer(buffer.send), pointer(buffer.recv),
                               Int32(count_per_rank), datatype,
                               nccl_comm.handle, CUDA.stream())
    return nothing
end

function nccl_transpose_y_to_x!(pf, nccl_comm; comm_stream=nothing)
    DC.pack_buffer_y_to_x!(pf.xybuff, pf.yfield)
    nccl_alltoall_with_stream!(pf.xybuff, pf.counts.xy, nccl_comm, comm_stream)
    DC.unpack_buffer_x_from_y!(pf.xfield, pf.yfield, pf.xybuff)
    return nothing
end

function nccl_transpose_x_to_y!(pf, nccl_comm; comm_stream=nothing)
    DC.pack_buffer_x_to_y!(pf.xybuff, pf.xfield)
    nccl_alltoall_with_stream!(pf.xybuff, pf.counts.xy, nccl_comm, comm_stream)
    DC.unpack_buffer_y_from_x!(pf.yfield, pf.xfield, pf.xybuff)
    return nothing
end

function nccl_transpose_z_to_y!(pf, nccl_comm; comm_stream=nothing)
    DC.pack_buffer_z_to_y!(pf.yzbuff, pf.zfield)
    nccl_alltoall_with_stream!(pf.yzbuff, pf.counts.yz, nccl_comm, comm_stream)
    DC.unpack_buffer_y_from_z!(pf.yfield, pf.zfield, pf.yzbuff)
    return nothing
end

function nccl_transpose_y_to_z!(pf, nccl_comm; comm_stream=nothing)
    DC.pack_buffer_y_to_z!(pf.yzbuff, pf.yfield)
    nccl_alltoall_with_stream!(pf.yzbuff, pf.counts.yz, nccl_comm, comm_stream)
    DC.unpack_buffer_z_from_y!(pf.zfield, pf.yfield, pf.yzbuff)
    return nothing
end

"""
    nccl_alltoall_with_stream!(buffer, counts, nccl_comm, comm_stream)

Run ncclAlltoAll on the default (task-local) CUDA stream. The comm_stream
argument is accepted for API compatibility but ignored — solver transposes
must run on the same stream as pack/unpack/FFT kernels to guarantee ordering.
"""
function nccl_alltoall_with_stream!(buffer, counts, nccl_comm, comm_stream)
    nccl_alltoall!(buffer, counts, nccl_comm)
    return nothing
end
