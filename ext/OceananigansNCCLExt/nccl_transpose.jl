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

function DC.transpose_y_to_x!(arch::NCCLDistributedArch, pf::DC.TransposableField)
    nccl_comm = _get_nccl_subcomm(pf.comms.xy)
    nccl_transpose_y_to_x!(pf, nccl_comm)
    return nothing
end

function DC.transpose_x_to_y!(arch::NCCLDistributedArch, pf::DC.TransposableField)
    nccl_comm = _get_nccl_subcomm(pf.comms.xy)
    nccl_transpose_x_to_y!(pf, nccl_comm)
    return nothing
end

function DC.transpose_z_to_y!(arch::NCCLDistributedArch, pf::DC.TransposableField)
    nccl_comm = _get_nccl_subcomm(pf.comms.yz)
    nccl_transpose_z_to_y!(pf, nccl_comm)
    return nothing
end

function DC.transpose_y_to_z!(arch::NCCLDistributedArch, pf::DC.TransposableField)
    nccl_comm = _get_nccl_subcomm(pf.comms.yz)
    nccl_transpose_y_to_z!(pf, nccl_comm)
    return nothing
end

# Standalone NCCL transpose implementations

"""
    nccl_alltoall!(buffer, counts, nccl_comm)

ncclAlltoAll with local copy bypass: the self-to-self chunk is copied
directly on the GPU without going through NCCL, and remote chunks use
grouped NCCL Send/Recv. Complex types have their count doubled since
NCCL treats ComplexF32 as 2× Float32.
"""
function nccl_alltoall!(buffer, counts, nccl_comm)
    my_rank = NCCL.rank(nccl_comm)
    nranks = NCCL.size(nccl_comm)
    count = counts[1]

    # Local copy: skip NCCL for self-to-self chunk
    local_offset = my_rank * count
    copyto!(buffer.recv, local_offset + 1, buffer.send, local_offset + 1, count)

    # Remote transfers via NCCL
    nranks == 1 && return nothing

    T = eltype(buffer.send)
    nccl_count = T <: Complex ? 2 * count : count
    datatype = NCCL.ncclDataType_t(T)
    stream = CUDA.stream()

    NCCL.groupStart()
    for r in 0:(nranks - 1)
        r == my_rank && continue
        send_offset = r * count
        recv_offset = r * count
        send_ptr = pointer(buffer.send, send_offset + 1)
        recv_ptr = pointer(buffer.recv, recv_offset + 1)
        NCCL.LibNCCL.ncclSend(send_ptr, Int32(nccl_count), datatype, Int32(r), nccl_comm.handle, stream)
        NCCL.LibNCCL.ncclRecv(recv_ptr, Int32(nccl_count), datatype, Int32(r), nccl_comm.handle, stream)
    end
    NCCL.groupEnd()

    return nothing
end

function nccl_transpose_y_to_x!(pf, nccl_comm)
    DC.pack_buffer_y_to_x!(pf.xybuff, pf.yfield)
    nccl_alltoall!(pf.xybuff, pf.counts.xy, nccl_comm)
    DC.unpack_buffer_x_from_y!(pf.xfield, pf.yfield, pf.xybuff)
    return nothing
end

function nccl_transpose_x_to_y!(pf, nccl_comm)
    DC.pack_buffer_x_to_y!(pf.xybuff, pf.xfield)
    nccl_alltoall!(pf.xybuff, pf.counts.xy, nccl_comm)
    DC.unpack_buffer_y_from_x!(pf.yfield, pf.xfield, pf.xybuff)
    return nothing
end

function nccl_transpose_z_to_y!(pf, nccl_comm)
    DC.pack_buffer_z_to_y!(pf.yzbuff, pf.zfield)
    nccl_alltoall!(pf.yzbuff, pf.counts.yz, nccl_comm)
    DC.unpack_buffer_y_from_z!(pf.yfield, pf.zfield, pf.yzbuff)
    return nothing
end

function nccl_transpose_y_to_z!(pf, nccl_comm)
    DC.pack_buffer_y_to_z!(pf.yzbuff, pf.yfield)
    nccl_alltoall!(pf.yzbuff, pf.counts.yz, nccl_comm)
    DC.unpack_buffer_z_from_y!(pf.zfield, pf.yfield, pf.yzbuff)
    return nothing
end
