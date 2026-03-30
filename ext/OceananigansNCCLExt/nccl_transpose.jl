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

# Dispatch on NCCLDistributedArch — proper extension, no method overwriting.
# The base code calls transpose_y_to_x!(arch, pf) which dispatches here.

function DC.transpose_y_to_x!(arch::NCCLDistributedArch, pf::DC.TransposableField)
    nccl_comm = _get_nccl_subcomm(pf.comms.xy)
    nccl_transpose_y_to_x!(pf, nccl_comm; comm_stream=arch.communicator.comm_stream)
    return nothing
end

function DC.transpose_x_to_y!(arch::NCCLDistributedArch, pf::DC.TransposableField)
    nccl_comm = _get_nccl_subcomm(pf.comms.xy)
    nccl_transpose_x_to_y!(pf, nccl_comm; comm_stream=arch.communicator.comm_stream)
    return nothing
end

function DC.transpose_z_to_y!(arch::NCCLDistributedArch, pf::DC.TransposableField)
    nccl_comm = _get_nccl_subcomm(pf.comms.yz)
    nccl_transpose_z_to_y!(pf, nccl_comm; comm_stream=arch.communicator.comm_stream)
    return nothing
end

function DC.transpose_y_to_z!(arch::NCCLDistributedArch, pf::DC.TransposableField)
    nccl_comm = _get_nccl_subcomm(pf.comms.yz)
    nccl_transpose_y_to_z!(pf, nccl_comm; comm_stream=arch.communicator.comm_stream)
    return nothing
end

# Standalone NCCL transpose implementations

"""
    nccl_alltoall!(buffer, counts, nccl_comm)

Replace MPI Alltoall with NCCL grouped Send/Recv.
No `sync_device!` needed — NCCL ops are GPU-stream-native.
"""
function nccl_alltoall!(buffer, counts, nccl_comm; stream_kw...)
    nranks = NCCL.size(nccl_comm)
    count_per_rank = counts[1]

    send = buffer.send
    recv = buffer.recv

    NCCL.groupStart()
    for r in 0:(nranks - 1)
        offset = r * count_per_rank
        send_view = view(send, (offset + 1):(offset + count_per_rank))
        recv_view = view(recv, (offset + 1):(offset + count_per_rank))
        NCCL.Send(send_view, nccl_comm; dest=r, stream_kw...)
        NCCL.Recv!(recv_view, nccl_comm; source=r, stream_kw...)
    end
    NCCL.groupEnd()

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

NCCL alltoall on comm_stream if available, with event-based synchronization.
"""
function nccl_alltoall_with_stream!(buffer, counts, nccl_comm, comm_stream)
    if comm_stream !== nothing
        event = CUDA.CuEvent()
        CUDA.record(event)
        CUDA.cuStreamWaitEvent(comm_stream, event, UInt32(0))

        nccl_alltoall!(buffer, counts, nccl_comm; stream=comm_stream)

        CUDA.record(event, comm_stream)
        CUDA.cuStreamWaitEvent(CUDA.stream(), event, UInt32(0))
    else
        nccl_alltoall!(buffer, counts, nccl_comm)
    end
    return nothing
end
