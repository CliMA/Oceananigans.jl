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

function DC.transpose_y_to_x!(arch::NCCLDistributedArchitecture, pf::DC.TransposableField)
    nccl_comm = _get_nccl_subcomm(pf.comms.xy)
    comm_stream = arch.communicator.comm_stream
    nccl_transpose_y_to_x!(pf, nccl_comm, comm_stream)
    return nothing
end

function DC.transpose_x_to_y!(arch::NCCLDistributedArchitecture, pf::DC.TransposableField)
    nccl_comm = _get_nccl_subcomm(pf.comms.xy)
    comm_stream = arch.communicator.comm_stream
    nccl_transpose_x_to_y!(pf, nccl_comm, comm_stream)
    return nothing
end

function DC.transpose_z_to_y!(arch::NCCLDistributedArchitecture, pf::DC.TransposableField)
    nccl_comm = _get_nccl_subcomm(pf.comms.yz)
    comm_stream = arch.communicator.comm_stream
    nccl_transpose_z_to_y!(pf, nccl_comm, comm_stream)
    return nothing
end

function DC.transpose_y_to_z!(arch::NCCLDistributedArchitecture, pf::DC.TransposableField)
    nccl_comm = _get_nccl_subcomm(pf.comms.yz)
    comm_stream = arch.communicator.comm_stream
    nccl_transpose_y_to_z!(pf, nccl_comm, comm_stream)
    return nothing
end

# Standalone NCCL transpose implementations

"""
    nccl_alltoall!(buffer, counts, nccl_comm, comm_stream)

Pipelined alltoall: local copy on default stream, remote transfers on
comm_stream. The caller can do work on the default stream while remote
data is in flight, then call `wait_for_nccl!(comm_stream)` before
accessing remote data.

If comm_stream is nothing, runs everything on the default stream (no overlap).
"""
function nccl_alltoall!(buffer, counts, nccl_comm, comm_stream=nothing)
    my_rank = NCCL.rank(nccl_comm)
    nranks = NCCL.size(nccl_comm)
    count = counts[1]

    # Local copy on default stream (available immediately)
    local_offset = my_rank * count
    copyto!(buffer.recv, local_offset + 1, buffer.send, local_offset + 1, count)

    nranks == 1 && return nothing

    T = eltype(buffer.send)
    nccl_count = T <: Complex ? 2 * count : count
    datatype = NCCL.ncclDataType_t(T)

    if comm_stream !== nothing
        # Pipelined: remote transfers on comm_stream
        # Make comm_stream wait for pack kernel on default stream
        event = CUDA.CuEvent()
        CUDA.record(event)
        CUDA.cuStreamWaitEvent(comm_stream, event, UInt32(0))

        NCCL.groupStart()
        for r in 0:(nranks - 1)
            r == my_rank && continue
            send_ptr = pointer(buffer.send, r * count + 1)
            recv_ptr = pointer(buffer.recv, r * count + 1)
            NCCL.LibNCCL.ncclSend(send_ptr, Int32(nccl_count), datatype, Int32(r), nccl_comm.handle, comm_stream)
            NCCL.LibNCCL.ncclRecv(recv_ptr, Int32(nccl_count), datatype, Int32(r), nccl_comm.handle, comm_stream)
        end
        NCCL.groupEnd()
        # NOTE: caller must call wait_for_nccl!(comm_stream) before accessing remote data
    else
        # Non-pipelined: everything on default stream
        stream = CUDA.stream()
        NCCL.groupStart()
        for r in 0:(nranks - 1)
            r == my_rank && continue
            send_ptr = pointer(buffer.send, r * count + 1)
            recv_ptr = pointer(buffer.recv, r * count + 1)
            NCCL.LibNCCL.ncclSend(send_ptr, Int32(nccl_count), datatype, Int32(r), nccl_comm.handle, stream)
            NCCL.LibNCCL.ncclRecv(recv_ptr, Int32(nccl_count), datatype, Int32(r), nccl_comm.handle, stream)
        end
        NCCL.groupEnd()
    end

    return nothing
end

"""
    wait_for_nccl!(comm_stream)

Make the default CUDA stream wait for comm_stream to complete.
Call this before accessing remote data after a pipelined nccl_alltoall!.
"""
function wait_for_nccl!(comm_stream)
    event = CUDA.CuEvent()
    CUDA.record(event, comm_stream)
    CUDA.cuStreamWaitEvent(CUDA.stream(), event, UInt32(0))
    return nothing
end

# Pipelined transpose: pack on default stream, NCCL on comm_stream,
# wait for comm_stream, unpack on default stream.

function nccl_transpose_y_to_x!(pf, nccl_comm, comm_stream=nothing)
    DC.pack_buffer_y_to_x!(pf.xybuff, pf.yfield)
    nccl_alltoall!(pf.xybuff, pf.counts.xy, nccl_comm, comm_stream)
    comm_stream !== nothing && wait_for_nccl!(comm_stream)
    DC.unpack_buffer_x_from_y!(pf.xfield, pf.yfield, pf.xybuff)
    return nothing
end

function nccl_transpose_x_to_y!(pf, nccl_comm, comm_stream=nothing)
    DC.pack_buffer_x_to_y!(pf.xybuff, pf.xfield)
    nccl_alltoall!(pf.xybuff, pf.counts.xy, nccl_comm, comm_stream)
    comm_stream !== nothing && wait_for_nccl!(comm_stream)
    DC.unpack_buffer_y_from_x!(pf.yfield, pf.xfield, pf.xybuff)
    return nothing
end

function nccl_transpose_z_to_y!(pf, nccl_comm, comm_stream=nothing)
    DC.pack_buffer_z_to_y!(pf.yzbuff, pf.zfield)
    nccl_alltoall!(pf.yzbuff, pf.counts.yz, nccl_comm, comm_stream)
    comm_stream !== nothing && wait_for_nccl!(comm_stream)
    DC.unpack_buffer_y_from_z!(pf.yfield, pf.zfield, pf.yzbuff)
    return nothing
end

function nccl_transpose_y_to_z!(pf, nccl_comm, comm_stream=nothing)
    DC.pack_buffer_y_to_z!(pf.yzbuff, pf.yfield)
    nccl_alltoall!(pf.yzbuff, pf.counts.yz, nccl_comm, comm_stream)
    comm_stream !== nothing && wait_for_nccl!(comm_stream)
    DC.unpack_buffer_z_from_y!(pf.zfield, pf.yfield, pf.yzbuff)
    return nothing
end
