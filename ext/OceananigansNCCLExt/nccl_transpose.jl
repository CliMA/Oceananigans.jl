"""
    nccl_alltoall!(buffer, counts, nccl_comm)

Replace MPI Alltoall with NCCL grouped Send/Recv.

The key advantage: NCCL operations are GPU-stream-native.
No `sync_device!` / `CUDA.synchronize()` is needed before calling this —
the NCCL ops are enqueued on the same CUDA stream as the preceding
pack kernel, so hardware stream ordering guarantees correctness.
"""
function nccl_alltoall!(buffer, counts, nccl_comm)
    nranks = NCCL.size(nccl_comm)
    count_per_rank = counts[1]  # equal-size chunks for uniform partition

    send = buffer.send
    recv = buffer.recv

    NCCL.groupStart()
    for r in 0:(nranks - 1)
        offset = r * count_per_rank
        send_view = view(send, (offset + 1):(offset + count_per_rank))
        recv_view = view(recv, (offset + 1):(offset + count_per_rank))
        NCCL.Send(send_view, nccl_comm; dest=r)
        NCCL.Recv!(recv_view, nccl_comm; source=r)
    end
    NCCL.groupEnd()

    return nothing
end

"""
    nccl_transpose_y_to_x!(storage, nccl_comm)

NCCL-based transpose from y-local to x-local configuration.
Replaces the MPI path: pack → sync_device! → Alltoall → unpack
with:                   pack → NCCL grouped Send/Recv → unpack
(no sync_device! needed).
"""
function nccl_transpose_y_to_x!(storage, nccl_comm)
    DC.pack_buffer_y_to_x!(storage.xybuff, storage.yfield)
    # No sync_device! — NCCL is stream-ordered
    nccl_alltoall!(storage.xybuff, storage.counts.xy, nccl_comm)
    DC.unpack_buffer_x_from_y!(storage.xfield, storage.yfield, storage.xybuff)
    return nothing
end

"""
    nccl_transpose_x_to_y!(storage, nccl_comm)

NCCL-based transpose from x-local to y-local configuration.
"""
function nccl_transpose_x_to_y!(storage, nccl_comm)
    DC.pack_buffer_x_to_y!(storage.xybuff, storage.xfield)
    # No sync_device! — NCCL is stream-ordered
    nccl_alltoall!(storage.xybuff, storage.counts.xy, nccl_comm)
    DC.unpack_buffer_y_from_x!(storage.yfield, storage.xfield, storage.xybuff)
    return nothing
end

"""
    nccl_transpose_z_to_y!(storage, nccl_comm)

NCCL-based transpose from z-local to y-local configuration.
"""
function nccl_transpose_z_to_y!(storage, nccl_comm)
    DC.pack_buffer_z_to_y!(storage.yzbuff, storage.zfield)
    nccl_alltoall!(storage.yzbuff, storage.counts.yz, nccl_comm)
    DC.unpack_buffer_y_from_z!(storage.yfield, storage.zfield, storage.yzbuff)
    return nothing
end

"""
    nccl_transpose_y_to_z!(storage, nccl_comm)

NCCL-based transpose from y-local to z-local configuration.
"""
function nccl_transpose_y_to_z!(storage, nccl_comm)
    DC.pack_buffer_y_to_z!(storage.yzbuff, storage.yfield)
    nccl_alltoall!(storage.yzbuff, storage.counts.yz, nccl_comm)
    DC.unpack_buffer_z_from_y!(storage.zfield, storage.yfield, storage.yzbuff)
    return nothing
end
