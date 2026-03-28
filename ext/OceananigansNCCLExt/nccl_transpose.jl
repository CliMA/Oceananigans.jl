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

# Override the generated transpose functions for NCCLDistributed grids.
# These are called as transpose_y_to_x!(storage) where storage is a TransposableField.
# We detect NCCL by checking the field's architecture communicator.

function DC.transpose_y_to_x!(pf::DC.TransposableField)
    arch = DC.architecture(pf.yfield)
    @debug "transpose_y_to_x! called, arch.communicator = $(typeof(arch.communicator))"
    if arch.communicator isa NCCLCommunicator
        @debug "  → NCCL path"
        nccl_comm = _get_nccl_subcomm(pf.comms.xy)
        nccl_transpose_y_to_x!(pf, nccl_comm)
    else
        @debug "  → MPI fallback"
        # Fall back to original — pack, sync, Alltoall, unpack
        DC.pack_buffer_y_to_x!(pf.xybuff, pf.yfield)
        DC.sync_device!(arch)
        counts = pf.counts.xy
        if allequal(counts)
            MPI.Alltoall!(MPI.UBuffer(pf.xybuff.send, counts[1]),
                          MPI.UBuffer(pf.xybuff.recv, counts[1]),
                          pf.comms.xy)
        else
            MPI.Alltoallv!(MPI.VBuffer(pf.xybuff.send, counts),
                           MPI.VBuffer(pf.xybuff.recv, counts),
                           pf.comms.xy)
        end
        DC.unpack_buffer_x_from_y!(pf.xfield, pf.yfield, pf.xybuff)
    end
    return nothing
end

function DC.transpose_x_to_y!(pf::DC.TransposableField)
    arch = DC.architecture(pf.yfield)  # yfield always carries the original arch
    if arch.communicator isa NCCLCommunicator
        nccl_comm = _get_nccl_subcomm(pf.comms.xy)
        nccl_transpose_x_to_y!(pf, nccl_comm)
    else
        DC.pack_buffer_x_to_y!(pf.xybuff, pf.xfield)
        DC.sync_device!(arch)
        counts = pf.counts.xy
        if all(c -> c == counts[1], counts)
            MPI.Alltoall!(MPI.UBuffer(pf.xybuff.send, counts[1]),
                          MPI.UBuffer(pf.xybuff.recv, counts[1]),
                          pf.comms.xy)
        else
            MPI.Alltoallv!(MPI.VBuffer(pf.xybuff.send, counts),
                           MPI.VBuffer(pf.xybuff.recv, counts),
                           pf.comms.xy)
        end
        DC.unpack_buffer_y_from_x!(pf.yfield, pf.xfield, pf.xybuff)
    end
    return nothing
end

function DC.transpose_z_to_y!(pf::DC.TransposableField)
    arch = DC.architecture(pf.yfield)  # yfield always carries the original arch
    if arch.communicator isa NCCLCommunicator
        nccl_comm = _get_nccl_subcomm(pf.comms.yz)
        nccl_transpose_z_to_y!(pf, nccl_comm)
    else
        DC.pack_buffer_z_to_y!(pf.yzbuff, pf.zfield)
        DC.sync_device!(arch)
        counts = pf.counts.yz
        if all(c -> c == counts[1], counts)
            MPI.Alltoall!(MPI.UBuffer(pf.yzbuff.send, counts[1]),
                          MPI.UBuffer(pf.yzbuff.recv, counts[1]),
                          pf.comms.yz)
        else
            MPI.Alltoallv!(MPI.VBuffer(pf.yzbuff.send, counts),
                           MPI.VBuffer(pf.yzbuff.recv, counts),
                           pf.comms.yz)
        end
        DC.unpack_buffer_y_from_z!(pf.yfield, pf.zfield, pf.yzbuff)
    end
    return nothing
end

function DC.transpose_y_to_z!(pf::DC.TransposableField)
    arch = DC.architecture(pf.yfield)
    if arch.communicator isa NCCLCommunicator
        nccl_comm = _get_nccl_subcomm(pf.comms.yz)
        nccl_transpose_y_to_z!(pf, nccl_comm)
    else
        DC.pack_buffer_y_to_z!(pf.yzbuff, pf.yfield)
        DC.sync_device!(arch)
        counts = pf.counts.yz
        if all(c -> c == counts[1], counts)
            MPI.Alltoall!(MPI.UBuffer(pf.yzbuff.send, counts[1]),
                          MPI.UBuffer(pf.yzbuff.recv, counts[1]),
                          pf.comms.yz)
        else
            MPI.Alltoallv!(MPI.VBuffer(pf.yzbuff.send, counts),
                           MPI.VBuffer(pf.yzbuff.recv, counts),
                           pf.comms.yz)
        end
        DC.unpack_buffer_z_from_y!(pf.zfield, pf.yfield, pf.yzbuff)
    end
    return nothing
end

# Keep existing standalone functions for NCCLDistributedFFTSolver
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
