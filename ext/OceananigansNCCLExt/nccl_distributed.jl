using Oceananigans.Architectures: child_architecture
using Oceananigans.BoundaryConditions: DistributedFillHalo, WestAndEast, SouthAndNorth,
                                       West, East, South, North, BottomAndTop, Bottom, Top

"""
    NCCLCommunicator

Wraps both an NCCL communicator (for GPU-to-GPU data transfer) and the
original MPI communicator (for initialization, tags, reductions).
"""
struct NCCLCommunicator{NC, MC}
    nccl :: NC   # NCCL.Communicator
    mpi  :: MC   # MPI.Comm (still needed for rank info, reductions, etc.)
end

# Forward MPI operations to the inner MPI comm
MPI.Comm_rank(c::NCCLCommunicator) = MPI.Comm_rank(c.mpi)
MPI.Comm_size(c::NCCLCommunicator) = MPI.Comm_size(c.mpi)
MPI.Allreduce!(sendbuf, recvbuf, op, c::NCCLCommunicator) = MPI.Allreduce!(sendbuf, recvbuf, op, c.mpi)
MPI.Allreduce!(buf, op, c::NCCLCommunicator) = MPI.Allreduce!(buf, op, c.mpi)
MPI.Allreduce(buf, op, c::NCCLCommunicator) = MPI.Allreduce(buf, op, c.mpi)
MPI.Comm_split(c::NCCLCommunicator, color, key) = MPI.Comm_split(c.mpi, color, key)
MPI.Comm_split_type(c::NCCLCommunicator, t, key; kwargs...) = MPI.Comm_split_type(c.mpi, t, key; kwargs...)
MPI.Barrier(c::NCCLCommunicator) = MPI.Barrier(c.mpi)
MPI.Bcast!(buf, c::NCCLCommunicator; kwargs...) = MPI.Bcast!(buf, c.mpi; kwargs...)
MPI.Isend(buf, dest, tag, c::NCCLCommunicator) = MPI.Isend(buf, dest, tag, c.mpi)
MPI.Irecv!(buf, src, tag, c::NCCLCommunicator) = MPI.Irecv!(buf, src, tag, c.mpi)

"""
    NCCLDistributed(child_arch = GPU(); partition, kwargs...)

Create a `Distributed` architecture that uses NCCL for GPU-to-GPU communication
(halo exchange, FFT transposes) instead of MPI.

MPI is still used for initialization and rank discovery.

Returns a `Distributed` whose communicator field is an `NCCLCommunicator`.
"""
function NCCLDistributed(child_arch = GPU(); partition = nothing, kwargs...)
    # First create normal MPI-based Distributed
    mpi_arch = Distributed(child_arch; partition, kwargs...)

    # Create NCCL comm mirroring the MPI world comm
    nccl_comm = create_nccl_comm_from_mpi(mpi_arch.communicator)
    nccl_communicator = NCCLCommunicator(nccl_comm, mpi_arch.communicator)

    # Reconstruct Distributed with NCCL communicator
    S = mpi_arch isa DC.SynchronizedDistributed
    return Distributed{S}(mpi_arch.child_architecture,
                          mpi_arch.partition,
                          mpi_arch.ranks,
                          mpi_arch.local_rank,
                          mpi_arch.local_index,
                          mpi_arch.connectivity,
                          nccl_communicator,
                          mpi_arch.mpi_requests,
                          mpi_arch.mpi_tag,
                          mpi_arch.devices)
end

# Type alias for dispatch
const _NCCLArch = Distributed{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:NCCLCommunicator}

# sync_device! is a no-op for NCCLDistributed — NCCL ops are GPU-stream-native
import Oceananigans.Utils: sync_device!
sync_device!(::_NCCLArch) = nothing

#####
##### NCCL halo communication
#####
##### The existing distributed_fill_halo_event! flow calls:
#####   fill_send_buffers! → sync_device!(arch) → kernel!(c, bcs..., grid, arch, buffers) → pool_requests
#####
##### With NCCLDistributed:
#####   - sync_device! is a no-op (defined above)
#####   - The DistributedFillHalo callables use NCCL grouped Send/Recv instead of MPI
#####   - They return nothing (no MPI requests), so pool_requests_or_complete_comm! skips waitall
#####   - recv_from_buffers! is called inside the callable since pool skips it for nothing
#####

# Helper: perform NCCL exchange and recv_from_buffers! in one step
function _nccl_exchange_and_recv!(nccl_comm, send_recv_pairs, c, buffers, grid, side)
    NCCL.groupStart()
    for (send_buf, recv_buf, peer) in send_recv_pairs
        NCCL.Send(send_buf, nccl_comm; dest=peer)
        NCCL.Recv!(recv_buf, nccl_comm; source=peer)
    end
    NCCL.groupEnd()
    DC.recv_from_buffers!(c, buffers, grid, side)
    return nothing  # no MPI requests
end

function (k::DistributedFillHalo{<:WestAndEast})(c, west_bc, east_bc, loc, grid, arch::_NCCLArch, buffers)
    nccl_comm = arch.communicator.nccl
    pairs = ((buffers.west.send, buffers.west.recv, west_bc.condition.to),
             (buffers.east.send, buffers.east.recv, east_bc.condition.to))
    _nccl_exchange_and_recv!(nccl_comm, pairs, c, buffers, grid, k.side)
end

function (k::DistributedFillHalo{<:SouthAndNorth})(c, south_bc, north_bc, loc, grid, arch::_NCCLArch, buffers)
    nccl_comm = arch.communicator.nccl
    pairs = ((buffers.south.send, buffers.south.recv, south_bc.condition.to),
             (buffers.north.send, buffers.north.recv, north_bc.condition.to))
    _nccl_exchange_and_recv!(nccl_comm, pairs, c, buffers, grid, k.side)
end

function (k::DistributedFillHalo{<:West})(c, bc, loc, grid, arch::_NCCLArch, buffers)
    nccl_comm = arch.communicator.nccl
    pairs = ((buffers.west.send, buffers.west.recv, bc.condition.to),)
    _nccl_exchange_and_recv!(nccl_comm, pairs, c, buffers, grid, k.side)
end

function (k::DistributedFillHalo{<:East})(c, bc, loc, grid, arch::_NCCLArch, buffers)
    nccl_comm = arch.communicator.nccl
    pairs = ((buffers.east.send, buffers.east.recv, bc.condition.to),)
    _nccl_exchange_and_recv!(nccl_comm, pairs, c, buffers, grid, k.side)
end

function (k::DistributedFillHalo{<:South})(c, bc, loc, grid, arch::_NCCLArch, buffers)
    nccl_comm = arch.communicator.nccl
    pairs = ((buffers.south.send, buffers.south.recv, bc.condition.to),)
    _nccl_exchange_and_recv!(nccl_comm, pairs, c, buffers, grid, k.side)
end

function (k::DistributedFillHalo{<:North})(c, bc, loc, grid, arch::_NCCLArch, buffers)
    nccl_comm = arch.communicator.nccl
    pairs = ((buffers.north.send, buffers.north.recv, bc.condition.to),)
    _nccl_exchange_and_recv!(nccl_comm, pairs, c, buffers, grid, k.side)
end

#####
##### NCCL corner communication
#####

function DC.fill_corners!(c, connectivity, indices, loc,
                          arch::Distributed{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:NCCLCommunicator},
                          grid, buffers, args...;
                          async=false, only_local_halos=false, kw...)
    only_local_halos && return nothing

    # Skip if no corner neighbors
    isnothing(connectivity.southwest) && isnothing(connectivity.southeast) &&
    isnothing(connectivity.northwest) && isnothing(connectivity.northeast) && return nothing

    DC.fill_send_buffers!(c, buffers, grid, Val(:corners))
    # No sync_device! for NCCL

    nccl_comm = arch.communicator.nccl
    NCCL.groupStart()
    _nccl_corner_send_recv!(nccl_comm, connectivity.southwest, buffers.southwest)
    _nccl_corner_send_recv!(nccl_comm, connectivity.southeast, buffers.southeast)
    _nccl_corner_send_recv!(nccl_comm, connectivity.northwest, buffers.northwest)
    _nccl_corner_send_recv!(nccl_comm, connectivity.northeast, buffers.northeast)
    NCCL.groupEnd()

    DC.recv_from_buffers!(c, buffers, grid, Val(:corners))
    return nothing
end

_nccl_corner_send_recv!(nccl_comm, ::Nothing, buffers) = nothing

function _nccl_corner_send_recv!(nccl_comm, corner_rank, buffers)
    NCCL.Send(buffers.send, nccl_comm; dest=corner_rank)
    NCCL.Recv!(buffers.recv, nccl_comm; source=corner_rank)
end

#####
##### Batched multi-field halo fill
#####
##### Instead of N NCCL groups (one per field per side), we do:
#####   1. Pack all fields' send buffers (N GPU kernels)
#####   2. One NCCL group with all Send/Recv for all fields (1 NCCL kernel)
#####   3. Unpack all fields' recv buffers (N GPU kernels)
#####
##### This reduces NCCL kernel launches from ~N*2 to 1 per batch.
#####

using Oceananigans.Fields: instantiated_location
using Oceananigans.BoundaryConditions: get_boundary_kernels, DistributedFillHalo

const DistributedField = Oceananigans.Fields.Field{<:Any, <:Any, <:Any, <:Any, <:DC.DistributedGrid}

function Oceananigans.BoundaryConditions.fill_halo_regions!(field::DistributedField, args...; kwargs...)
    arch = DC.architecture(field.grid)
    if arch.communicator isa NCCLCommunicator
        _nccl_fill_halo_regions!(field, args...; kwargs...)
    else
        # Fall through to default per-field path
        invoke(Oceananigans.BoundaryConditions.fill_halo_regions!,
               Tuple{Oceananigans.Fields.Field, typeof.(args)...},
               field, args...; kwargs...)
    end
end

function _nccl_fill_halo_regions!(field, args...; only_local_halos=false, kwargs...)
    grid = field.grid
    arch = DC.architecture(grid)
    nccl_comm = arch.communicator.nccl

    c    = field.data
    bcs  = field.boundary_conditions
    idx  = field.indices
    loc  = instantiated_location(field)
    bufs = field.communication_buffers

    kernels!, bc_tuples = get_boundary_kernels(bcs, c, grid, loc, idx)

    if only_local_halos
        # Only fill local (non-communicating) halos
        for task in 1:length(kernels!)
            k = kernels![task]
            k isa DistributedFillHalo && continue
            Oceananigans.BoundaryConditions.fill_halo_event!(c, k, bc_tuples[task], loc, grid, args...; kwargs...)
        end
        return nothing
    end

    # Phase 1: Pack send buffers for all distributed sides
    for task in 1:length(kernels!)
        k = kernels![task]
        k isa DistributedFillHalo || continue
        DC.fill_send_buffers!(c, bufs, grid, k.side)
    end

    # Phase 2: One NCCL group for all distributed sends/recvs
    NCCL.groupStart()
    for task in 1:length(kernels!)
        k = kernels![task]
        k isa DistributedFillHalo || continue
        _enqueue_nccl_send_recv!(k, bc_tuples[task], nccl_comm, bufs)
    end
    # Also corners
    conn = arch.connectivity
    if !(isnothing(conn.southwest) && isnothing(conn.southeast) &&
         isnothing(conn.northwest) && isnothing(conn.northeast))
        DC.fill_send_buffers!(c, bufs, grid, Val(:corners))
        _nccl_corner_send_recv!(nccl_comm, conn.southwest, bufs.southwest)
        _nccl_corner_send_recv!(nccl_comm, conn.southeast, bufs.southeast)
        _nccl_corner_send_recv!(nccl_comm, conn.northwest, bufs.northwest)
        _nccl_corner_send_recv!(nccl_comm, conn.northeast, bufs.northeast)
    end
    NCCL.groupEnd()

    # Phase 3: Unpack recv buffers + fill local halos
    for task in 1:length(kernels!)
        k = kernels![task]
        if k isa DistributedFillHalo
            DC.recv_from_buffers!(c, bufs, grid, k.side)
        else
            Oceananigans.BoundaryConditions.fill_halo_event!(c, k, bc_tuples[task], loc, grid, args...; kwargs...)
        end
    end
    if !(isnothing(conn.southwest) && isnothing(conn.southeast) &&
         isnothing(conn.northwest) && isnothing(conn.northeast))
        DC.recv_from_buffers!(c, bufs, grid, Val(:corners))
    end

    return nothing
end

# Enqueue Send/Recv for one side (called inside groupStart/groupEnd)
function _enqueue_nccl_send_recv!(::DistributedFillHalo{<:WestAndEast}, bcs, nccl_comm, bufs)
    NCCL.Send(bufs.west.send, nccl_comm; dest=bcs[1].condition.to)
    NCCL.Recv!(bufs.west.recv, nccl_comm; source=bcs[1].condition.to)
    NCCL.Send(bufs.east.send, nccl_comm; dest=bcs[2].condition.to)
    NCCL.Recv!(bufs.east.recv, nccl_comm; source=bcs[2].condition.to)
end
function _enqueue_nccl_send_recv!(::DistributedFillHalo{<:SouthAndNorth}, bcs, nccl_comm, bufs)
    NCCL.Send(bufs.south.send, nccl_comm; dest=bcs[1].condition.to)
    NCCL.Recv!(bufs.south.recv, nccl_comm; source=bcs[1].condition.to)
    NCCL.Send(bufs.north.send, nccl_comm; dest=bcs[2].condition.to)
    NCCL.Recv!(bufs.north.recv, nccl_comm; source=bcs[2].condition.to)
end
function _enqueue_nccl_send_recv!(::DistributedFillHalo{<:West}, bcs, nccl_comm, bufs)
    NCCL.Send(bufs.west.send, nccl_comm; dest=bcs[1].condition.to)
    NCCL.Recv!(bufs.west.recv, nccl_comm; source=bcs[1].condition.to)
end
function _enqueue_nccl_send_recv!(::DistributedFillHalo{<:East}, bcs, nccl_comm, bufs)
    NCCL.Send(bufs.east.send, nccl_comm; dest=bcs[1].condition.to)
    NCCL.Recv!(bufs.east.recv, nccl_comm; source=bcs[1].condition.to)
end
function _enqueue_nccl_send_recv!(::DistributedFillHalo{<:South}, bcs, nccl_comm, bufs)
    NCCL.Send(bufs.south.send, nccl_comm; dest=bcs[1].condition.to)
    NCCL.Recv!(bufs.south.recv, nccl_comm; source=bcs[1].condition.to)
end
function _enqueue_nccl_send_recv!(::DistributedFillHalo{<:North}, bcs, nccl_comm, bufs)
    NCCL.Send(bufs.north.send, nccl_comm; dest=bcs[1].condition.to)
    NCCL.Recv!(bufs.north.recv, nccl_comm; source=bcs[1].condition.to)
end
_enqueue_nccl_send_recv!(::DistributedFillHalo{<:BottomAndTop}, args...) = nothing
_enqueue_nccl_send_recv!(::DistributedFillHalo{<:Bottom}, args...) = nothing
_enqueue_nccl_send_recv!(::DistributedFillHalo{<:Top}, args...) = nothing
