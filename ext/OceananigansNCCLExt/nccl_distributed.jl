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
MPI.Comm_split(c::NCCLCommunicator, color, key) = MPI.Comm_split(c.mpi, color, key)
MPI.Barrier(c::NCCLCommunicator) = MPI.Barrier(c.mpi)
MPI.Bcast!(buf, c::NCCLCommunicator; kwargs...) = MPI.Bcast!(buf, c.mpi; kwargs...)

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

# Helper to check if an architecture uses NCCL
_uses_nccl(arch) = false
_uses_nccl(arch::Distributed) = arch.communicator isa NCCLCommunicator

#####
##### NCCL halo communication — replaces MPI Isend/Irecv
#####

# Override distributed_fill_halo_event! to skip sync_device! and use NCCL
function DC.distributed_fill_halo_event!(c, kernel!::DistributedFillHalo, bcs, loc,
                                         grid::DC.DistributedGrid, buffers, args...;
                                         async = false, only_local_halos = false, kwargs...)

    only_local_halos && return nothing

    arch = DC.architecture(grid)
    buffer_side = kernel!.side

    if _uses_nccl(arch)
        # NCCL path: no sync_device!, batch sends/recvs in NCCL group
        DC.fill_send_buffers!(c, buffers, grid, buffer_side)
        _nccl_halo_exchange!(kernel!, bcs, arch, buffers)
        DC.recv_from_buffers!(c, buffers, grid, buffer_side)
    else
        # Original MPI path
        DC.fill_send_buffers!(c, buffers, grid, buffer_side)
        DC.sync_device!(arch)
        requests = kernel!(c, bcs..., loc, grid, arch, buffers)
        DC.pool_requests_or_complete_comm!(c, arch, grid, buffers, requests, async, buffer_side)
    end

    return nothing
end

#####
##### NCCL grouped Send/Recv for each halo exchange pattern
#####

function _nccl_halo_exchange!(::DistributedFillHalo{<:WestAndEast}, bcs, arch, buffers)
    west_bc, east_bc = bcs
    nccl_comm = arch.communicator.nccl
    NCCL.groupStart()
    NCCL.Send(buffers.west.send, nccl_comm; dest=west_bc.condition.to)
    NCCL.Recv!(buffers.west.recv, nccl_comm; source=west_bc.condition.to)
    NCCL.Send(buffers.east.send, nccl_comm; dest=east_bc.condition.to)
    NCCL.Recv!(buffers.east.recv, nccl_comm; source=east_bc.condition.to)
    NCCL.groupEnd()
end

function _nccl_halo_exchange!(::DistributedFillHalo{<:SouthAndNorth}, bcs, arch, buffers)
    south_bc, north_bc = bcs
    nccl_comm = arch.communicator.nccl
    NCCL.groupStart()
    NCCL.Send(buffers.south.send, nccl_comm; dest=south_bc.condition.to)
    NCCL.Recv!(buffers.south.recv, nccl_comm; source=south_bc.condition.to)
    NCCL.Send(buffers.north.send, nccl_comm; dest=north_bc.condition.to)
    NCCL.Recv!(buffers.north.recv, nccl_comm; source=north_bc.condition.to)
    NCCL.groupEnd()
end

function _nccl_halo_exchange!(::DistributedFillHalo{<:West}, bcs, arch, buffers)
    bc = bcs[1]
    nccl_comm = arch.communicator.nccl
    NCCL.groupStart()
    NCCL.Send(buffers.west.send, nccl_comm; dest=bc.condition.to)
    NCCL.Recv!(buffers.west.recv, nccl_comm; source=bc.condition.to)
    NCCL.groupEnd()
end

function _nccl_halo_exchange!(::DistributedFillHalo{<:East}, bcs, arch, buffers)
    bc = bcs[1]
    nccl_comm = arch.communicator.nccl
    NCCL.groupStart()
    NCCL.Send(buffers.east.send, nccl_comm; dest=bc.condition.to)
    NCCL.Recv!(buffers.east.recv, nccl_comm; source=bc.condition.to)
    NCCL.groupEnd()
end

function _nccl_halo_exchange!(::DistributedFillHalo{<:South}, bcs, arch, buffers)
    bc = bcs[1]
    nccl_comm = arch.communicator.nccl
    NCCL.groupStart()
    NCCL.Send(buffers.south.send, nccl_comm; dest=bc.condition.to)
    NCCL.Recv!(buffers.south.recv, nccl_comm; source=bc.condition.to)
    NCCL.groupEnd()
end

function _nccl_halo_exchange!(::DistributedFillHalo{<:North}, bcs, arch, buffers)
    bc = bcs[1]
    nccl_comm = arch.communicator.nccl
    NCCL.groupStart()
    NCCL.Send(buffers.north.send, nccl_comm; dest=bc.condition.to)
    NCCL.Recv!(buffers.north.recv, nccl_comm; source=bc.condition.to)
    NCCL.groupEnd()
end

# No vertical communication needed
_nccl_halo_exchange!(::DistributedFillHalo{<:BottomAndTop}, args...) = nothing
_nccl_halo_exchange!(::DistributedFillHalo{<:Bottom}, args...) = nothing
_nccl_halo_exchange!(::DistributedFillHalo{<:Top}, args...) = nothing

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
