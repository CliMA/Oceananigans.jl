using Oceananigans.Architectures: child_architecture
using Oceananigans.Fields: instantiated_location
using Oceananigans.BoundaryConditions: DistributedFillHalo, WestAndEast, SouthAndNorth,
                                       West, East, South, North, BottomAndTop, Bottom, Top,
                                       get_boundary_kernels

import Oceananigans.Utils: sync_device!

#####
##### NCCLCommunicator
#####

struct NCCLCommunicator{NC, MC}
    nccl :: NC   # NCCL.Communicator
    mpi  :: MC   # MPI.Comm
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

#####
##### Type aliases for dispatch
#####

const NCCLDistributedArch  = Distributed{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:NCCLCommunicator}
const NCCLDistributedGrid  = Oceananigans.Grids.AbstractUnderlyingGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:NCCLDistributedArch}
const NCCLDistributedField = Oceananigans.Fields.Field{<:Any, <:Any, <:Any, <:Any, <:NCCLDistributedGrid}

#####
##### NCCLDistributed constructor
#####

function NCCLDistributed(child_arch = GPU(); partition = nothing, kwargs...)
    mpi_arch = Distributed(child_arch; partition, kwargs...)
    nccl_comm = create_nccl_comm_from_mpi(mpi_arch.communicator)
    nccl_communicator = NCCLCommunicator(nccl_comm, mpi_arch.communicator)

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

#####
##### sync_device! no-op for NCCL (stream-native)
#####

sync_device!(::NCCLDistributedArch) = nothing

#####
##### DistributedFillHalo callables — NCCL grouped Send/Recv
#####

function nccl_exchange_and_recv!(nccl_comm, send_recv_pairs, c, buffers, grid, side)
    NCCL.groupStart()
    for (send_buf, recv_buf, peer) in send_recv_pairs
        NCCL.Send(send_buf, nccl_comm; dest=peer)
        NCCL.Recv!(recv_buf, nccl_comm; source=peer)
    end
    NCCL.groupEnd()
    DC.recv_from_buffers!(c, buffers, grid, side)
    return nothing
end

function (k::DistributedFillHalo{<:WestAndEast})(c, west_bc, east_bc, loc, grid, arch::NCCLDistributedArch, buffers)
    pairs = ((buffers.west.send, buffers.west.recv, west_bc.condition.to),
             (buffers.east.send, buffers.east.recv, east_bc.condition.to))
    return nccl_exchange_and_recv!(arch.communicator.nccl, pairs, c, buffers, grid, k.side)
end

function (k::DistributedFillHalo{<:SouthAndNorth})(c, south_bc, north_bc, loc, grid, arch::NCCLDistributedArch, buffers)
    pairs = ((buffers.south.send, buffers.south.recv, south_bc.condition.to),
             (buffers.north.send, buffers.north.recv, north_bc.condition.to))
    return nccl_exchange_and_recv!(arch.communicator.nccl, pairs, c, buffers, grid, k.side)
end

for side in (:West, :East, :South, :North)
    side_sym = Symbol(lowercase(String(side)))
    @eval function (k::DistributedFillHalo{<:$side})(c, bc, loc, grid, arch::NCCLDistributedArch, buffers)
        pairs = ((buffers.$side_sym.send, buffers.$side_sym.recv, bc.condition.to),)
        return nccl_exchange_and_recv!(arch.communicator.nccl, pairs, c, buffers, grid, k.side)
    end
end

#####
##### NCCL corner communication
#####

function DC.fill_corners!(c, connectivity, indices, loc, arch::NCCLDistributedArch,
                          grid, buffers, args...; async=false, only_local_halos=false, kw...)
    only_local_halos && return nothing

    isnothing(connectivity.southwest) && isnothing(connectivity.southeast) &&
    isnothing(connectivity.northwest) && isnothing(connectivity.northeast) && return nothing

    DC.fill_send_buffers!(c, buffers, grid, Val(:corners))

    nccl_comm = arch.communicator.nccl
    NCCL.groupStart()
    nccl_corner_send_recv!(nccl_comm, connectivity.southwest, buffers.southwest)
    nccl_corner_send_recv!(nccl_comm, connectivity.southeast, buffers.southeast)
    nccl_corner_send_recv!(nccl_comm, connectivity.northwest, buffers.northwest)
    nccl_corner_send_recv!(nccl_comm, connectivity.northeast, buffers.northeast)
    NCCL.groupEnd()

    DC.recv_from_buffers!(c, buffers, grid, Val(:corners))
    return nothing
end

nccl_corner_send_recv!(nccl_comm, ::Nothing, buffers) = nothing

function nccl_corner_send_recv!(nccl_comm, corner_rank, buffers)
    NCCL.Send(buffers.send, nccl_comm; dest=corner_rank)
    NCCL.Recv!(buffers.recv, nccl_comm; source=corner_rank)
    return nothing
end

#####
##### Batched multi-field halo fill
#####
##### Pack all fields → one NCCL group for all Send/Recv → unpack all fields.
##### No extra memory needed — each field uses its own existing buffers.
#####

using Oceananigans.Fields: instantiated_location

function Oceananigans.BoundaryConditions.fill_halo_regions!(field::NCCLDistributedField, args...; kwargs...)
    return nccl_fill_halo_regions!((field,), args...; kwargs...)
end

function nccl_fill_halo_regions!(fields, args...; only_local_halos=false, async=false, kwargs...)
    isempty(fields) && return nothing

    grid = first(fields).grid
    arch = DC.architecture(grid)
    nccl_comm = arch.communicator.nccl
    conn = arch.connectivity
    has_corners = !(isnothing(conn.southwest) && isnothing(conn.southeast) &&
                    isnothing(conn.northwest) && isnothing(conn.northeast))

    # Pre-compute kernel info for all fields
    field_infos = map(fields) do field
        c    = field.data
        bcs  = field.boundary_conditions
        idx  = field.indices
        loc  = instantiated_location(field)
        bufs = field.communication_buffers
        ks, bts = get_boundary_kernels(bcs, c, grid, loc, idx)
        (; c, loc, bufs, kernels=ks, bc_tuples=bts)
    end

    if only_local_halos
        for info in field_infos
            for task in 1:length(info.kernels)
                k = info.kernels[task]
                k isa DistributedFillHalo && continue
                Oceananigans.BoundaryConditions.fill_halo_event!(info.c, k, info.bc_tuples[task],
                    info.loc, grid, args...; kwargs...)
            end
        end
        return nothing
    end

    # Phase 1: Pack ALL fields' send buffers
    for info in field_infos
        for task in 1:length(info.kernels)
            k = info.kernels[task]
            k isa DistributedFillHalo || continue
            DC.fill_send_buffers!(info.c, info.bufs, grid, k.side)
        end
        if has_corners
            DC.fill_send_buffers!(info.c, info.bufs, grid, Val(:corners))
        end
    end

    # Phase 2: ONE NCCL group for ALL fields
    NCCL.groupStart()
    for info in field_infos
        for task in 1:length(info.kernels)
            k = info.kernels[task]
            k isa DistributedFillHalo || continue
            enqueue_nccl_send_recv!(k, info.bc_tuples[task], nccl_comm, info.bufs)
        end
        if has_corners
            nccl_corner_send_recv!(nccl_comm, conn.southwest, info.bufs.southwest)
            nccl_corner_send_recv!(nccl_comm, conn.southeast, info.bufs.southeast)
            nccl_corner_send_recv!(nccl_comm, conn.northwest, info.bufs.northwest)
            nccl_corner_send_recv!(nccl_comm, conn.northeast, info.bufs.northeast)
        end
    end
    NCCL.groupEnd()

    # Phase 3: Unpack ALL fields + fill local halos
    for info in field_infos
        for task in 1:length(info.kernels)
            k = info.kernels[task]
            if k isa DistributedFillHalo
                DC.recv_from_buffers!(info.c, info.bufs, grid, k.side)
            else
                Oceananigans.BoundaryConditions.fill_halo_event!(info.c, k, info.bc_tuples[task],
                    info.loc, grid, args...; kwargs...)
            end
        end
        if has_corners
            DC.recv_from_buffers!(info.c, info.bufs, grid, Val(:corners))
        end
    end

    return nothing
end

#####
##### Enqueue NCCL Send/Recv for each halo side (called inside groupStart/groupEnd)
#####

function enqueue_nccl_send_recv!(::DistributedFillHalo{<:WestAndEast}, bcs, nccl_comm, bufs)
    NCCL.Send(bufs.west.send, nccl_comm; dest=bcs[1].condition.to)
    NCCL.Recv!(bufs.west.recv, nccl_comm; source=bcs[1].condition.to)
    NCCL.Send(bufs.east.send, nccl_comm; dest=bcs[2].condition.to)
    NCCL.Recv!(bufs.east.recv, nccl_comm; source=bcs[2].condition.to)
    return nothing
end

function enqueue_nccl_send_recv!(::DistributedFillHalo{<:SouthAndNorth}, bcs, nccl_comm, bufs)
    NCCL.Send(bufs.south.send, nccl_comm; dest=bcs[1].condition.to)
    NCCL.Recv!(bufs.south.recv, nccl_comm; source=bcs[1].condition.to)
    NCCL.Send(bufs.north.send, nccl_comm; dest=bcs[2].condition.to)
    NCCL.Recv!(bufs.north.recv, nccl_comm; source=bcs[2].condition.to)
    return nothing
end

for side in (:West, :East, :South, :North)
    side_sym = Symbol(lowercase(String(side)))
    @eval function enqueue_nccl_send_recv!(::DistributedFillHalo{<:$side}, bcs, nccl_comm, bufs)
        NCCL.Send(bufs.$side_sym.send, nccl_comm; dest=bcs[1].condition.to)
        NCCL.Recv!(bufs.$side_sym.recv, nccl_comm; source=bcs[1].condition.to)
        return nothing
    end
end

enqueue_nccl_send_recv!(::DistributedFillHalo{<:BottomAndTop}, args...) = nothing
enqueue_nccl_send_recv!(::DistributedFillHalo{<:Bottom}, args...) = nothing
enqueue_nccl_send_recv!(::DistributedFillHalo{<:Top}, args...) = nothing
