using Oceananigans.Architectures: child_architecture
using Oceananigans.Fields: instantiated_location
using Oceananigans.BoundaryConditions: DistributedFillHalo, WestAndEast, SouthAndNorth,
                                       West, East, South, North, BottomAndTop, Bottom, Top,
                                       get_boundary_kernels

import Oceananigans.Utils: sync_device!
import Oceananigans.DistributedComputations: synchronize_communication!

#####
##### NCCLCommunicator
#####

struct NCCLCommunicator{NC, MC, CS, EV}
    nccl        :: NC   # NCCL.Communicator
    mpi         :: MC   # MPI.Comm
    comm_stream :: CS   # Dedicated CUDA stream for async NCCL ops
    sync_event  :: EV   # CUDA event for cross-stream synchronization
end

NCCLCommunicator(nccl, mpi) = NCCLCommunicator(nccl, mpi, nothing, nothing)

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
    comm_stream = CUDA.CuStream(; flags=CUDA.STREAM_NON_BLOCKING)
    sync_event = CUDA.CuEvent()
    nccl_communicator = NCCLCommunicator(nccl_comm, mpi_arch.communicator, comm_stream, sync_event)

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
##### distributed_fill_halo_event! for NCCLDistributedGrid
#####
##### Extends the base method (dispatches on more specific NCCLDistributedGrid).
#####
##### Sync mode (async=false):  pack → NCCL on default stream → unpack
##### Async mode (async=true):  pack → NCCL on comm_stream → return (defer unpack)
#####   Interior computation proceeds on default stream while NCCL transfers on comm_stream.
#####   synchronize_communication!() later waits for comm_stream and unpacks.
#####

# Storage for pending async unpacks: (data, buffers, grid, side) tuples
const pending_unpacks = Vector{Any}()

function DC.distributed_fill_halo_event!(c, kernel!::DistributedFillHalo, bcs, loc,
                                         grid::NCCLDistributedGrid, buffers, args...;
                                         async = false, only_local_halos = false, kwargs...)
    only_local_halos && return nothing

    arch = DC.architecture(grid)
    communicator = arch.communicator
    nccl_comm = communicator.nccl
    buffer_side = kernel!.side

    # Pack send buffers (GPU broadcast kernel)
    DC.fill_send_buffers!(c, buffers, grid, buffer_side)

    if async && communicator.comm_stream !== nothing
        # Async: NCCL on comm_stream, defer unpack
        CUDA.record(communicator.sync_event)
        CUDA.cuStreamWaitEvent(communicator.comm_stream, communicator.sync_event, UInt32(0))

        NCCL.groupStart()
        enqueue_nccl_send_recv!(kernel!, bcs, nccl_comm, buffers; stream=communicator.comm_stream)
        NCCL.groupEnd()

        CUDA.record(communicator.sync_event, communicator.comm_stream)
        push!(pending_unpacks, (; c, buffers, grid, side=buffer_side))
        return nothing
    end

    # Sync with comm_stream
    if communicator.comm_stream !== nothing
        CUDA.record(communicator.sync_event)
        CUDA.cuStreamWaitEvent(communicator.comm_stream, communicator.sync_event, UInt32(0))

        NCCL.groupStart()
        enqueue_nccl_send_recv!(kernel!, bcs, nccl_comm, buffers; stream=communicator.comm_stream)
        NCCL.groupEnd()

        CUDA.record(communicator.sync_event, communicator.comm_stream)
        CUDA.cuStreamWaitEvent(CUDA.stream(), communicator.sync_event, UInt32(0))
    else
        NCCL.groupStart()
        enqueue_nccl_send_recv!(kernel!, bcs, nccl_comm, buffers)
        NCCL.groupEnd()
    end

    DC.recv_from_buffers!(c, buffers, grid, buffer_side)

    return nothing
end

#####
##### synchronize_communication! for NCCLDistributedField
#####
##### Waits for async NCCL comm_stream to complete, then unpacks recv buffers.
#####

function synchronize_communication!(field::NCCLDistributedField)
    if !isempty(pending_unpacks)
        arch = DC.architecture(field.grid)
        # Make default stream wait for comm_stream completion
        CUDA.cuStreamWaitEvent(CUDA.stream(), arch.communicator.sync_event, UInt32(0))

        for pending in pending_unpacks
            DC.recv_from_buffers!(pending.c, pending.buffers, pending.grid, pending.side)
        end
        empty!(pending_unpacks)
    end
    return nothing
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
##### Batched multi-field halo fill (synchronous only)
#####
##### For single-field fill_halo_regions! calls (like pressure), batch
##### all sides into one NCCL group. For multi-field async calls from
##### update_state!, the base code iterates per-field through our
##### distributed_fill_halo_event! which handles async properly.
#####

function Oceananigans.BoundaryConditions.fill_halo_regions!(field::NCCLDistributedField, args...;
                                                             async=false, kwargs...)
    if async
        # Async: use per-task path via distributed_fill_halo_event! (handles comm_stream)
        return Oceananigans.BoundaryConditions.fill_halo_regions!(
            field.data, field.boundary_conditions, field.indices,
            instantiated_location(field), field.grid, field.communication_buffers,
            args...; async, kwargs...)
    end
    # Synchronous: use batched path (all sides in one NCCL group)
    return nccl_fill_halo_regions!((field,), args...; kwargs...)
end


function nccl_fill_halo_regions!(fields, args...; only_local_halos=false, kwargs...)
    isempty(fields) && return nothing

    grid = first(fields).grid
    arch = DC.architecture(grid)
    nccl_comm = arch.communicator.nccl
    conn = arch.connectivity
    has_corners = !(isnothing(conn.southwest) && isnothing(conn.southeast) &&
                    isnothing(conn.northwest) && isnothing(conn.northeast))

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
##### Enqueue NCCL Send/Recv (called inside groupStart/groupEnd)
#####

function enqueue_nccl_send_recv!(::DistributedFillHalo{<:WestAndEast}, bcs, nccl_comm, bufs; stream_kw...)
    NCCL.Send(bufs.west.send, nccl_comm; dest=bcs[1].condition.to, stream_kw...)
    NCCL.Recv!(bufs.west.recv, nccl_comm; source=bcs[1].condition.to, stream_kw...)
    NCCL.Send(bufs.east.send, nccl_comm; dest=bcs[2].condition.to, stream_kw...)
    NCCL.Recv!(bufs.east.recv, nccl_comm; source=bcs[2].condition.to, stream_kw...)
    return nothing
end

function enqueue_nccl_send_recv!(::DistributedFillHalo{<:SouthAndNorth}, bcs, nccl_comm, bufs; stream_kw...)
    NCCL.Send(bufs.south.send, nccl_comm; dest=bcs[1].condition.to, stream_kw...)
    NCCL.Recv!(bufs.south.recv, nccl_comm; source=bcs[1].condition.to, stream_kw...)
    NCCL.Send(bufs.north.send, nccl_comm; dest=bcs[2].condition.to, stream_kw...)
    NCCL.Recv!(bufs.north.recv, nccl_comm; source=bcs[2].condition.to, stream_kw...)
    return nothing
end

for side in (:West, :East, :South, :North)
    side_sym = Symbol(lowercase(String(side)))
    @eval function enqueue_nccl_send_recv!(::DistributedFillHalo{<:$side}, bcs, nccl_comm, bufs; stream_kw...)
        NCCL.Send(bufs.$side_sym.send, nccl_comm; dest=bcs[1].condition.to, stream_kw...)
        NCCL.Recv!(bufs.$side_sym.recv, nccl_comm; source=bcs[1].condition.to, stream_kw...)
        return nothing
    end
end

enqueue_nccl_send_recv!(::DistributedFillHalo{<:BottomAndTop}, args...; kw...) = nothing
enqueue_nccl_send_recv!(::DistributedFillHalo{<:Bottom}, args...; kw...) = nothing
enqueue_nccl_send_recv!(::DistributedFillHalo{<:Top}, args...; kw...) = nothing
