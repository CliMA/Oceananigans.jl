using Oceananigans.BoundaryConditions: DistributedFillHalo, WestAndEast, SouthAndNorth,
                                       West, East, South, North, BottomAndTop, Bottom, Top

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

const NCCLDistributedArchitecture = Distributed{<:GPU, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:NCCLCommunicator}
const NCCLDistributedGrid{FT, TX, TY, TZ}  = Oceananigans.Grids.AbstractGrid{FT, TX, TY, TZ, <:NCCLDistributedArchitecture}
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

# NCCL is stream-native; sync is handled via CUDA events on comm_stream.
sync_device!(::NCCLDistributedArchitecture) = nothing

#####
##### distributed_fill_halo_event! for NCCLDistributedGrid
#####
##### Sync mode (async=false):  pack → NCCL on comm_stream → wait → unpack
##### Async mode (async=true):  pack → NCCL on comm_stream → defer unpack
#####   Interior computation proceeds on default stream while NCCL transfers on comm_stream.
#####   synchronize_communication! later waits for comm_stream and unpacks.
#####

# Storage for pending async unpacks: (data, buffers, grid, side) tuples
const pending_unpacks = Vector{Any}()
const pending_unpacks_lock = ReentrantLock()

function DC.distributed_fill_halo_event!(c, kernel!::DistributedFillHalo, bcs, loc,
                                         grid::NCCLDistributedGrid, buffers, args...;
                                         async = false, only_local_halos = false,
                                         fill_open_bcs = true, kwargs...)
    only_local_halos && return nothing

    arch = DC.architecture(grid)
    communicator = arch.communicator
    nccl_comm = communicator.nccl
    buffer_side = kernel!.side

    # Pack send buffers, then make comm_stream wait for the pack to complete.
    DC.fill_send_buffers!(c, buffers, grid, buffer_side)
    CUDA.record(communicator.sync_event)
    CUDA.cuStreamWaitEvent(communicator.comm_stream, communicator.sync_event, UInt32(0))

    NCCL.groupStart()
    enqueue_nccl_send_recv!(kernel!, bcs, nccl_comm, buffers; stream=communicator.comm_stream)
    NCCL.groupEnd()

    CUDA.record(communicator.sync_event, communicator.comm_stream)

    if async
        lock(pending_unpacks_lock) do
            push!(pending_unpacks, (; c, buffers, grid, side=buffer_side))
        end
        return nothing
    end

    # Sync: have the default stream wait for comm_stream, then unpack.
    CUDA.cuStreamWaitEvent(CUDA.stream(), communicator.sync_event, UInt32(0))
    DC.recv_from_buffers!(c, buffers, grid, buffer_side)
    return nothing
end

#####
##### synchronize_communication! for NCCLDistributedField
#####
##### Waits for async NCCL comm_stream to complete, then unpacks recv buffers.
#####

function synchronize_communication!(field::NCCLDistributedField)
    lock(pending_unpacks_lock) do
        if !isempty(pending_unpacks)
            arch = DC.architecture(field.grid)
            CUDA.cuStreamWaitEvent(CUDA.stream(), arch.communicator.sync_event, UInt32(0))
            for pending in pending_unpacks
                DC.recv_from_buffers!(pending.c, pending.buffers, pending.grid, pending.side)
            end
            empty!(pending_unpacks)
        end
    end
    return nothing
end

#####
##### NCCL corner communication
#####

function DC.fill_corners!(c, connectivity, indices, loc, arch::NCCLDistributedArchitecture,
                          grid, buffers, args...; only_local_halos=false, kw...)
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
nccl_corner_send_recv!(nccl_comm, corner_rank, ::Nothing) = nothing
nccl_corner_send_recv!(nccl_comm, ::Nothing, ::Nothing) = nothing

function nccl_corner_send_recv!(nccl_comm, corner_rank, buffers)
    NCCL.Send(buffers.send, nccl_comm; dest=corner_rank)
    NCCL.Recv!(buffers.recv, nccl_comm; source=corner_rank)
    return nothing
end

#####
##### Enqueue NCCL Send/Recv (called inside groupStart/groupEnd)
#####

function _nccl_send_recv_pair!(buf, bc, nccl_comm; stream_kw...)
    isnothing(buf) && return nothing
    NCCL.Send(buf.send, nccl_comm; dest=bc.condition.to, stream_kw...)
    NCCL.Recv!(buf.recv, nccl_comm; source=bc.condition.to, stream_kw...)
    return nothing
end

function enqueue_nccl_send_recv!(::DistributedFillHalo{<:WestAndEast}, bcs, nccl_comm, bufs; stream_kw...)
    _nccl_send_recv_pair!(bufs.west, bcs[1], nccl_comm; stream_kw...)
    _nccl_send_recv_pair!(bufs.east, bcs[2], nccl_comm; stream_kw...)
    return nothing
end

function enqueue_nccl_send_recv!(::DistributedFillHalo{<:SouthAndNorth}, bcs, nccl_comm, bufs; stream_kw...)
    _nccl_send_recv_pair!(bufs.south, bcs[1], nccl_comm; stream_kw...)
    _nccl_send_recv_pair!(bufs.north, bcs[2], nccl_comm; stream_kw...)
    return nothing
end

for side in (:West, :East, :South, :North)
    side_sym = Symbol(lowercase(String(side)))
    @eval function enqueue_nccl_send_recv!(::DistributedFillHalo{<:$side}, bcs, nccl_comm, bufs; stream_kw...)
        _nccl_send_recv_pair!(bufs.$side_sym, bcs[1], nccl_comm; stream_kw...)
        return nothing
    end
end

enqueue_nccl_send_recv!(::DistributedFillHalo{<:BottomAndTop}, args...; kw...) = nothing
enqueue_nccl_send_recv!(::DistributedFillHalo{<:Bottom}, args...; kw...) = nothing
enqueue_nccl_send_recv!(::DistributedFillHalo{<:Top}, args...; kw...) = nothing
