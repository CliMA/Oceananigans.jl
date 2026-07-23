using Oceananigans.BoundaryConditions: DistributedFillHalo, WestAndEast, SouthAndNorth,
                                       West, East, South, North, BottomAndTop, Bottom, Top

import Oceananigans.DistributedComputations: synchronize_communication!

#####
##### NCCLCommunicator
#####

struct NCCLCommunicator{NC, MC, CS}
    nccl        :: NC   # NCCL.Communicator
    mpi         :: MC   # MPI.Comm
    comm_stream :: CS   # Dedicated CUDA stream for async NCCL ops
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

function DC.NCCLDistributed(child_arch = GPU(); partition = nothing, kwargs...)
    mpi_arch = Distributed(child_arch; partition, kwargs...)
    nccl_comm = create_nccl_comm_from_mpi(mpi_arch.communicator)
    comm_stream = CUDA.CuStream(; flags=CUDA.STREAM_NON_BLOCKING)
    nccl_communicator = NCCLCommunicator(nccl_comm, mpi_arch.communicator, comm_stream)

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

# No sync_device! override here: mainline MPI halo fills and `maybe_all_reduce!`
# need a real device sync before GPU-aware MPI reads GPU buffers. The NCCL fill
# path below is stream-ordered and never calls it.

#####
##### distributed_fill_halo_event! for NCCLDistributedGrid
#####
##### Sync mode (async=false):  pack → NCCL on comm_stream → wait → unpack
##### Async mode (async=true):  pack → NCCL on comm_stream → defer unpack
#####   Interior computation proceeds on default stream while NCCL transfers on comm_stream.
#####   synchronize_communication! later waits for comm_stream and unpacks.
#####

# Storage for pending async unpacks: (data, buffers, grid, side, event) tuples
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

    # Each stream dependency uses a fresh event: a CUDA event only remembers its
    # latest record, so a shared event re-recorded by every fill can leave waits
    # targeting the wrong record and let unpacks race in-flight transfers.

    # Pack send buffers, then make comm_stream wait for the pack to complete.
    DC.fill_send_buffers!(c, buffers, grid, buffer_side)
    pack_done = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
    CUDA.record(pack_done)
    CUDA.cuStreamWaitEvent(communicator.comm_stream, pack_done, UInt32(0))

    NCCL.groupStart()
    enqueue_nccl_send_recv!(kernel!, bcs, nccl_comm, buffers; stream=communicator.comm_stream)
    NCCL.groupEnd()

    comm_done = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
    CUDA.record(comm_done, communicator.comm_stream)

    if async
        lock(pending_unpacks_lock) do
            push!(pending_unpacks, (; c, buffers, grid, side=buffer_side, event=comm_done))
        end
        return nothing
    end

    # Sync: have the default stream wait for this fill's NCCL ops, then unpack.
    CUDA.cuStreamWaitEvent(CUDA.stream(), comm_done, UInt32(0))
    DC.recv_from_buffers!(c, buffers, grid, buffer_side)
    return nothing
end

#####
##### synchronize_communication! for NCCLDistributedField
#####
##### Waits for async NCCL comm_stream to complete, then unpacks recv buffers.
#####

function synchronize_communication!(field::NCCLDistributedField)
    arch = DC.architecture(field.grid)

    # Complete requests pooled by fills that took the mainline MPI path and reset
    # the MPI tag, as the mainline method does — otherwise the tag counter grows
    # past MPI_TAG_UB and MPI_Irecv errors. Unlike mainline, do NOT unpack
    # `field`'s MPI buffers: the pooled requests belong to other fields, and
    # `field`'s halos arrived via NCCL, so its MPI buffers hold stale data.
    if !isempty(arch.mpi_requests)
        DC.cooperative_waitall!(arch.mpi_requests)
        arch.mpi_tag[] = 0
        empty!(arch.mpi_requests)
    end

    lock(pending_unpacks_lock) do
        if !isempty(pending_unpacks)
            # Wait on each fill's completion event before unpacking; this also
            # fences the next iteration's packs behind the finished transfers.
            for pending in pending_unpacks
                CUDA.cuStreamWaitEvent(CUDA.stream(), pending.event, UInt32(0))
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
    send_buf = eltype(buffers.send) === Bool ? reinterpret(UInt8, buffers.send) : buffers.send
    recv_buf = eltype(buffers.recv) === Bool ? reinterpret(UInt8, buffers.recv) : buffers.recv
    NCCL.Send(send_buf, nccl_comm; dest=corner_rank)
    NCCL.Recv!(recv_buf, nccl_comm; source=corner_rank)
    return nothing
end

#####
##### Enqueue NCCL Send/Recv (called inside groupStart/groupEnd)
#####

function _nccl_send_recv_pair!(buf, bc, nccl_comm; stream_kw...)
    isnothing(buf) && return nothing
    # NCCL has no Bool dtype; reinterpret as UInt8 (same size)
    send_buf = eltype(buf.send) === Bool ? reinterpret(UInt8, buf.send) : buf.send
    recv_buf = eltype(buf.recv) === Bool ? reinterpret(UInt8, buf.recv) : buf.recv
    NCCL.Send(send_buf, nccl_comm; dest=bc.condition.to, stream_kw...)
    NCCL.Recv!(recv_buf, nccl_comm; source=bc.condition.to, stream_kw...)
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
