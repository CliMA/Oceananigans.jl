using Oceananigans.BoundaryConditions: DistributedFillHalo, extract_bc

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
DC.mpi_communicator(c::NCCLCommunicator) = c.mpi
DC.check_field_tags(c::NCCLCommunicator, field_count) = DC.check_field_tags(c.mpi, field_count)

# The host MPI library need not be CUDA-aware: device buffers reaching these forwarded
# collectives (e.g. the tiny result arrays of distributed field reductions) must be
# staged through the host, or MPI's host memcpy segfaults on the device pointer.
function MPI.Allreduce!(sendbuf::CUDA.AnyCuArray, recvbuf::CUDA.AnyCuArray, op, c::NCCLCommunicator)
    host_sendbuf = Array(sendbuf)
    host_recvbuf = similar(host_sendbuf)
    MPI.Allreduce!(host_sendbuf, host_recvbuf, op, c.mpi)
    copyto!(recvbuf, host_recvbuf)
    return recvbuf
end

function MPI.Allreduce!(sendrecvbuf::CUDA.AnyCuArray, op, c::NCCLCommunicator)
    host_buffer = Array(sendrecvbuf)
    MPI.Allreduce!(host_buffer, op, c.mpi)
    copyto!(sendrecvbuf, host_buffer)
    return sendrecvbuf
end

MPI.Allreduce(sendbuf::CUDA.AnyCuArray, op, c::NCCLCommunicator) = MPI.Allreduce(Array(sendbuf), op, c.mpi)

function MPI.Bcast!(buf::CUDA.AnyCuArray, c::NCCLCommunicator; kwargs...)
    host_buffer = Array(buf)
    MPI.Bcast!(host_buffer, c.mpi; kwargs...)
    copyto!(buf, host_buffer)
    return buf
end

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
                          mpi_arch.field_count,
                          mpi_arch.devices)
end

#####
##### distributed_fill_halo_event! for NCCLDistributedGrid
#####
##### Sync mode (async=false):  pack → NCCL on comm_stream → wait → unpack
##### Async mode (async=true):  pack → NCCL on comm_stream → defer unpack
#####   Interior computation proceeds on default stream while NCCL transfers on comm_stream.
#####   synchronize_communication! later waits for comm_stream and unpacks.
#####

# Storage for pending async unpacks: (data, sides, event) tuples
const pending_unpacks = Vector{Any}()
const pending_unpacks_lock = ReentrantLock()

function DC.distributed_fill_halo_event!(c, kernel!::DistributedFillHalo, bcs, loc,
                                         arch::NCCLDistributedArchitecture, grid, buffers, args;
                                         async = false, only_local_halos = false,
                                         fill_open_bcs = true, kwargs...)
    only_local_halos && return nothing

    communicator = arch.communicator
    nccl_comm = communicator.nccl
    sides = extract_bc(buffers, kernel!.side)

    # Pack send buffers, then make comm_stream wait for the pack to complete.
    DC.fill_send_buffers!(c, sides)
    pack_done = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
    CUDA.record(pack_done)
    CUDA.wait(pack_done, communicator.comm_stream)

    NCCL.groupStart()
    enqueue_nccl_send_recv!(sides, nccl_comm; stream=communicator.comm_stream)
    NCCL.groupEnd()

    comm_done = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
    CUDA.record(comm_done, communicator.comm_stream)

    if async
        lock(pending_unpacks_lock) do
            push!(pending_unpacks, (; c, sides, event=comm_done))
        end
        return nothing
    end

    # Sync: have the default stream wait for this fill's NCCL ops, then unpack.
    CUDA.wait(comm_done, CUDA.stream())
    DC.recv_from_buffers!(c, sides)
    return nothing
end

#####
##### synchronize_communication! for NCCLDistributedField
#####
##### Waits for async NCCL comm_stream to complete, then unpacks recv buffers.
#####

function synchronize_communication!(field::NCCLDistributedField)
    arch = DC.architecture(field.grid)

    # Synchronize when using heterogeneous NCCL/MPI
    DC.wait_for_messages!(field)

    lock(pending_unpacks_lock) do
        if !isempty(pending_unpacks)
            # Wait on each fill's completion event before unpacking; this also
            # fences the next iteration's packs behind the finished transfers.
            for pending in pending_unpacks
                CUDA.wait(pending.event, CUDA.stream())
                DC.recv_from_buffers!(pending.c, pending.sides)
            end
            empty!(pending_unpacks)
        end
    end
    return nothing
end

#####
##### Enqueue NCCL Send/Recv (called inside groupStart/groupEnd)
#####

# NCCL has no Bool dtype; reinterpret as UInt8 (same size)
nccl_buffer(a::AbstractArray{Bool}) = reinterpret(UInt8, a)
nccl_buffer(a) = a

# NCCL matches Send/Recv by post order per peer, not by tag: both ranks post the messages to a peer in tag order
function enqueue_nccl_send_recv!(sides, nccl_comm; stream_kw...)
    send = sort!([buffer for side in sides for buffer in side.send], by = buffer -> (buffer.rank, buffer.tag))
    recv = sort!([buffer for side in sides for buffer in side.recv], by = buffer -> (buffer.rank, buffer.tag))

    for buffer in send
        NCCL.Send(nccl_buffer(buffer.data), nccl_comm; dest=buffer.rank, stream_kw...)
    end
    for buffer in recv
        NCCL.Recv!(nccl_buffer(buffer.data), nccl_comm; source=buffer.rank, stream_kw...)
    end

    return nothing
end
