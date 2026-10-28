using Oceananigans.BoundaryConditions: FieldBoundaryConditions
using Oceananigans.Grids: halo_size

using Adapt

import Oceananigans.Architectures: on_architecture
import Oceananigans.Fields: communication_buffers

"""
    CommunicationBuffers(west, east, south, north, state)

The buffers of the halo communication of a distributed field. Every side holds the buffers it sends (`send`) to and
receives (`recv`) from other ranks, none on a side that does not communicate. Corners belong to the south and north sides.
"""
struct CommunicationBuffers{W, E, S, N, CS}
    west :: W
    east :: E
   south :: S
   north :: N
   state :: CS
end

# Buffers are never needed in kernels and stay where they are
Adapt.adapt_structure(to, buff::CommunicationBuffers) = nothing
on_architecture(arch, buff::CommunicationBuffers) = buff

"""
    HaloBuffer(rank, tag, data, irange, jrange, sign, requests)

The `data` exchanged with `rank` for the cells `irange, jrange` of a field (parent indices, over the whole vertical).
Sent cells are multiplied by `sign`.
"""
struct HaloBuffer{D}
      rank :: Int
       tag :: Int
      data :: D
    irange :: StepRange{Int, Int}
    jrange :: StepRange{Int, Int}
      sign :: Int
  requests :: MPI.UnsafeMultiRequest
end

communication_buffers(grid::DistributedGrid, data, ::Nothing, loc) = nothing
communication_buffers(grid::DistributedGrid, data, ::Missing, loc) = nothing

function communication_buffers(grid::DistributedGrid, data, bcs::FieldBoundaryConditions, loc)
    arch = architecture(grid)
    local_rank = arch.local_rank

    FT = eltype(data)
    Nz = size(parent(data), 3)
    D  = typeof(zeros(arch, FT, 0, 0, 0))
    west  = (send = HaloBuffer{D}[], recv = HaloBuffer{D}[])
    east  = (send = HaloBuffer{D}[], recv = HaloBuffer{D}[])
    south = (send = HaloBuffer{D}[], recv = HaloBuffer{D}[])
    north = (send = HaloBuffer{D}[], recv = HaloBuffer{D}[])
    sides = (; west, east, south, north)

    field_tag = mod(next_field_tag!(arch), MPI.tag_ub() ÷ halo_tag_slots)
    parent_x, parent_y = halo_geometry(arch.connectivity, local_rank, halo_size(grid), loc).parent
    i₀ = first(parent_x) - 1
    j₀ = first(parent_y) - 1

    # A message is sent with the fill of the sender's side that faces the receiver (the fold, for a mirror image), or of
    # the opposite side when the facing side does not communicate
    opposite = (west = :east, east = :west, south = :north, north = :south)

    for rank in 0:length(arch.connectivity.ranges)-1
        for (slot, overlap) in enumerate(halo_overlaps(grid, local_rank, rank, loc))
            (; side, cells) = overlap
            tag = halo_tag_slots * field_tag + slot
            i = cells[1] .- i₀
            j = cells[2] .- j₀
            data = zeros(arch, FT, length(i), length(j), Nz)
            push!(sides[side].recv, HaloBuffer{D}(rank, tag, data, i, j, 1, MPI.UnsafeMultiRequest(1)))
        end

        for (slot, overlap) in enumerate(halo_overlaps(grid, rank, local_rank, loc))
            (; side, cells, σ, ox, oy) = overlap
            tag = halo_tag_slots * field_tag + slot

            # The cells that the halo cells copy, in the order of the halo cells
            i = σ * first(cells[1]) + ox - i₀ : σ : σ * last(cells[1]) + ox - i₀
            j = σ * first(cells[2]) + oy - j₀ : σ : σ * last(cells[2]) + oy - j₀

            data = zeros(arch, FT, length(i), length(j), Nz)
            sign = σ == 1 ? 1 : something(zipper_sign(bcs.north), 1)
            facing_side = σ == 1 ? opposite[side] : side
            sending_side = getproperty(bcs, facing_side) isa DCBC ? facing_side : opposite[facing_side]
            push!(sides[sending_side].send, HaloBuffer{D}(rank, tag, data, i, j, sign, MPI.UnsafeMultiRequest(1)))
        end
    end

    return CommunicationBuffers(west, east, south, north, communication_state(arch))
end

#####
##### Packing, unpacking and completing the messages of the sides of a fill
#####

function fill_send_buffers!(c::OffsetArray, sides::Tuple)
    for side in sides, buffer in side.send
        buffer.data .= buffer.sign .* view(parent(c), buffer.irange, buffer.jrange, :)
    end
    return nothing
end

function recv_from_buffers!(c::OffsetArray, sides::Tuple)
    for side in sides, buffer in side.recv
        view(parent(c), buffer.irange, buffer.jrange, :) .= buffer.data
    end
    return nothing
end

# The requests completed by the progress worker are already null, so `Waitall` only completes those posted by the main thread
function wait_for_messages!(buffers::CommunicationBuffers)
    while buffers.state.pending_messages[] != 0
        rethrow_progress_worker_failure()
        yield()
    end
    for side in (buffers.west, buffers.east, buffers.south, buffers.north)
        foreach(buffer -> MPI.Waitall(buffer.requests), side.send)
        foreach(buffer -> MPI.Waitall(buffer.requests), side.recv)
    end
    return nothing
end
