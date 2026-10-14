using Oceananigans.BoundaryConditions
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, BoundaryCondition
using Oceananigans.BoundaryConditions: MultiRegionCommunication, DistributedCommunication, side_name
using Oceananigans.Grids: halo_size, size, topology
using Oceananigans.Utils: launch!

using Adapt

import Oceananigans.Architectures: on_architecture
import Oceananigans.Fields: communication_buffers

const MCBC = BoundaryCondition{<:MultiRegionCommunication}
const DCBC = BoundaryCondition{<:DistributedCommunication}

"""
    CommunicationBuffers(west, east, south, north, state)

Buffers for the halo exchange of a field on each side. On a `Distributed` architecture, a side holds the halo
messages it `sends` to and `receives` from other ranks; corners belong to the south and north sides.
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
    HaloSegment(rank, tag, buffer, i, j, sign, requests)

The cells `i, j` (parent indices, over the whole vertical) of a field that are exchanged with `rank` through `buffer`.
Sent cells are multiplied by `sign`.
"""
struct HaloSegment{B, R}
    rank :: Int
    tag :: Int
    buffer :: B
    i :: R
    j :: R
    sign :: Int
    requests :: MPI.UnsafeMultiRequest
end

pack!(c, segment::HaloSegment) = segment.buffer .= segment.sign .* view(c, segment.i, segment.j, :)
unpack!(c, segment::HaloSegment) = view(c, segment.i, segment.j, :) .= segment.buffer

halo_messages(bc, B) = nothing
halo_messages(::DCBC, B) = (sends = HaloSegment{B, StepRange{Int, Int}}[], receives = HaloSegment{B, UnitRange{Int}}[])

# The sign of a distributed tripolar fold
fold_sign(bc) = 1
fold_sign(bc::BoundaryCondition{<:DistributedCommunication, <:Integer}) = bc.condition

communication_buffers(grid::DistributedGrid, data, ::Nothing, loc) = nothing
communication_buffers(grid::DistributedGrid, data, ::Missing, loc) = nothing

function communication_buffers(grid::DistributedGrid, data, bcs::FieldBoundaryConditions, loc)
    arch = architecture(grid)
    layout = arch.connectivity
    local_rank = arch.local_rank
    H = halo_size(grid)
    sign = fold_sign(bcs.north)

    FT = eltype(data)
    Nz = size(parent(data), 3)
    B = typeof(zeros(arch, FT, 0, 0, 0))
    sides = (west = halo_messages(bcs.west, B), east = halo_messages(bcs.east, B),
             south = halo_messages(bcs.south, B), north = halo_messages(bcs.north, B))

    field_tag = mod(next_field_tag!(arch), MPI.tag_ub() ÷ halo_tag_slots)
    message_tag(slot) = Int(halo_tag_slots * field_tag + slot)
    offset(rank, d) = first(parent_range(layout, rank, d, H[d], loc)) - 1

    for rank in 0:length(layout.ranges)-1
        for (slot, overlap) in enumerate(halo_overlaps(layout, local_rank, rank, loc, sign, H))
            (; side, cells) = overlap
            i = cells[1] .- offset(local_rank, 1)
            j = cells[2] .- offset(local_rank, 2)
            buffer = zeros(arch, FT, length(i), length(j), Nz)
            push!(sides[side_name(side)].receives, HaloSegment(rank, message_tag(slot), buffer, i, j, 1, MPI.UnsafeMultiRequest(1)))
        end

        for (slot, overlap) in enumerate(halo_overlaps(layout, rank, local_rank, loc, sign, H))
            (; side, cells, image) = overlap
            ox = image.ox - offset(local_rank, 1)
            oy = image.oy - offset(local_rank, 2)
            i = image.σx * first(cells[1]) + ox : image.σx : image.σx * last(cells[1]) + ox
            j = image.σy * first(cells[2]) + oy : image.σy : image.σy * last(cells[2]) + oy
            buffer = zeros(arch, FT, length(i), length(j), Nz)
            # A copy leaves through the opposite side, a mirror image through the same side
            sending_side = image.σx == 1 ? opposite_side(side) : side
            push!(sides[side_name(sending_side)].sends, HaloSegment(rank, message_tag(slot), buffer, i, j, image.sign, MPI.UnsafeMultiRequest(1)))
        end
    end

    return CommunicationBuffers(sides..., communication_state(arch))
end

#####
##### Packing and unpacking the messages of one or more sides
#####

side_messages(buff, ::West)          = (buff.west,)
side_messages(buff, ::East)          = (buff.east,)
side_messages(buff, ::South)         = (buff.south,)
side_messages(buff, ::North)         = (buff.north,)
side_messages(buff, ::WestAndEast)   = (buff.west, buff.east)
side_messages(buff, ::SouthAndNorth) = (buff.south, buff.north)
side_messages(buff)                  = (buff.west, buff.east, buff.south, buff.north)

function fill_send_buffers!(c::OffsetArray, buff::CommunicationBuffers, side)
    for messages in side_messages(buff, side), segment in messages.sends
        pack!(parent(c), segment)
    end
    return nothing
end

recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers) = unpack_messages!(c, side_messages(buff))
recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers, side) = unpack_messages!(c, side_messages(buff, side))

function unpack_messages!(c, sides)
    for messages in sides
        isnothing(messages) && continue
        for segment in messages.receives
            unpack!(parent(c), segment)
        end
    end
    return nothing
end

#####
##### MultiRegion buffers: one message per side, holding the whole strip
#####

struct OneDBuffer{B}
    send :: B
    recv :: B
end

Adapt.adapt_structure(to, buff::OneDBuffer) = nothing

CommunicationBuffers(grid, data, ::Missing) = nothing
CommunicationBuffers(grid, data, ::Nothing) = nothing

function CommunicationBuffers(grid, data, boundary_conditions::FieldBoundaryConditions)
    arch = architecture(grid)
    Hx, Hy, _ = halo_size(grid)

    west  = x_communication_buffer(arch, grid, data, Hx, boundary_conditions.west)
    east  = x_communication_buffer(arch, grid, data, Hx, boundary_conditions.east)
    south = y_communication_buffer(arch, grid, data, Hy, boundary_conditions.south)
    north = y_communication_buffer(arch, grid, data, Hy, boundary_conditions.north)

    return CommunicationBuffers(west, east, south, north, nothing)
end

x_communication_buffer(arch, grid, data, H, bc) = nothing
y_communication_buffer(arch, grid, data, H, bc) = nothing

function x_communication_buffer(arch, grid, data, H, ::MCBC)
    _, Ty, Tz = size(parent(data))
    FT = eltype(data)
    send = on_architecture(arch, zeros(FT, H, Ty, Tz))
    recv = on_architecture(arch, zeros(FT, H, Ty, Tz))
    return OneDBuffer(send, recv)
end

function y_communication_buffer(arch, grid, data, H, ::MCBC)
    Tx, _, Tz = size(parent(data))
    FT = eltype(data)
    send = on_architecture(arch, zeros(FT, Tx, H, Tz))
    recv = on_architecture(arch, zeros(FT, Tx, H, Tz))
    return OneDBuffer(send, recv)
end

function fill_send_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid::AbstractGrid)
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)
    fill_send_buffer!(buff.west,  view(parent(c), 1+Hx:2Hx,   :, :))
    fill_send_buffer!(buff.east,  view(parent(c), 1+Nx:Nx+Hx, :, :))
    fill_send_buffer!(buff.south, view(parent(c), :, 1+Hy:2Hy,   :))
    fill_send_buffer!(buff.north, view(parent(c), :, 1+Ny:Ny+Hy, :))
    return nothing
end

fill_send_buffers!(c::OffsetArray, ::Nothing, args...) = nothing
recv_from_buffers!(c::OffsetArray, ::Nothing, args...) = nothing

fill_send_buffer!(::Nothing, strip) = nothing
fill_send_buffer!(buff::OneDBuffer, strip) = buff.send .= strip

# The requests completed by the progress worker are already null, so `Waitall` only completes those posted by the main thread
function wait_for_messages!(buff::CommunicationBuffers)
    while buff.state.pending_messages[] != 0
        rethrow_progress_worker_failure()
        yield()
    end
    foreach(waitall_requests!, side_messages(buff))
    return nothing
end

waitall_requests!(::Nothing) = nothing

function waitall_requests!(messages)
    foreach(segment -> MPI.Waitall(segment.requests), messages.sends)
    foreach(segment -> MPI.Waitall(segment.requests), messages.receives)
    return nothing
end

posted_by_main_thread(buff::CommunicationBuffers) = CommunicationBuffers(buff.west, buff.east, buff.south, buff.north, posted_by_main_thread(buff.state))
