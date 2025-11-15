using Oceananigans.BoundaryConditions
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, BoundaryCondition
using Oceananigans.BoundaryConditions: MultiRegionCommunication, DistributedCommunication
using Oceananigans.Grids: halo_size, size, topology
using Oceananigans.Utils: launch!

using Adapt

using KernelAbstractions: @kernel, @index

import Oceananigans.Architectures: on_architecture
import Oceananigans.Fields: communication_buffers

const MCBC = BoundaryCondition{<:MultiRegionCommunication}
const DCBC = BoundaryCondition{<:DistributedCommunication}

struct CommunicationBuffers{W, E, S, N, SW, SE, NW, NE}
    west :: W
    east :: E
   south :: S
   north :: N
   southwest :: SW
   southeast :: SE
   northwest :: NW
   northeast :: NE
end

Adapt.adapt_structure(to, buff::CommunicationBuffers) =
    CommunicationBuffers(Adapt.adapt(to, buff.west), 
                         Adapt.adapt(to, buff.east),    
                         Adapt.adapt(to, buff.north), 
                         Adapt.adapt(to, buff.south), 
                         Adapt.adapt(to, buff.southwest), 
                         Adapt.adapt(to, buff.southeast), 
                         Adapt.adapt(to, buff.northwest), 
                         Adapt.adapt(to, buff.northeast))

on_architecture(arch, buff::CommunicationBuffers) =
    CommunicationBuffers(on_architecture(arch, buff.west), 
                         on_architecture(arch, buff.east),    
                         on_architecture(arch, buff.north), 
                         on_architecture(arch, buff.south), 
                         on_architecture(arch, buff.southwest), 
                         on_architecture(arch, buff.southeast), 
                         on_architecture(arch, buff.northwest), 
                         on_architecture(arch, buff.northeast))

communication_buffers(grid::DistributedGrid, data, boundary_conditions) = CommunicationBuffers(grid, data, boundary_conditions)

"""
    CommunicationBuffers(grid, data, boundary_conditions)

Construct communication buffers for distributed halo exchange.

`CommunicationBuffers` stores send/receive buffers for each spatial direction and corner 
in a distributed grid. During halo exchange, data is copied from the interior domain into 
send buffers, communicated via MPI to neighboring processes, and then unpacked from receive 
buffers into halo regions.

# Buffer Types
Edge buffers (`west`, `east`, `south`, `north`) can be:
- `OneDBuffer`: For 1D parallelization or when at domain edges (includes corners)
- `TwoDBuffer`: For 2D parallelization interior processes (excludes corners)
- `nothing`: When no communication is needed in that direction

Corner buffers (`southwest`, `southeast`, `northwest`, `northeast`) can be:
- `CornerBuffer`: For 2D parallelization where corners need separate communication
- `nothing`: For 1D parallelization or when corners are handled by edge buffers

# See also
[`OneDBuffer`](@ref), [`TwoDBuffer`](@ref), [`CornerBuffer`](@ref)
"""
function CommunicationBuffers(grid, data, boundary_conditions::FieldBoundaryConditions)
    Hx, Hy, _ = halo_size(grid)
    arch = architecture(grid)
    Hx, Hy, Hz = halo_size(grid)

    west  = x_communication_buffer(arch, grid, data, Hx, boundary_conditions.west)
    east  = x_communication_buffer(arch, grid, data, Hx, boundary_conditions.east)
    south = y_communication_buffer(arch, grid, data, Hy, boundary_conditions.south)
    north = y_communication_buffer(arch, grid, data, Hy, boundary_conditions.north)

    sw = corner_communication_buffer(arch, grid, data, Hx, Hy, west, south)
    se = corner_communication_buffer(arch, grid, data, Hx, Hy, east, south)
    nw = corner_communication_buffer(arch, grid, data, Hx, Hy, west, north)
    ne = corner_communication_buffer(arch, grid, data, Hx, Hy, east, north)

    return CommunicationBuffers(west, east, south, north, sw, se, nw, ne)
end

CommunicationBuffers(grid, data, ::Missing) = nothing
CommunicationBuffers(grid, data, ::Nothing) = nothing

"""
    OneDBuffer{B}

Communication buffer for one-dimensional domain decomposition or edge boundaries.

In a one-dimensional parallelization (e.g., only in x or only in y), `OneDBuffer` 
contains the full extent of the halo region including the corners. This allows corner 
data to be communicated in a single pass along with the edge data.

`OneDBuffer` is also used in two-dimensional parallelizations for processes at domain 
edges (e.g., boundaries with `RightConnected` or `LeftConnected` topologies), where 
corner communication is not needed in that direction.

# Size
For x-direction: `(Hx, Ty, Tz)` where `Hx` is the halo size in x, `Ty` includes all y points (with halos)
For y-direction: `(Tx, Hy, Tz)` where `Hy` is the halo size in y, `Tx` includes all x points (with halos)
"""
struct OneDBuffer{B}
    send :: B
    recv :: B
end

"""
    TwoDBuffer{B}

Communication buffer for two-dimensional domain decomposition without corner regions.

In a two-dimensional parallelization where corners are communicated separately via 
`CornerBuffer`, `TwoDBuffer` contains only the edge halo region excluding the corners. 
This enables efficient communication patterns where edge and corner data are sent in 
separate MPI passes.

# Size
For x-direction (west/east): `(Hx, Ny, Tz)` where `Hx` is the halo size in x, `Ny` is the interior size in y
For y-direction (south/north): `(Nx, Hy, Tz)` where `Nx` is the interior size in x, `Hy` is the halo size in y
"""
struct TwoDBuffer{B}
    send :: B
    recv :: B
end

"""
    CornerBuffer{B}

Communication buffer for corner regions in two-dimensional domain decomposition.

In a two-dimensional parallelization, `CornerBuffer` handles the communication of 
diagonal corner regions (southwest, southeast, northwest, northeast) that are not 
covered by the edge communication buffers (`TwoDBuffer`). Corner communication ensures 
that all halo regions are properly synchronized between neighboring processes in both 
x and y directions.

# Size
`(Hx, Hy, Tz)` where `Hx` is the halo size in x, `Hy` is the halo size in y, and `Tz` is the size in z

# Note
Corner buffers are only created for `Distributed` architectures with two-dimensional 
parallelization and are `nothing` otherwise.
"""
struct CornerBuffer{B}
    send :: B
    recv :: B
end

# We never need to access buffers on the GPU!
Adapt.adapt_structure(to, buff::OneDBuffer) = nothing
Adapt.adapt_structure(to, buff::TwoDBuffer) = nothing
Adapt.adapt_structure(to, buff::CornerBuffer) = nothing

####
#### X and Y communication buffers
####

# Fallback
x_communication_buffer(arch, grid, data, H, bc) = nothing
y_communication_buffer(arch, grid, data, H, bc) = nothing

function x_communication_buffer(arch::Distributed, grid::AbstractGrid{<:Any, TX, TY}, data, H, ::DCBC) where {TX, TY}
    _, Ty, Tz = size(parent(data))
    Ny = size(grid, 2)
    FT = eltype(data)
    if (ranks(arch)[2] == 1) || (TY == RightConnected) || (TY == LeftConnected)
        send = on_architecture(arch, zeros(FT, H, Ty, Tz))
        recv = on_architecture(arch, zeros(FT, H, Ty, Tz))
        return OneDBuffer(send, recv)
    else
        send = on_architecture(arch, zeros(FT, H, Ny, Tz))
        recv = on_architecture(arch, zeros(FT, H, Ny, Tz))
        return TwoDBuffer(send, recv)
    end
end

function y_communication_buffer(arch::Distributed, grid::AbstractGrid{<:Any, TX, TY}, data, H, ::DCBC) where {TX, TY}
    Tx, _, Tz = size(parent(data))
    FT = eltype(data)
    Nx = size(grid, 1)
    if (ranks(arch)[1] == 1) || (TX == RightConnected) || (TX == LeftConnected)
        send = on_architecture(arch, zeros(FT, Tx, H, Tz))
        recv = on_architecture(arch, zeros(FT, Tx, H, Tz))
        return OneDBuffer(send, recv)
    else
        send = on_architecture(arch, zeros(FT, Nx, H, Tz))
        recv = on_architecture(arch, zeros(FT, Nx, H, Tz))
        return TwoDBuffer(send, recv)
    end
end

# Never pass corners in a MCBC.
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

#####
##### Corner communication buffers
#####

# Only used for `Distributed` architectures
corner_communication_buffer(arch, grid, data, Hx, Hy, xedge, yedge) = nothing

# Disambiguation
corner_communication_buffer(::Distributed, grid, data, Hx, Hy, ::Nothing, ::Nothing) = nothing
corner_communication_buffer(arch::Distributed, grid, data, Hx, Hy, ::Nothing, yedge) = nothing
corner_communication_buffer(arch::Distributed, grid, data, Hx, Hy, xedge, ::Nothing) = nothing
    
# CornerBuffer are used only  in the two-dimensional partitioning case, in all other cases they are equal to `nothing`
function corner_communication_buffer(arch::Distributed, grid, data, Hx, Hy, xedge, yedge)
    Tz = size(parent(data), 3)
    FT = eltype(data)
    send = on_architecture(arch, zeros(FT, Hx, Hy, Tz))
    recv = on_architecture(arch, zeros(FT, Hx, Hy, Tz))
    return CornerBuffer(send, recv)
end

"""
    fill_send_buffers!(c::OffsetArray, buffers::CommunicationBuffers, grid)

fills `buffers.send` from OffsetArray `c` preparing for message passing.
"""
function fill_send_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid)
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

     _fill_west_send_buffer!(parent(c), buff.west,  Hx, Hy, Nx, Ny)
     _fill_east_send_buffer!(parent(c), buff.east,  Hx, Hy, Nx, Ny)
    _fill_south_send_buffer!(parent(c), buff.south, Hx, Hy, Nx, Ny)
    _fill_north_send_buffer!(parent(c), buff.north, Hx, Hy, Nx, Ny)

    _fill_southwest_send_buffer!(parent(c), buff.southwest, Hx, Hy, Nx, Ny)
    _fill_southeast_send_buffer!(parent(c), buff.southeast, Hx, Hy, Nx, Ny)
    _fill_northwest_send_buffer!(parent(c), buff.northwest, Hx, Hy, Nx, Ny)
    _fill_northeast_send_buffer!(parent(c), buff.northeast, Hx, Hy, Nx, Ny)

    return nothing
end

fill_send_buffers!(c::OffsetArray, ::Nothing, grid) = nothing
fill_send_buffers!(c::OffsetArray, ::Nothing, grid, ::Val{:corners}) = nothing

function fill_send_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::Val{:corners})
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

    _fill_southwest_send_buffer!(parent(c), buff.southwest, Hx, Hy, Nx, Ny)
    _fill_southeast_send_buffer!(parent(c), buff.southeast, Hx, Hy, Nx, Ny)
    _fill_northwest_send_buffer!(parent(c), buff.northwest, Hx, Hy, Nx, Ny)
    _fill_northeast_send_buffer!(parent(c), buff.northeast, Hx, Hy, Nx, Ny)

    return nothing
end

#####
##### Single sided fill_send_buffers!
#####

fill_send_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::West) = 
    _fill_west_send_buffer!(parent(c), buff.west, halo_size(grid)[[1, 2]]..., size(grid)[[1, 2]]...)
fill_send_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::East) = 
    _fill_east_send_buffer!(parent(c), buff.east, halo_size(grid)[[1, 2]]..., size(grid)[[1, 2]]...)
fill_send_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::South) = 
    _fill_south_send_buffer!(parent(c), buff.south, halo_size(grid)[[1, 2]]..., size(grid)[[1, 2]]...)
fill_send_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::North) = 
    _fill_north_send_buffer!(parent(c), buff.north, halo_size(grid)[[1, 2]]..., size(grid)[[1, 2]]...)
fill_send_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::Bottom) = nothing
fill_send_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::Top) = nothing

#####
##### Double sided fill_send_buffers!
#####

function fill_send_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::WestAndEast)
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

     _fill_west_send_buffer!(parent(c), buff.west,  Hx, Hy, Nx, Ny)
     _fill_east_send_buffer!(parent(c), buff.east,  Hx, Hy, Nx, Ny)
end

function fill_send_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::SouthAndNorth)
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

    _fill_south_send_buffer!(parent(c), buff.south, Hx, Hy, Nx, Ny)
    _fill_north_send_buffer!(parent(c), buff.north, Hx, Hy, Nx, Ny)
end

fill_send_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::BottomAndTop) = nothing

"""
    recv_from_buffers!(c::OffsetArray, buffers::CommunicationBuffers, grid)

fills OffsetArray `c` from `buffers.recv` after message passing occurred.
"""
function recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid)
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

     _recv_from_west_buffer!(parent(c), buff.west,  Hx, Hy, Nx, Ny)
     _recv_from_east_buffer!(parent(c), buff.east,  Hx, Hy, Nx, Ny)
    _recv_from_south_buffer!(parent(c), buff.south, Hx, Hy, Nx, Ny)
    _recv_from_north_buffer!(parent(c), buff.north, Hx, Hy, Nx, Ny)

   _recv_from_southwest_buffer!(parent(c), buff.southwest, Hx, Hy, Nx, Ny)
   _recv_from_southeast_buffer!(parent(c), buff.southeast, Hx, Hy, Nx, Ny)
   _recv_from_northwest_buffer!(parent(c), buff.northwest, Hx, Hy, Nx, Ny)
   _recv_from_northeast_buffer!(parent(c), buff.northeast, Hx, Hy, Nx, Ny)

   return nothing
end

function recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::Val{:corners})
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

   _recv_from_southwest_buffer!(parent(c), buff.southwest, Hx, Hy, Nx, Ny)
   _recv_from_southeast_buffer!(parent(c), buff.southeast, Hx, Hy, Nx, Ny)
   _recv_from_northwest_buffer!(parent(c), buff.northwest, Hx, Hy, Nx, Ny)
   _recv_from_northeast_buffer!(parent(c), buff.northeast, Hx, Hy, Nx, Ny)

   return nothing
end

#####
##### Single sided recv_from_buffers!
#####

recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::West) = 
    _recv_from_west_buffer!(parent(c), buff.west, halo_size(grid)[[1, 2]]..., size(grid)[[1, 2]]...)
recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::East) = 
    _recv_from_east_buffer!(parent(c), buff.east, halo_size(grid)[[1, 2]]..., size(grid)[[1, 2]]...)
recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::South) = 
    _recv_from_south_buffer!(parent(c), buff.south, halo_size(grid)[[1, 2]]..., size(grid)[[1, 2]]...)
recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::North) = 
    _recv_from_north_buffer!(parent(c), buff.north, halo_size(grid)[[1, 2]]..., size(grid)[[1, 2]]...)
recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::Bottom) = nothing
recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::Top) = nothing

#####
##### Double sided recv_from_buffers!
#####

function recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::WestAndEast)
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

    _recv_from_west_buffer!(parent(c), buff.west, Hx, Hy, Nx, Ny)
    _recv_from_east_buffer!(parent(c), buff.east, Hx, Hy, Nx, Ny)

    return nothing
end

function recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::SouthAndNorth)
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

   _recv_from_south_buffer!(parent(c), buff.south, Hx, Hy, Nx, Ny)
   _recv_from_north_buffer!(parent(c), buff.north, Hx, Hy, Nx, Ny)

   return nothing
end

recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::BottomAndTop) = nothing

#####
##### Individual _fill_send_buffers and _recv_from_buffer kernels
#####

for dir in (:west, :east, :south, :north, :southwest, :southeast, :northwest, :northeast)
    _fill_send_buffer! = Symbol(:_fill_, dir, :_send_buffer!)
    _recv_from_buffer! = Symbol(:_recv_from_, dir, :_buffer!)
    @eval $_fill_send_buffer!(c, ::Nothing, args...) = nothing
    @eval $_recv_from_buffer!(c, ::Nothing, args...) = nothing
end

#####
##### 1D Parallelizations (cover corners with 1 MPI pass)
#####

 _fill_west_send_buffer!(c, buff::OneDBuffer, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Hx:2Hx,   :, :)
 _fill_east_send_buffer!(c, buff::OneDBuffer, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Nx:Nx+Hx, :, :)
_fill_south_send_buffer!(c, buff::OneDBuffer, Hx, Hy, Nx, Ny) = buff.send .= view(c, :, 1+Hy:2Hy,  :)
_fill_north_send_buffer!(c, buff::OneDBuffer, Hx, Hy, Nx, Ny) = buff.send .= view(c, :, 1+Ny:Ny+Hy, :)

 _recv_from_west_buffer!(c, buff::OneDBuffer, Hx, Hy, Nx, Ny) = view(c, 1:Hx,           :,     :) .= buff.recv
 _recv_from_east_buffer!(c, buff::OneDBuffer, Hx, Hy, Nx, Ny) = view(c, 1+Nx+Hx:Nx+2Hx, :,     :) .= buff.recv
_recv_from_south_buffer!(c, buff::OneDBuffer, Hx, Hy, Nx, Ny) = view(c, :,     1:Hy,           :) .= buff.recv
_recv_from_north_buffer!(c, buff::OneDBuffer, Hx, Hy, Nx, Ny) = view(c, :,     1+Ny+Hy:Ny+2Hy, :) .= buff.recv

#####
##### 2D Parallelizations (explicitly send all corners)
#####

 _fill_west_send_buffer!(c, buff::TwoDBuffer, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Hx:2Hx,   1+Hy:Ny+Hy, :)
 _fill_east_send_buffer!(c, buff::TwoDBuffer, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Nx:Nx+Hx, 1+Hy:Ny+Hy, :)
_fill_south_send_buffer!(c, buff::TwoDBuffer, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Hx:Nx+Hx, 1+Hy:2Hy,  :)
_fill_north_send_buffer!(c, buff::TwoDBuffer, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Hx:Nx+Hx, 1+Ny:Ny+Hy, :)

 _recv_from_west_buffer!(c, buff::TwoDBuffer, Hx, Hy, Nx, Ny) = view(c, 1:Hx,           1+Hy:Ny+Hy,     :) .= buff.recv
 _recv_from_east_buffer!(c, buff::TwoDBuffer, Hx, Hy, Nx, Ny) = view(c, 1+Nx+Hx:Nx+2Hx, 1+Hy:Ny+Hy,     :) .= buff.recv
_recv_from_south_buffer!(c, buff::TwoDBuffer, Hx, Hy, Nx, Ny) = view(c, 1+Hx:Nx+Hx,     1:Hy,           :) .= buff.recv
_recv_from_north_buffer!(c, buff::TwoDBuffer, Hx, Hy, Nx, Ny) = view(c, 1+Hx:Nx+Hx,     1+Ny+Hy:Ny+2Hy, :) .= buff.recv

_fill_southwest_send_buffer!(c, buff::CornerBuffer, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Hx:2Hx,   1+Hy:2Hy,   :)
_fill_southeast_send_buffer!(c, buff::CornerBuffer, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Nx:Nx+Hx, 1+Hy:2Hy,   :)
_fill_northwest_send_buffer!(c, buff::CornerBuffer, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Hx:2Hx,   1+Ny:Ny+Hy, :)
_fill_northeast_send_buffer!(c, buff::CornerBuffer, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Nx:Nx+Hx, 1+Ny:Ny+Hy, :)

_recv_from_southwest_buffer!(c, buff::CornerBuffer, Hx, Hy, Nx, Ny) = view(c, 1:Hx,           1:Hy,           :) .= buff.recv
_recv_from_southeast_buffer!(c, buff::CornerBuffer, Hx, Hy, Nx, Ny) = view(c, 1+Nx+Hx:Nx+2Hx, 1:Hy,           :) .= buff.recv
_recv_from_northwest_buffer!(c, buff::CornerBuffer, Hx, Hy, Nx, Ny) = view(c, 1:Hx,           1+Ny+Hy:Ny+2Hy, :) .= buff.recv
_recv_from_northeast_buffer!(c, buff::CornerBuffer, Hx, Hy, Nx, Ny) = view(c, 1+Nx+Hx:Nx+2Hx, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv
