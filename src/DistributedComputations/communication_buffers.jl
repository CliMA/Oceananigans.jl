using Oceananigans.BoundaryConditions
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, BoundaryCondition
using Oceananigans.BoundaryConditions: MultiRegionCommunication, DistributedCommunication
using Oceananigans.Grids: halo_size, size
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

communication_buffers(grid::DistributedGrid, data, boundary_conditions) = CommunicationBuffers(grid, data, boundary_conditions)

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

# OneDBuffers are associated with partitioning without corner passing,
# therefore the "corner zones" are communicated within the one-dimensional pass.
const OneDBuffers = CommunicationBuffers{<:Any, <:Any, <:Any, <:Any, <:Nothing, <:Nothing, <:Nothing, <:Nothing}

x_communication_buffer(arch, grid, data, H, bc) = nothing
y_communication_buffer(arch, grid, data, H, bc) = nothing

# Only used for `Distributed` architectures
corner_communication_buffer(arch, grid, data, Hx, Hy, edge1, edge2) = nothing

# Disambiguation
corner_communication_buffer(::Distributed, grid, data, Hx, Hy, ::Nothing, edge2) = nothing
corner_communication_buffer(::Distributed, grid, data, Hx, Hy, edge1, ::Nothing) = nothing
corner_communication_buffer(::Distributed, grid, data, Hx, Hy, ::Nothing, ::Nothing) = nothing

function corner_communication_buffer(arch::Distributed, grid, data, Hx, Hy, edge1, edge2)
    return (send = on_architecture(arch, zeros(eltype(data), Hx, Hy, size(parent(data), 3))), 
            recv = on_architecture(arch, zeros(eltype(data), Hx, Hy, size(parent(data), 3))))    
end

function x_communication_buffer(arch::Distributed, grid, data, H, ::DCBC) 
    # Either we pass corners or it is a 1D parallelization in x
    size_y = arch.ranks[2] == 1 ? size(parent(data), 2) : size(grid, 2)
    return (send = on_architecture(arch, zeros(eltype(data), H, size_y, size(parent(data), 3))),
            recv = on_architecture(arch, zeros(eltype(data), H, size_y, size(parent(data), 3))))
end

function y_communication_buffer(arch::Distributed, grid, data, H, ::DCBC)
    # Either we pass corners or it is a 1D parallelization in y
    size_x = arch.ranks[1] == 1 ? size(parent(data), 1) : size(grid, 1)
    return (send = on_architecture(arch, zeros(eltype(data), size_x, H, size(parent(data), 3))),
            recv = on_architecture(arch, zeros(eltype(data), size_x, H, size(parent(data), 3))))
end

x_communication_buffer(arch, grid, data, H, ::MCBC) = 
           (send = on_architecture(arch, zeros(eltype(data), H, size(parent(data), 2), size(parent(data), 3))), 
            recv = on_architecture(arch, zeros(eltype(data), H, size(parent(data), 2), size(parent(data), 3))))    

y_communication_buffer(arch, grid, data, H, ::MCBC) = 
           (send = on_architecture(arch, zeros(eltype(data), size(parent(data), 1), H, size(parent(data), 3))), 
            recv = on_architecture(arch, zeros(eltype(data), size(parent(data), 1), H, size(parent(data), 3))))

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

fill_send_buffers!(c::OffsetArray, ::Nothing, grid) = nothing

"""
    fill_send_buffers!(c::OffsetArray, buffers::CommunicationBuffers, grid)

fills `buffers.send` from OffsetArray `c` preparing for message passing.
"""
function fill_send_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid)
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

     _fill_west_send_buffer!(parent(c), buff, buff.west,  Hx, Hy, Nx, Ny)
     _fill_east_send_buffer!(parent(c), buff, buff.east,  Hx, Hy, Nx, Ny)
    _fill_south_send_buffer!(parent(c), buff, buff.south, Hx, Hy, Nx, Ny)
    _fill_north_send_buffer!(parent(c), buff, buff.north, Hx, Hy, Nx, Ny)

    _fill_southwest_send_buffer!(parent(c), buff, buff.southwest, Hx, Hy, Nx, Ny)
    _fill_southeast_send_buffer!(parent(c), buff, buff.southeast, Hx, Hy, Nx, Ny)
    _fill_northwest_send_buffer!(parent(c), buff, buff.northwest, Hx, Hy, Nx, Ny)
    _fill_northeast_send_buffer!(parent(c), buff, buff.northeast, Hx, Hy, Nx, Ny)

    return nothing
end

fill_send_buffers!(c::OffsetArray, ::Nothing, grid, ::Val{:corners}) = nothing

function fill_send_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::Val{:corners})
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

    _fill_southwest_send_buffer!(parent(c), buff, buff.southwest, Hx, Hy, Nx, Ny)
    _fill_southeast_send_buffer!(parent(c), buff, buff.southeast, Hx, Hy, Nx, Ny)
    _fill_northwest_send_buffer!(parent(c), buff, buff.northwest, Hx, Hy, Nx, Ny)
    _fill_northeast_send_buffer!(parent(c), buff, buff.northeast, Hx, Hy, Nx, Ny)

    return nothing
end

#####
##### Single sided fill_send_buffers!
#####

fill_send_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::West) = 
    _fill_west_send_buffer!(parent(c), buff, buff.west, halo_size(grid)[[1, 2]]..., size(grid)[[1, 2]]...)
fill_send_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::East) = 
    _fill_east_send_buffer!(parent(c), buff, buff.east, halo_size(grid)[[1, 2]]..., size(grid)[[1, 2]]...)
fill_send_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::South) = 
    _fill_south_send_buffer!(parent(c), buff, buff.south, halo_size(grid)[[1, 2]]..., size(grid)[[1, 2]]...)
fill_send_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::North) = 
    _fill_north_send_buffer!(parent(c), buff, buff.north, halo_size(grid)[[1, 2]]..., size(grid)[[1, 2]]...)
fill_send_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::Bottom) = nothing
fill_send_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::Top) = nothing

#####
##### Double sided fill_send_buffers!
#####

function fill_send_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::WestAndEast)
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

     _fill_west_send_buffer!(parent(c), buff, buff.west,  Hx, Hy, Nx, Ny)
     _fill_east_send_buffer!(parent(c), buff, buff.east,  Hx, Hy, Nx, Ny)
end

function fill_send_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::SouthAndNorth)
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

    _fill_south_send_buffer!(parent(c), buff, buff.south, Hx, Hy, Nx, Ny)
    _fill_north_send_buffer!(parent(c), buff, buff.north, Hx, Hy, Nx, Ny)
end

fill_send_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::BottomAndTop) = nothing

"""
    recv_from_buffers!(c::OffsetArray, buffers::CommunicationBuffers, grid)

fills OffsetArray `c` from `buffers.recv` after message passing occurred.
"""
function recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid)
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

     _recv_from_west_buffer!(parent(c), buff, buff.west,  Hx, Hy, Nx, Ny)
     _recv_from_east_buffer!(parent(c), buff, buff.east,  Hx, Hy, Nx, Ny)
    _recv_from_south_buffer!(parent(c), buff, buff.south, Hx, Hy, Nx, Ny)
    _recv_from_north_buffer!(parent(c), buff, buff.north, Hx, Hy, Nx, Ny)

   _recv_from_southwest_buffer!(parent(c), buff, buff.southwest, Hx, Hy, Nx, Ny)
   _recv_from_southeast_buffer!(parent(c), buff, buff.southeast, Hx, Hy, Nx, Ny)
   _recv_from_northwest_buffer!(parent(c), buff, buff.northwest, Hx, Hy, Nx, Ny)
   _recv_from_northeast_buffer!(parent(c), buff, buff.northeast, Hx, Hy, Nx, Ny)

   return nothing
end

function recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::Val{:corners})
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

   _recv_from_southwest_buffer!(parent(c), buff, buff.southwest, Hx, Hy, Nx, Ny)
   _recv_from_southeast_buffer!(parent(c), buff, buff.southeast, Hx, Hy, Nx, Ny)
   _recv_from_northwest_buffer!(parent(c), buff, buff.northwest, Hx, Hy, Nx, Ny)
   _recv_from_northeast_buffer!(parent(c), buff, buff.northeast, Hx, Hy, Nx, Ny)

   return nothing
end

#####
##### Single sided recv_from_buffers!
#####

recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::West) = 
    _recv_from_west_buffer!(parent(c), buff, buff.west, halo_size(grid)[[1, 2]]..., size(grid)[[1, 2]]...)
recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::East) = 
    _recv_from_east_buffer!(parent(c), buff, buff.east, halo_size(grid)[[1, 2]]..., size(grid)[[1, 2]]...)
recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::South) = 
    _recv_from_south_buffer!(parent(c), buff, buff.south, halo_size(grid)[[1, 2]]..., size(grid)[[1, 2]]...)
recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::North) = 
    _recv_from_north_buffer!(parent(c), buff, buff.north, halo_size(grid)[[1, 2]]..., size(grid)[[1, 2]]...)
recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::Bottom) = nothing
recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::Top) = nothing

#####
##### Double sided recv_from_buffers!
#####

function recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::WestAndEast)
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

    _recv_from_west_buffer!(parent(c), buff, buff.west, Hx, Hy, Nx, Ny)
    _recv_from_east_buffer!(parent(c), buff, buff.east, Hx, Hy, Nx, Ny)

    return nothing
end

function recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::SouthAndNorth)
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

   _recv_from_south_buffer!(parent(c), buff, buff.south, Hx, Hy, Nx, Ny)
   _recv_from_north_buffer!(parent(c), buff, buff.north, Hx, Hy, Nx, Ny)

   return nothing
end

recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers, grid, ::BottomAndTop) = nothing

#####
##### Individual _fill_send_buffers and _recv_from_buffer kernels
#####

for dir in (:west, :east, :south, :north, :southwest, :southeast, :northwest, :northeast)
    _fill_send_buffer! = Symbol(:_fill_, dir, :_send_buffer!)
    _recv_from_buffer! = Symbol(:_recv_from_, dir, :_buffer!)

    @eval $_fill_send_buffer!(c, b, ::Nothing, args...) = nothing
    @eval $_recv_from_buffer!(c, b, ::Nothing, args...) = nothing
    @eval $_fill_send_buffer!(c, ::OneDBuffers, ::Nothing, args...) = nothing
    @eval $_recv_from_buffer!(c, ::OneDBuffers, ::Nothing, args...) = nothing
end

#####
##### 1D Parallelizations (cover corners with 1 MPI pass)
#####

 _fill_west_send_buffer!(c, ::OneDBuffers, buff, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Hx:2Hx,   :, :)
 _fill_east_send_buffer!(c, ::OneDBuffers, buff, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Nx:Nx+Hx, :, :)
_fill_south_send_buffer!(c, ::OneDBuffers, buff, Hx, Hy, Nx, Ny) = buff.send .= view(c, :, 1+Hy:2Hy,  :)
_fill_north_send_buffer!(c, ::OneDBuffers, buff, Hx, Hy, Nx, Ny) = buff.send .= view(c, :, 1+Ny:Ny+Hy, :)

 _recv_from_west_buffer!(c, ::OneDBuffers, buff, Hx, Hy, Nx, Ny) = view(c, 1:Hx,           :,     :) .= buff.recv
 _recv_from_east_buffer!(c, ::OneDBuffers, buff, Hx, Hy, Nx, Ny) = view(c, 1+Nx+Hx:Nx+2Hx, :,     :) .= buff.recv
_recv_from_south_buffer!(c, ::OneDBuffers, buff, Hx, Hy, Nx, Ny) = view(c, :,     1:Hy,           :) .= buff.recv
_recv_from_north_buffer!(c, ::OneDBuffers, buff, Hx, Hy, Nx, Ny) = view(c, :,     1+Ny+Hy:Ny+2Hy, :) .= buff.recv

#####
##### 2D Parallelizations (explicitly send corners)
#####

 _fill_west_send_buffer!(c, b, buff, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Hx:2Hx,   1+Hy:Ny+Hy, :)
 _fill_east_send_buffer!(c, b, buff, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Nx:Nx+Hx, 1+Hy:Ny+Hy, :)
_fill_south_send_buffer!(c, b, buff, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Hx:Nx+Hx, 1+Hy:2Hy,  :)
_fill_north_send_buffer!(c, b, buff, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Hx:Nx+Hx, 1+Ny:Ny+Hy, :)

 _recv_from_west_buffer!(c, b, buff, Hx, Hy, Nx, Ny) = view(c, 1:Hx,           1+Hy:Ny+Hy,     :) .= buff.recv
 _recv_from_east_buffer!(c, b, buff, Hx, Hy, Nx, Ny) = view(c, 1+Nx+Hx:Nx+2Hx, 1+Hy:Ny+Hy,     :) .= buff.recv
_recv_from_south_buffer!(c, b, buff, Hx, Hy, Nx, Ny) = view(c, 1+Hx:Nx+Hx,     1:Hy,           :) .= buff.recv
_recv_from_north_buffer!(c, b, buff, Hx, Hy, Nx, Ny) = view(c, 1+Hx:Nx+Hx,     1+Ny+Hy:Ny+2Hy, :) .= buff.recv

_fill_southwest_send_buffer!(c, b, buff, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Hx:2Hx,   1+Hy:2Hy,   :)
_fill_southeast_send_buffer!(c, b, buff, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Nx:Nx+Hx, 1+Hy:2Hy,   :)
_fill_northwest_send_buffer!(c, b, buff, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Hx:2Hx,   1+Ny:Ny+Hy, :)
_fill_northeast_send_buffer!(c, b, buff, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Nx:Nx+Hx, 1+Ny:Ny+Hy, :)

_recv_from_southwest_buffer!(c, b, buff, Hx, Hy, Nx, Ny) = view(c, 1:Hx,           1:Hy,           :) .= buff.recv
_recv_from_southeast_buffer!(c, b, buff, Hx, Hy, Nx, Ny) = view(c, 1+Nx+Hx:Nx+2Hx, 1:Hy,           :) .= buff.recv
_recv_from_northwest_buffer!(c, b, buff, Hx, Hy, Nx, Ny) = view(c, 1:Hx,           1+Ny+Hy:Ny+2Hy, :) .= buff.recv
_recv_from_northeast_buffer!(c, b, buff, Hx, Hy, Nx, Ny) = view(c, 1+Nx+Hx:Nx+2Hx, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv

