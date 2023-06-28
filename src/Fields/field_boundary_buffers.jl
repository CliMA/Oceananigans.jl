using Oceananigans.BoundaryConditions: MCBC, DCBC
using Oceananigans.Architectures: arch_array
using Oceananigans.Grids: halo_size, size
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll

struct FieldBoundaryBuffers{W, E, S, N, SW, SE, NW, NE}
    west :: W
    east :: E
   south :: S
   north :: N
   southwest :: SW
   southeast :: SE
   northwest :: NW
   northeast :: NE
end

FieldBoundaryBuffers() = FieldBoundaryBuffers(nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
FieldBoundaryBuffers(grid, data, ::Missing) = nothing
FieldBoundaryBuffers(grid, data, ::Nothing) = nothing

function FieldBoundaryBuffers(grid, data, boundary_conditions)

    Hx, Hy, Hz = halo_size(grid)

    arch = architecture(grid)

    west  = create_buffer_x(architecture(grid), grid, data, Hx, boundary_conditions.west)
    east  = create_buffer_x(architecture(grid), grid, data, Hx, boundary_conditions.east)
    south = create_buffer_y(architecture(grid), grid, data, Hy, boundary_conditions.south)
    north = create_buffer_y(architecture(grid), grid, data, Hy, boundary_conditions.north)

    if hasproperty(arch, :connectivity)
        sw = create_buffer_corner(arch, grid, data, Hx, Hy, arch.connectivity.southwest)
        se = create_buffer_corner(arch, grid, data, Hx, Hy, arch.connectivity.southeast)
        nw = create_buffer_corner(arch, grid, data, Hx, Hy, arch.connectivity.northwest)
        ne = create_buffer_corner(arch, grid, data, Hx, Hy, arch.connectivity.northeast)
    else
        sw = nothing
        se = nothing
        nw = nothing
        ne = nothing
    end

    return FieldBoundaryBuffers(west, east, south, north, sw, se, nw, ne)
end

create_buffer_x(arch, grid, data, H, bc) = nothing
create_buffer_y(arch, grid, data, H, bc) = nothing

create_buffer_corner(arch, grid, data, Hx, Hy, ::Nothing) = nothing

function create_buffer_corner(arch, grid, data, Hx, Hy, side)
    return (send = arch_array(arch, zeros(eltype(data), Hx, Hy, size(parent(data), 3))), 
            recv = arch_array(arch, zeros(eltype(data), Hx, Hy, size(parent(data), 3))))    
end

function create_buffer_x(arch, grid, data, H, ::DCBC) 
    return (send = arch_array(arch, zeros(eltype(data), H, size(grid, 2), size(parent(data), 3))), 
            recv = arch_array(arch, zeros(eltype(data), H, size(grid, 2), size(parent(data), 3))))    
end

function create_buffer_y(arch, grid, data, H, ::DCBC)
    return (send = arch_array(arch, zeros(eltype(data), size(grid, 1), H, size(parent(data), 3))), 
            recv = arch_array(arch, zeros(eltype(data), size(grid, 1), H, size(parent(data), 3))))
end

create_buffer_x(arch, grid, data, H, ::MCBC) = 
           (send = arch_array(arch, zeros(eltype(data), H, size(parent(data), 2), size(parent(data), 3))), 
            recv = arch_array(arch, zeros(eltype(data), H, size(parent(data), 2), size(parent(data), 3))))    

create_buffer_y(arch, grid, data, H, ::MCBC) = 
           (send = arch_array(arch, zeros(eltype(data), size(parent(data), 1), H, size(parent(data), 3))), 
            recv = arch_array(arch, zeros(eltype(data), size(parent(data), 1), H, size(parent(data), 3))))

Adapt.adapt_structure(to, buff::FieldBoundaryBuffers) =
    FieldBoundaryBuffers(Adapt.adapt(to, buff.west), 
                         Adapt.adapt(to, buff.east),    
                         Adapt.adapt(to, buff.north), 
                         Adapt.adapt(to, buff.south), 
                         Adapt.adapt(to, buff.southwest), 
                         Adapt.adapt(to, buff.southeast), 
                         Adapt.adapt(to, buff.northwest), 
                         Adapt.adapt(to, buff.northeast))

"""
    fill_send_buffers(c, buffers, arch)

fills `buffers.send` from OffsetArray `c` preparing for message passing. 
"""
function fill_send_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid)
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

     _fill_west_send_buffer!(parent(c), buffers.west, Hx, Hy, Nx, Ny)
     _fill_east_send_buffer!(parent(c), buffers.east, Hx, Hy, Nx, Ny)
    _fill_south_send_buffer!(parent(c), buffers.south, Hx, Hy, Nx, Ny)
    _fill_north_send_buffer!(parent(c), buffers.north, Hx, Hy, Nx, Ny)

    _fill_southwest_send_buffer!(parent(c), buffers.southwest, Hx, Hy, Nx, Ny)
    _fill_southeast_send_buffer!(parent(c), buffers.southwest, Hx, Hy, Nx, Ny)
    _fill_northwest_send_buffer!(parent(c), buffers.southwest, Hx, Hy, Nx, Ny)
    _fill_northeast_send_buffer!(parent(c), buffers.southwest, Hx, Hy, Nx, Ny)

    return nothing
end

"""
    recv_from_buffers(c, buffers, arch)

fills OffsetArray `c` from `buffers.recv` after message passing occurred. 
"""
function recv_from_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid)
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

     _recv_from_west_buffer!(parent(c), buffers.west,  Hx, Hy, Nx, Ny)
     _recv_from_east_buffer!(parent(c), buffers.east,  Hx, Hy, Nx, Ny)
    _recv_from_south_buffer!(parent(c), buffers.south, Hx, Hy, Nx, Ny)
    _recv_from_north_buffer!(parent(c), buffers.north, Hx, Hy, Nx, Ny)
   
   _recv_from_southwest_buffer!(parent(c), buffers.southwest, Hx, Hy, Nx, Ny)
   _recv_from_southeast_buffer!(parent(c), buffers.southeast, Hx, Hy, Nx, Ny)
   _recv_from_northwest_buffer!(parent(c), buffers.northwest, Hx, Hy, Nx, Ny)
   _recv_from_northeast_buffer!(parent(c), buffers.northeast, Hx, Hy, Nx, Ny)

   return nothing
end

function recv_from_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid, ::Val{:corners})
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

   _recv_from_southwest_buffer!(parent(c), buffers.southwest, Hx, Hy, Nx, Ny)
   _recv_from_southeast_buffer!(parent(c), buffers.southeast, Hx, Hy, Nx, Ny)
   _recv_from_northwest_buffer!(parent(c), buffers.northwest, Hx, Hy, Nx, Ny)
   _recv_from_northeast_buffer!(parent(c), buffers.northeast, Hx, Hy, Nx, Ny)

   return nothing
end

function recv_from_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid, ::Val{:west_and_east})
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

    _recv_from_west_buffer!(parent(c), buffers.west, Hx, Hy, Nx, Ny)
    _recv_from_east_buffer!(parent(c), buffers.east, Hx, Hy, Nx, Ny)

    return nothing
end

function recv_from_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid, ::Val{:south_and_north})
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

   _recv_from_south_buffer!(parent(c), buffers.south, Hx, Hy, Nx, Ny)
   _recv_from_north_buffer!(parent(c), buffers.north, Hx, Hy, Nx, Ny)

   return nothing
end

recv_from_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid, ::Val{:bottom_and_top}) = nothing

#####
##### Individual _fill_send_buffers and _recv_from_buffer kernels
#####

 _fill_west_send_buffer!(c, ::Nothing, args...) = nothing
 _fill_east_send_buffer!(c, ::Nothing, args...) = nothing
_fill_north_send_buffer!(c, ::Nothing, args...) = nothing
_fill_south_send_buffer!(c, ::Nothing, args...) = nothing

_fill_southwest_send_buffer!(c, ::Nothing, args...) = nothing
_fill_southeast_send_buffer!(c, ::Nothing, args...) = nothing
_fill_northwest_send_buffer!(c, ::Nothing, args...) = nothing
_fill_northeast_send_buffer!(c, ::Nothing, args...) = nothing

 _recv_from_west_buffer!(c, ::Nothing, args...) = nothing
 _recv_from_east_buffer!(c, ::Nothing, args...) = nothing
_recv_from_north_buffer!(c, ::Nothing, args...) = nothing
_recv_from_south_buffer!(c, ::Nothing, args...) = nothing

_recv_from_southwest_buffer!(c, ::Nothing, args...) = nothing
_recv_from_southeast_buffer!(c, ::Nothing, args...) = nothing
_recv_from_northwest_buffer!(c, ::Nothing, args...) = nothing
_recv_from_northeast_buffer!(c, ::Nothing, args...) = nothing

 _fill_west_send_buffer!(c, buff, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Hx:2Hx,   1+Hy:Ny+Hy, :)
 _fill_east_send_buffer!(c, buff, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Nx:Nx+Hx, 1+Hy:Ny+Hy, :)
_fill_south_send_buffer!(c, buff, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Hx:Nx+Hx, 1+Hy:2Hy,  :)
_fill_north_send_buffer!(c, buff, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Hx:Nx+Hx, 1+Ny:Ny+Hy, :)

 _recv_from_west_buffer!(c, buff, Hx, Hy, Nx, Ny) = view(c, 1:Hx,           1+Hy:Ny+Hy,     :) .= buff.recv
 _recv_from_east_buffer!(c, buff, Hx, Hy, Nx, Ny) = view(c, 1+Nx+Hx:Nx+2Hx, 1+Hy:Ny+Hy,     :) .= buff.recv
_recv_from_south_buffer!(c, buff, Hx, Hy, Nx, Ny) = view(c, 1+Hx:Nx+Hx,     1:Hy,           :) .= buff.recv
_recv_from_north_buffer!(c, buff, Hx, Hy, Nx, Ny) = view(c, 1+Hx:Nx+Hx,     1+Ny+Hy:Ny+2Hy, :) .= buff.recv

_fill_southwest_send_buffer!(c, buff, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Hx:2Hx,   1+Hy:2Hy,   :)
_fill_southeast_send_buffer!(c, buff, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Nx:Nx+Hx, 1+Hy:2Hy,   :)
_fill_northwest_send_buffer!(c, buff, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Hx:2Hx,   1+Ny:Ny+Hy, :)
_fill_northeast_send_buffer!(c, buff, Hx, Hy, Nx, Ny) = buff.send .= view(c, 1+Nx:Nx+Hx, 1+Ny:Ny+Hy, :)

_recv_from_southwest_buffer!(c, buff, Hx, Hy, Nx, Ny) = view(c, 1:Hx,           1:Hy,           :) .= buff.recv
_recv_from_southeast_buffer!(c, buff, Hx, Hy, Nx, Ny) = view(c, 1+Nx+Hx:Nx+2Hx, 1:Hy,           :) .= buff.recv
_recv_from_northwest_buffer!(c, buff, Hx, Hy, Nx, Ny) = view(c, 1:Hx,           1+Ny+Hy:Ny+2Hy, :) .= buff.recv
_recv_from_northeast_buffer!(c, buff, Hx, Hy, Nx, Ny) = view(c, 1+Nx+Hx:Nx+2Hx, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv


