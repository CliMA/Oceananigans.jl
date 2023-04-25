using Oceananigans.BoundaryConditions: MCBC, DCBC
using Oceananigans.Architectures: arch_array
using Oceananigans.Grids: halo_size
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll

struct FieldBoundaryBuffers{W, E, S, N}
    west :: W
    east :: E
   south :: S
   north :: N
end

FieldBoundaryBuffers() = FieldBoundaryBuffers(nothing, nothing, nothing, nothing)
FieldBoundaryBuffers(grid, data, ::Missing) = nothing
FieldBoundaryBuffers(grid, data, ::Nothing) = nothing

function FieldBoundaryBuffers(grid, data, boundary_conditions)

    Hx, Hy, Hz = halo_size(grid)

    west  = create_buffer_x(architecture(grid), data, Hx, boundary_conditions.west)
    east  = create_buffer_x(architecture(grid), data, Hx, boundary_conditions.east)
    south = create_buffer_y(architecture(grid), data, Hy, boundary_conditions.south)
    north = create_buffer_y(architecture(grid), data, Hy, boundary_conditions.north)

    return FieldBoundaryBuffers(west, east, south, north)
end

create_buffer_x(arch, data, H, bc) = nothing
create_buffer_y(arch, data, H, bc) = nothing

using_buffered_communication(arch) = true

const PassingBC = Union{MCBC, DCBC}

function create_buffer_x(arch, data, H, ::PassingBC) 
    if !using_buffered_communication(arch)
        return nothing
    end
    return (send = arch_array(arch, zeros(eltype(data), H, size(parent(data), 2), size(parent(data), 3))), 
            recv = arch_array(arch, zeros(eltype(data), H, size(parent(data), 2), size(parent(data), 3))))    
end

function create_buffer_y(arch, data, H, ::PassingBC)
    if !using_buffered_communication(arch)
        return nothing
    end
    return (send = arch_array(arch, zeros(eltype(data), size(parent(data), 1), H, size(parent(data), 3))), 
            recv = arch_array(arch, zeros(eltype(data), size(parent(data), 1), H, size(parent(data), 3))))
end

Adapt.adapt_structure(to, buff::FieldBoundaryBuffers) =
    FieldBoundaryBuffers(Adapt.adapt(to, buff.west), 
                         Adapt.adapt(to, buff.east),    
                         Adapt.adapt(to, buff.north), 
                         Adapt.adapt(to, buff.south))

"""
    fill_send_buffers(c, buffers, arch)

fills `buffers.send` from OffsetArray `c` preparing for message passing. If we are on CPU
we do not need to fill the buffers as the transfer can happen through views
"""
function fill_west_and_east_send_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid)
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

    _fill_west_send_buffer!(parent(c), buffers.west, Hx, Nx)
    _fill_east_send_buffer!(parent(c), buffers.east, Hx, Nx)
end

function fill_south_and_north_send_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid)
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

    _fill_south_send_buffer!(parent(c), buffers.south, Hy, Ny)
    _fill_north_send_buffer!(parent(c), buffers.north, Hy, Ny)
end

fill_west_send_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid) = 
    _fill_west_send_buffer!(parent(c), buffers.west, halo_size(grid)[1], size(grid)[1])

fill_east_send_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid) = 
    _fill_east_send_buffer!(parent(c), buffers.east, halo_size(grid)[1], size(grid)[1])

fill_south_send_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid) = 
    _fill_south_send_buffer!(parent(c), buffers.south, halo_size(grid)[2], size(grid)[2])

fill_north_send_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid) = 
    _fill_north_send_buffer!(parent(c), buffers.north, halo_size(grid)[2], size(grid)[2])

"""
    recv_from_buffers(c, buffers, arch)

fills OffsetArray `c` from `buffers.recv` after message passing occurred. If we are on CPU
we do not need to fill the buffers as the transfer can happen through views
"""
function recv_from_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid)
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

     _recv_from_west_buffer!(parent(c), buffers.west,  Hx, Nx)
     _recv_from_east_buffer!(parent(c), buffers.east,  Hx, Nx)
    _recv_from_south_buffer!(parent(c), buffers.south, Hy, Ny)
    _recv_from_north_buffer!(parent(c), buffers.north, Hy, Ny)
end

function recv_from_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid, ::Val{:west_and_east})
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

    _recv_from_west_buffer!(parent(c), buffers.west, Hx, Nx)
    _recv_from_east_buffer!(parent(c), buffers.east, Hx, Nx)
end

function recv_from_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid, ::Val{:south_and_north})
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

   _recv_from_south_buffer!(parent(c), buffers.south, Hy, Ny)
   _recv_from_north_buffer!(parent(c), buffers.north, Hy, Ny)
end

recv_from_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid, ::Val{:bottom_and_top}) = nothing

 _fill_west_send_buffer!(c, ::Nothing, args...) = nothing
 _fill_east_send_buffer!(c, ::Nothing, args...) = nothing
_fill_north_send_buffer!(c, ::Nothing, args...) = nothing
_fill_south_send_buffer!(c, ::Nothing, args...) = nothing

 _recv_from_west_buffer!(c, ::Nothing, args...) = nothing
 _recv_from_east_buffer!(c, ::Nothing, args...) = nothing
_recv_from_north_buffer!(c, ::Nothing, args...) = nothing
_recv_from_south_buffer!(c, ::Nothing, args...) = nothing

 _fill_west_send_buffer!(c, buff, H, N) = buff.send .= view(c, 1+H:2H,  :, :)
 _fill_east_send_buffer!(c, buff, H, N) = buff.send .= view(c, 1+N:N+H, :, :)
_fill_south_send_buffer!(c, buff, H, N) = buff.send .= view(c, :, 1+H:2H,  :)
_fill_north_send_buffer!(c, buff, H, N) = buff.send .= view(c, :, 1+N:N+H, :)

 _recv_from_west_buffer!(c, buff, H, N) = view(c, 1:H,        :, :) .= buff.recv
 _recv_from_east_buffer!(c, buff, H, N) = view(c, 1+N+H:N+2H, :, :) .= buff.recv
_recv_from_south_buffer!(c, buff, H, N) = view(c, :, 1:H,        :) .= buff.recv
_recv_from_north_buffer!(c, buff, H, N) = view(c, :, 1+N+H:N+2H, :) .= buff.recv
