using Oceananigans.BoundaryConditions: CBC, HBC
using Oceananigans.Architectures: arch_array
using Oceananigans.Grids: halo_size

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

const PassingBC = Union{CBC, HBC}

create_buffer_x(arch, data, H, ::PassingBC) = (send = arch_array(arch, zeros(eltype(data), H, size(parent(data), 2), size(parent(data), 3))), 
                                               recv = arch_array(arch, zeros(eltype(data), H, size(parent(data), 2), size(parent(data), 3))))    
create_buffer_y(arch, data, H, ::PassingBC) = (send = arch_array(arch, zeros(eltype(data), size(parent(data), 1), H, size(parent(data), 3))), 
                                               recv = arch_array(arch, zeros(eltype(data), size(parent(data), 1), H, size(parent(data), 3))))

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
fill_send_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid, ::CPU) = nothing

function fill_send_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid, arch)

    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

     fill_west_send_buffer!(parent(c), buffers.west,  Hx, Nx)
     fill_east_send_buffer!(parent(c), buffers.east,  Hx, Nx)
    fill_south_send_buffer!(parent(c), buffers.south, Hy, Ny)
    fill_north_send_buffer!(parent(c), buffers.north, Hy, Ny)
end

"""
    fill_recv_buffers(c, buffers, arch)

fills OffsetArray `c` from `buffers.recv` after message passing occurred. If we are on CPU
we do not need to fill the buffers as the transfer can happen through views
"""
fill_recv_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid, ::CPU) = nothing

function fill_recv_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid, arch)

    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)

     fill_west_recv_buffer!(parent(c), buffers.west,  Hx, Nx)
     fill_east_recv_buffer!(parent(c), buffers.east,  Hx, Nx)
    fill_south_recv_buffer!(parent(c), buffers.south, Hy, Ny)
    fill_north_recv_buffer!(parent(c), buffers.north, Hy, Ny)
end

fill_west_send_buffer!(c, ::Nothing, args...) = nothing
fill_east_send_buffer!(c, ::Nothing, args...) = nothing
fill_west_recv_buffer!(c, ::Nothing, args...) = nothing
fill_east_recv_buffer!(c, ::Nothing, args...) = nothing

fill_north_send_buffer!(c, ::Nothing, args...) = nothing
fill_south_send_buffer!(c, ::Nothing, args...) = nothing
fill_north_recv_buffer!(c, ::Nothing, args...) = nothing
fill_south_recv_buffer!(c, ::Nothing, args...) = nothing

 fill_east_send_buffer!(c, buff, H, N) = buff.send .= c[1+H:2H,  :, :]
 fill_west_send_buffer!(c, buff, H, N) = buff.send .= c[1+N:N+H, :, :]
fill_north_send_buffer!(c, buff, H, N) = buff.send .= c[:, 1+H:2H,  :]
fill_south_send_buffer!(c, buff, H, N) = buff.send .= c[:, 1+N:N+H, :]

 fill_east_recv_buffer!(c, buff, H, N) = c[1:H,        :, :] .= buff.recv
 fill_west_recv_buffer!(c, buff, H, N) = c[1+N+H:N+2H, :, :] .= buff.recv
fill_north_recv_buffer!(c, buff, H, N) = c[:, 1:H,        :] .= buff.recv
fill_south_recv_buffer!(c, buff, H, N) = c[:, 1+N+H:N+2H, :] .= buff.recv
