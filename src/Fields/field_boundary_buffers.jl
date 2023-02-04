using Oceananigans.BoundaryConditions: CBC, HBC
using Oceananigans.Architectures: arch_array
using Oceananigans.Grids: halo_size
using Oceananigans.Utils: launch!
using KernelAbstractions: MultiEvent, @kernel, @index
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
    arch = architecture(grid)
    H    = halo_size(grid)
    N    = size(grid)

     west_event =  fill_west_send_buffer!(parent(c), buffers.west,  H, N, arch, grid)
     east_event =  fill_east_send_buffer!(parent(c), buffers.east,  H, N, arch, grid)
    south_event = fill_south_send_buffer!(parent(c), buffers.south, H, N, arch, grid)
    north_event = fill_north_send_buffer!(parent(c), buffers.north, H, N, arch, grid)

    wait(device(arch), MultiEvent((west_event, east_event, south_event, north_event)))
end

"""
    fill_recv_buffers(c, buffers, arch)

fills OffsetArray `c` from `buffers.recv` after message passing occurred. If we are on CPU
we do not need to fill the buffers as the transfer can happen through views
"""
fill_recv_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid, ::CPU) = nothing

function fill_recv_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid, arch)

    arch = architecture(grid)
    H    = halo_size(grid)
    N    = size(grid)

     west_event =  fill_west_recv_buffer!(parent(c), buffers.west,  H, N, arch, grid)
     east_event =  fill_east_recv_buffer!(parent(c), buffers.east,  H, N, arch, grid)
    south_event = fill_south_recv_buffer!(parent(c), buffers.south, H, N, arch, grid)
    north_event = fill_north_recv_buffer!(parent(c), buffers.north, H, N, arch, grid)

    wait(device(arch), MultiEvent((west_event, east_event, south_event, north_event)))
end

fill_west_send_buffer!(c, ::Nothing, args...) = nothing
fill_east_send_buffer!(c, ::Nothing, args...) = nothing
fill_west_recv_buffer!(c, ::Nothing, args...) = nothing
fill_east_recv_buffer!(c, ::Nothing, args...) = nothing

fill_north_send_buffer!(c, ::Nothing, args...) = nothing
fill_south_send_buffer!(c, ::Nothing, args...) = nothing
fill_north_recv_buffer!(c, ::Nothing, args...) = nothing
fill_south_recv_buffer!(c, ::Nothing, args...) = nothing

 fill_west_send_buffer!(c, b, H, N, arch, grid) = launch!(arch, grid, size(c)[[2, 3]],  _fill_west_send_buffer!, c, b.send, H[1], N[1])
 fill_east_send_buffer!(c, b, H, N, arch, grid) = launch!(arch, grid, size(c)[[2, 3]],  _fill_east_send_buffer!, c, b.send, H[1], N[1])
fill_north_send_buffer!(c, b, H, N, arch, grid) = launch!(arch, grid, size(c)[[1, 3]], _fill_north_send_buffer!, c, b.send, H[2], N[2])
fill_south_send_buffer!(c, b, H, N, arch, grid) = launch!(arch, grid, size(c)[[1, 3]], _fill_south_send_buffer!, c, b.send, H[2], N[2])

 fill_west_recv_buffer!(c, b, H, N, arch, grid) = launch!(arch, grid, size(c)[[2, 3]],  _fill_west_recv_buffer!, c, b.recv, H[1], N[1])
 fill_east_recv_buffer!(c, b, H, N, arch, grid) = launch!(arch, grid, size(c)[[2, 3]],  _fill_east_recv_buffer!, c, b.recv, H[1], N[1])
fill_north_recv_buffer!(c, b, H, N, arch, grid) = launch!(arch, grid, size(c)[[1, 3]], _fill_north_recv_buffer!, c, b.recv, H[2], N[2])
fill_south_recv_buffer!(c, b, H, N, arch, grid) = launch!(arch, grid, size(c)[[1, 3]], _fill_south_recv_buffer!, c, b.recv, H[2], N[2])

@kernel function _fill_west_send_buffer!(c, b, H, N)
    j, k = @index(Global, NTuple)
    @unroll for i in 1:H
        b[i, j, k] = c[i+H, j, k]
    end
end

@kernel function _fill_east_send_buffer!(c, b, H, N)
    j, k = @index(Global, NTuple)
    @unroll for i in 1:H
        b[i, j, k] = c[i+N, j, k]
    end
end

@kernel function _fill_south_send_buffer!(c, b, H, N)
    i, k = @index(Global, NTuple)
    @unroll for j in 1:H
        b[i, j, k] = c[i, j+H, k]
    end
end

@kernel function _fill_north_send_buffer!(c, b, H, N)
    i, k = @index(Global, NTuple)
    @unroll for j in 1:H
        b[i, j, k] = c[i, j+N, k]
    end
end

@kernel function _fill_west_recv_buffer!(c, b, H, N)
    j, k = @index(Global, NTuple)
    @unroll for i in 1:H
        c[i, j, k] = b[i, j, k]
    end
end

@kernel function _fill_east_recv_buffer!(c, b, H, N)
    j, k = @index(Global, NTuple)
    @unroll for i in 1:H
        c[i+N+H, j, k] = b[i, j, k]
    end
end

@kernel function _fill_south_recv_buffer!(c, b, H, N)
    i, k = @index(Global, NTuple)
    @unroll for j in 1:H
        c[i, j, k] = b[i, j, k]
    end
end

@kernel function _fill_north_recv_buffer!(c, b, H, N)
    i, k = @index(Global, NTuple)
    @unroll for j in 1:H
        c[i, j+N+H, k] = b[i, j, k]
    end
end

#  fill_west_send_buffer!(c, buff, H, N) = buff.send .= c[1+H:2H,  :, :]
#  fill_east_send_buffer!(c, buff, H, N) = buff.send .= c[1+N:N+H, :, :]
# fill_south_send_buffer!(c, buff, H, N) = buff.send .= c[:, 1+H:2H,  :]
# fill_north_send_buffer!(c, buff, H, N) = buff.send .= c[:, 1+N:N+H, :]

#  fill_west_recv_buffer!(c, buff, H, N) = c[1:H,        :, :] .= buff.recv
#  fill_east_recv_buffer!(c, buff, H, N) = c[1+N+H:N+2H, :, :] .= buff.recv
# fill_south_recv_buffer!(c, buff, H, N) = c[:, 1:H,        :] .= buff.recv
# fill_north_recv_buffer!(c, buff, H, N) = c[:, 1+N+H:N+2H, :] .= buff.recv
