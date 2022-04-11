using Oceananigans.Architectures: arch_array, device_event
using Oceananigans.Operators: assumed_field_location
using Oceananigans.Fields: reduced_dimensions

using Oceananigans.BoundaryConditions: 
            ContinuousBoundaryFunction, 
            DiscreteBoundaryFunction, 
            CBCT, 
            CBC

import Oceananigans.Fields: fill_halo_regions_field_tuple!, extract_field_bcs, extract_field_data

import Oceananigans.BoundaryConditions: 
            fill_halo_regions!,
            fill_west_and_east_halo!,
            fill_south_and_north_halo!,
            fill_bottom_and_top_halo!,
            fill_west_halo!, 
            fill_east_halo!, 
            fill_south_halo!,
            fill_north_halo!

@inline bc_str(::MultiRegionObject) = "MultiRegion Boundary Conditions"

@inline extract_field_buffers(field::Field)        = field.boundary_buffers
@inline extract_field_bcs(field::MultiRegionField) = field.boundary_conditions

@inline function fill_halo_regions_field_tuple!(full_fields, grid::MultiRegionGrid, args...; kwargs...) 
    for field in full_fields
        fill_halo_regions!(field, args...; kwargs...)
    end
end

# @inline fill_halo_regions_field_tuple!(full_fields, grid::MultiRegionGrid, args...; kwargs...) =
#     fill_halo_regions!(extract_field_data.(full_fields), extract_field_bcs.(full_fields), grid, args...; kwargs...)

function fill_halo_regions!(field::MultiRegionField, args...; kwargs...)
    reduced_dims = reduced_dimensions(field)

    return fill_halo_regions!(field.data,
                              field.boundary_conditions,
                              field.grid,
                              field.boundary_buffers,
                              args...;
                              reduced_dimensions = reduced_dims,
                              kwargs...)
end

fill_halo_regions!(c::MultiRegionObject, ::Nothing, args...; kwargs...) = nothing

fill_halo_regions!(c::MultiRegionObject, bcs, mrg::MultiRegionGrid, buffers, args...; kwargs...) =
    apply_regionally!(fill_halo_regions!, c, bcs, mrg, Reference(c.regions), Reference(buffers.regions), args...; kwargs...)

## Fill communicating boundary condition halos
for (lside, rside) in zip([:west, :south, :bottom], [:east, :north, :bottom])
    fill_both_halo! = Symbol(:fill_, lside, :_and_, rside, :_halo!)
    fill_left_halo!  = Symbol(:fill_, lside, :_halo!)
    fill_right_halo! = Symbol(:fill_, rside, :_halo!)

    @eval begin
        function $fill_both_halo!(c, left_bc::CBC, right_bc::CBC, arch, dep, grid, args...; kwargs...) 
             $fill_left_halo!(c,  left_bc, arch, dep, grid, args...; kwargs...)
            $fill_right_halo!(c, right_bc, arch, dep, grid, args...; kwargs...)
            return NoneEvent()
        end   
        function $fill_both_halo!(c, left_bc::CBC, right_bc, arch, dep, grid, args...; kwargs...) 
             $fill_left_halo!(c,  left_bc, arch, dep, grid, args...; kwargs...)
            $fill_right_halo!(c, right_bc, arch, dep, grid, args...; kwargs...)
            return NoneEvent()
        end   
        function $fill_both_halo!(c, left_bc, right_bc::CBC, arch, dep, grid, args...; kwargs...) 
             $fill_left_halo!(c,  left_bc, arch, dep, grid, args...; kwargs...)
            $fill_right_halo!(c, right_bc, arch, dep, grid, args...; kwargs...)
            return NoneEvent()
        end   
    end
end

function fill_west_halo!(c, bc::CBC, arch, dep, grid, neighbors, buffers, args...; kwargs...)
    
    wait(device(arch), dep)

    H = halo_size(grid)[1]
    N = size(grid)[1]
    w = neighbors[bc.condition.from_rank]
    dst = buffers[bc.condition.rank].west.recv
    
    switch_device!(getdevice(w))
    src = buffers[bc.condition.from_rank].east.send
    src .= view(parent(w), N+1:N+H, :, :)
    sync_device!(getdevice(w))

    switch_device!(getdevice(c))
    copyto!(dst, src)

    p  = view(parent(c), 1:H, :, :)
    p .= dst

    return nothing
end

function fill_east_halo!(c, bc::CBC, arch, dep, grid, neighbors, buffers, args...; kwargs...)

    wait(device(arch), dep)

    H = halo_size(grid)[1]
    N = size(grid)[1]
    e = neighbors[bc.condition.from_rank]
    dst = buffers[bc.condition.rank].east.recv
        
    switch_device!(getdevice(e))
    src = buffers[bc.condition.from_rank].west.send
    src .= view(parent(e), H+1:2H, :, :)
    sync_device!(getdevice(e))

    switch_device!(getdevice(c))    
    copyto!(dst, src)

    p  = view(parent(c), N+H+1:N+2H, :, :)
    p .= dst

    return nothing
end

function fill_south_halo!(c, bc::CBC, arch, dep, grid, neighbors, buffers, args...; kwargs...)
    
    wait(device(arch), dep)
    
    H = halo_size(grid)[2]
    N = size(grid)[2]
    s = neighbors[bc.condition.from_rank]
    dst = buffers[bc.condition.rank].south.recv
    
    switch_device!(getdevice(s))
    src = buffers[bc.condition.from_rank].north.send
    src .= view(parent(s), :, N+1:N+H, :)
    sync_device!(getdevice(s))

    switch_device!(getdevice(c))
    copyto!(dst, src)

    p  = view(parent(c), :, 1:H, :)
    p .= dst

    return nothing
end

function fill_north_halo!(c, bc::CBC, arch, dep, grid, neighbors, buffers, args...; kwargs...)
    
    wait(device(arch), dep)

    H = halo_size(grid)[2]
    N = size(grid)[2]
    n = neighbors[bc.condition.from_rank]
    dst = buffers[bc.condition.rank].north.recv
    
    switch_device!(getdevice(n))
    src = buffers[bc.condition.from_rank].south.send
    src .= view(parent(n), :, H+1:2H, :)
    sync_device!(getdevice(n))

    switch_device!(getdevice(c))    
    copyto!(dst, src)

    p  = view(parent(c), :, N+H+1:N+2H, :)
    p .= dst

    return nothing
end

## Tupled field boundary conditions for 

# function fill_west_halo!(c::NTuple, bc::NTuple{M, CBC}, arch, dep, grid, neighbors, buffers, args...; kwargs...) where M
    
#     ## Can we take this off??
#     wait(dep)
    
#     H = halo_size(grid)[1]
#     N = size(grid)[1]

#     dst = []
#     src = []
#     for n in M
#         push!(dst, buffers[n][bc[n].condition.rank].west...)
#         push!(src, buffers[n][bc[n].condition.from_rank].west...)
#     end
    
#     switch_device!(getdevice(neighbors[1][bc[1].condition.from_rank]))
    
#     @sync for n in 1:M
#         @async begin
#             w = neighbors[n][bc[n].condition.from_rank]
#             src[n] .= parent(w)[N+1:N+H, :, :]
#         end
#     end
#     sync_device!(getdevice(src[1]))

#     switch_device!(getdevice(c[1]))
#     copyto!(dst, src)

#     @sync for n in 1:M
#         @async begin
#             p  = view(parent(c[n]), 1:H, :, :)
#             p .= dst[n]
#         end
#     end

#     return nothing
# end

# function fill_east_halo!(c::NTuple, bc::NTuple{M, CBC}, arch, dep, grid, neighbors, args...; kwargs...) where M
    
#     ## Can we take this off??
#     wait(dep)
    
#     H = halo_size(grid)[1]
#     N = size(grid)[1]

#     dst = arch_array(arch, zeros(M, H, size(parent(c[1]), 2), size(parent(c[1]), 3)))

#     switch_device!(getdevice(neighbors[1][bc[1].condition.from_rank]))
#     src = arch_array(arch, zeros(M, H, size(parent(c[1]), 2), size(parent(c[1]), 3)))
    
#     @sync for n in 1:M
#         @async begin
#             e = neighbors[n][bc[n].condition.from_rank]
#             src[n, :, :, :] .= parent(e)[H+1:2H, :, :]
#         end
#     end

#     sync_device!(getdevice(src[1]))
    
#     switch_device!(getdevice(c[1]))
#     copyto!(dst, src)
#     @sync for n in 1:M
#         @async begin
#             p  = view(parent(c[n]),  N+H+1:N+2H, :, :)
#             p .= dst[n, :, :, :]
#         end
#     end

#     return nothing
# end

@inline @inbounds getregion(fc::FieldBoundaryConditions, i) = 
        FieldBoundaryConditions(getregion(fc.west, i), 
                                getregion(fc.east, i), 
                                getregion(fc.south, i), 
                                getregion(fc.north, i), 
                                getregion(fc.bottom, i),
                                getregion(fc.top, i),
                                fc.immersed)

@inline @inbounds getregion(bc::BoundaryCondition, i) = BoundaryCondition(bc.classification, getregion(bc.condition, i))

@inline @inbounds getregion(cf::ContinuousBoundaryFunction{X, Y, Z, I}, i) where {X, Y, Z, I} =
    ContinuousBoundaryFunction{X, Y, Z, I}(cf.func::F,
                                           getregion(cf.parameters, i),
                                           cf.field_dependencies,
                                           cf.field_dependencies_indices,
                                           cf.field_dependencies_interp)

@inline @inbounds getregion(df::DiscreteBoundaryFunction, i) =
    DiscreteBoundaryFunction(df.func, getregion(df.parameters, i))

# Everything goes for multi-region BC
validate_boundary_condition_location(::MultiRegionObject, ::Center, side)       = nothing 
validate_boundary_condition_location(::MultiRegionObject, ::Face, side)         = nothing 
validate_boundary_condition_topology(::MultiRegionObject, topo::Periodic, side) = nothing
validate_boundary_condition_topology(::MultiRegionObject, topo::Flat,     side) = nothing
