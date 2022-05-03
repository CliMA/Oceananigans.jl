using Oceananigans: instantiated_location
using Oceananigans.Architectures: arch_array, device_event
using Oceananigans.Operators: assumed_field_location
using Oceananigans.Fields: reduced_dimensions

using Oceananigans.BoundaryConditions: 
            ContinuousBoundaryFunction, 
            DiscreteBoundaryFunction, 
            permute_boundary_conditions,
            fill_halo_event!,
            CBCT, 
            CBC

import Oceananigans.Fields: tupled_fill_halo_regions!, boundary_conditions, data

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

@inline extract_field_buffers(field::Field)          = field.boundary_buffers
@inline boundary_conditions(field::MultiRegionField) = field.boundary_conditions

# This can be implemented once we have a buffer for field_tuples
@inline function tupled_fill_halo_regions!(full_fields, grid::MultiRegionGrid, args...; kwargs...) 
    for field in full_fields
        fill_halo_regions!(field, args...; kwargs...)
    end
end

function fill_halo_regions!(field::MultiRegionField, args...; kwargs...)
    reduced_dims = reduced_dimensions(field)

    return fill_halo_regions!(field.data,
                              field.boundary_conditions,
                              instantiated_location(field),
                              field.grid,
                              field.boundary_buffers,
                              args...;
                              reduced_dimensions = reduced_dims,
                              kwargs...)
end

fill_halo_regions!(c::MultiRegionObject, ::Nothing, args...; kwargs...) = nothing

#####
##### fill_halo_regions! for a MultiRegionObject
#####

# this can be used once we don't need to fill Value, Flux and Gradient anymore!
# fill_halo_regions!(c::MultiRegionObject, bcs, mrg::MultiRegionGrid, buffers, args...; kwargs...) = 
#     apply_regionally!(fill_halo_regions!, c, bcs, mrg, Reference(c.regions), Reference(buffers.regions), args...; kwargs...)

function fill_halo_regions!(c::MultiRegionObject, bcs, loc, mrg::MultiRegionGrid, buffers, args...; kwargs...) 

    arch = architecture(mrg)

    halo_tuple = construct_regionally(permute_boundary_conditions, bcs)
    
    for task = 1:3
        barrier = device_event(arch)
        apply_regionally!(fill_halo_event!, task, halo_tuple, 
                          c, loc, arch, barrier, mrg, Reference(c.regions), Reference(buffers.regions), 
                          args...; kwargs...)
    end

    return nothing
end

#####
##### fill_halo! for Communicating boundary condition 
#####
    
## Fill communicating boundary condition halos
for (lside, rside) in zip([:west, :south, :bottom], [:east, :north, :bottom])
    fill_both_halo! = Symbol(:fill_, lside, :_and_, rside, :_halo!)
    fill_left_halo!  = Symbol(:fill_, lside, :_halo!)
    fill_right_halo! = Symbol(:fill_, rside, :_halo!)

    @eval begin
        function $fill_both_halo!(c, left_bc::CBC, right_bc::CBC, loc, arch, dep, grid, args...; kwargs...) 
             $fill_left_halo!(c,  left_bc, arch, dep, grid, args...; kwargs...)
            $fill_right_halo!(c, right_bc, arch, dep, grid, args...; kwargs...)
            return NoneEvent()
        end   
        function $fill_both_halo!(c, left_bc::CBC, right_bc, loc, arch, dep, grid, args...; kwargs...) 
            event = $fill_right_halo!(c, right_bc, arch, dep, grid, args...; kwargs...)
            $fill_left_halo!(c,  left_bc, arch, event, grid, args...; kwargs...)
            return NoneEvent()
        end   
        function $fill_both_halo!(c, left_bc, right_bc::CBC, loc, arch, dep, grid, args...; kwargs...) 
            event = $fill_left_halo!(c, left_bc, arch, dep, grid, args...; kwargs...)
            $fill_right_halo!(c, right_bc, arch, event, grid, args...; kwargs...)
            return NoneEvent()
        end   
    end
end

#####
##### Single fill_halo! for Communicating boundary condition 
#####
    
function fill_west_halo!(c, bc::CBC, arch, dep, grid, neighbors, buffers, args...; kwargs...)
    
    H = halo_size(grid)[1]
    N = size(grid)[1]
    w = neighbors[bc.condition.from_rank]
    dst = buffers[bc.condition.rank].west.recv

    wait(device(arch), dep)

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

    H = halo_size(grid)[1]
    N = size(grid)[1]
    e = neighbors[bc.condition.from_rank]
    dst = buffers[bc.condition.rank].east.recv

    wait(device(arch), dep)

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
        
    H = halo_size(grid)[2]
    N = size(grid)[2]
    s = neighbors[bc.condition.from_rank]
    dst = buffers[bc.condition.rank].south.recv

    wait(device(arch), dep)

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
    
    H = halo_size(grid)[2]
    N = size(grid)[2]
    n = neighbors[bc.condition.from_rank]
    dst = buffers[bc.condition.rank].north.recv

    wait(device(arch), dep)

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

#####
##### Tupled fill_halo! for Communicating boundary condition 
#####
    
function fill_west_halo!(c::NTuple, bc::NTuple{M, CBC}, arch, dep, grid, neighbors, buffers, args...; kwargs...) where M
    
    ## Can we take this off??
    wait(dep)
    
    H = halo_size(grid)[1]
    N = size(grid)[1]

    dst = []
    src = []
    for n in M
        push!(dst, buffers[n][bc[n].condition.rank].west...)
        push!(src, buffers[n][bc[n].condition.from_rank].west...)
    end
    
    switch_device!(getdevice(neighbors[1][bc[1].condition.from_rank]))
    
    @sync for n in 1:M
        @async begin
            w = neighbors[n][bc[n].condition.from_rank]
            src[n] .= parent(w)[N+1:N+H, :, :]
        end
    end
    sync_device!(getdevice(src[1]))

    switch_device!(getdevice(c[1]))
    copyto!(dst, src)

    @sync for n in 1:M
        @async begin
            p  = view(parent(c[n]), 1:H, :, :)
            p .= dst[n]
        end
    end

    return nothing
end

function fill_east_halo!(c::NTuple, bc::NTuple{M, CBC}, arch, dep, grid, neighbors, args...; kwargs...) where M
    
    ## Can we take this off??
    wait(dep)
    
    H = halo_size(grid)[1]
    N = size(grid)[1]

    dst = arch_array(arch, zeros(M, H, size(parent(c[1]), 2), size(parent(c[1]), 3)))

    switch_device!(getdevice(neighbors[1][bc[1].condition.from_rank]))
    src = arch_array(arch, zeros(M, H, size(parent(c[1]), 2), size(parent(c[1]), 3)))
    
    @sync for n in 1:M
        @async begin
            e = neighbors[n][bc[n].condition.from_rank]
            src[n, :, :, :] .= parent(e)[H+1:2H, :, :]
        end
    end

    sync_device!(getdevice(src[1]))
    
    switch_device!(getdevice(c[1]))
    copyto!(dst, src)
    @sync for n in 1:M
        @async begin
            p  = view(parent(c[n]),  N+H+1:N+2H, :, :)
            p .= dst[n, :, :, :]
        end
    end

    return nothing
end

#####
##### MultiRegion boundary condition utils
#####

@inline getregion(fc::FieldBoundaryConditions, i) = 
        FieldBoundaryConditions(_getregion(fc.west, i), 
                                _getregion(fc.east, i), 
                                _getregion(fc.south, i), 
                                _getregion(fc.north, i), 
                                _getregion(fc.bottom, i),
                                _getregion(fc.top, i),
                                fc.immersed)

@inline getregion(bc::BoundaryCondition, i) = BoundaryCondition(bc.classification, _getregion(bc.condition, i))

@inline getregion(cf::ContinuousBoundaryFunction{X, Y, Z, I}, i) where {X, Y, Z, I} =
    ContinuousBoundaryFunction{X, Y, Z, I}(cf.func::F,
                                           _getregion(cf.parameters, i),
                                           cf.field_dependencies,
                                           cf.field_dependencies_indices,
                                           cf.field_dependencies_interp)

@inline getregion(df::DiscreteBoundaryFunction, i) =
    DiscreteBoundaryFunction(df.func, _getregion(df.parameters, i))


@inline _getregion(fc::FieldBoundaryConditions, i) = 
FieldBoundaryConditions(getregion(fc.west, i), 
                        getregion(fc.east, i), 
                        getregion(fc.south, i), 
                        getregion(fc.north, i), 
                        getregion(fc.bottom, i),
                        getregion(fc.top, i),
                        fc.immersed)

@inline _getregion(bc::BoundaryCondition, i) = BoundaryCondition(bc.classification, getregion(bc.condition, i))

@inline _getregion(cf::ContinuousBoundaryFunction{X, Y, Z, I}, i) where {X, Y, Z, I} =
ContinuousBoundaryFunction{X, Y, Z, I}(cf.func::F,
                                   getregion(cf.parameters, i),
                                   cf.field_dependencies,
                                   cf.field_dependencies_indices,
                                   cf.field_dependencies_interp)

@inline _getregion(df::DiscreteBoundaryFunction, i) =
DiscreteBoundaryFunction(df.func, getregion(df.parameters, i))

# Everything goes for multi-region BC
validate_boundary_condition_location(::MultiRegionObject, ::Center, side)       = nothing 
validate_boundary_condition_location(::MultiRegionObject, ::Face, side)         = nothing 
validate_boundary_condition_topology(::MultiRegionObject, topo::Periodic, side) = nothing
validate_boundary_condition_topology(::MultiRegionObject, topo::Flat,     side) = nothing
