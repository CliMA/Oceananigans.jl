using Oceananigans: instantiated_location
using Oceananigans.Architectures: arch_array, device_event, device_copy_to!
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
                              field.indices,
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
# fill_halo_regions!(c::MultiRegionObject, bcs, loc, mrg::MultiRegionGrid, buffers, args...; kwargs...) = 
#     apply_regionally!(fill_halo_regions!, c, bcs, loc, mrg, Reference(c.regions), Reference(buffers.regions), args...; kwargs...)

# This results in too large parameter space required on the GPU for too many regions!! (we are passing the whole buffers and regions)
function fill_halo_regions!(c::MultiRegionObject, bcs, indices, loc, mrg::MultiRegionGrid, buffers, args...; kwargs...) 

    arch = architecture(mrg)

    halo_tuple = construct_regionally(permute_boundary_conditions, bcs)

    @apply_regionally fill_send_buffers!(c, buffers, grid, arch)

    for task = 1:3
        barrier = device_event(arch)
        apply_regionally!(fill_halo_event!, task, halo_tuple, 
                          c, indices, loc, arch, barrier, mrg, Reference(buffers.regions), 
                          args...; kwargs...)
    end

    @apply_regionally fill_recv_buffers!(c, buffers, grid, arch)

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
        function $fill_both_halo!(c, left_bc::CBC, right_bc, kernel_size, offset, loc, arch, dep, grid, args...; kwargs...) 
            event = $fill_right_halo!(c, right_bc, kernel_size, offset, loc, arch, dep, grid, args...; kwargs...)
            $fill_left_halo!(c, left_bc, kernel_size, offset, loc, arch, event, grid, args...; kwargs...)
            return NoneEvent()
        end   
        function $fill_both_halo!(c, left_bc, right_bc::CBC, kernel_size, offset, loc, arch, dep, grid, args...; kwargs...) 
            event = $fill_left_halo!(c, left_bc, kernel_size, offset, loc, arch, dep, grid, args...; kwargs...)
            $fill_right_halo!(c, right_bc, kernel_size, offset, loc, arch, event, grid, args...; kwargs...)
            return NoneEvent()
        end   
    end
end

function fill_west_and_east_halo!(c, westbc::CBC, eastbc::CBC, kernel_size, offset, loc, arch, dep, grid, buffers, args...; kwargs...)

    westdst = buffers[westbc.condition.rank].west.recv
    eastdst = buffers[eastbc.condition.rank].east.recv
    westsrc = buffers[westbc.condition.from_rank].east.send
    eastsrc = buffers[eastbc.condition.from_rank].west.send

    wait(Oceananigans.Architectures.device(arch), dep)
    
    switch_device!(getdevice(c))    
    device_copy_to!(westdst, westsrc)
    device_copy_to!(eastdst, eastsrc)

    return NoneEvent()
end

function fill_south_and_north_halo!(c, southbc::CBC, northbc::CBC, kernel_size, offset, loc, arch, dep, grid, buffers, args...; kwargs...)

    wait(Oceananigans.Architectures.device(arch), dep)

    southdst = buffers[southbc.condition.rank].south.recv
    northdst = buffers[northbc.condition.rank].north.recv
    southsrc = buffers[southbc.condition.from_rank].north.send
    northsrc = buffers[northbc.condition.from_rank].south.send

    switch_device!(getdevice(c))    
    device_copy_to!(southdst, southsrc)
    device_copy_to!(northdst, northsrc)

    return NoneEvent()
end

#####
##### Single fill_halo! for Communicating boundary condition 
#####
    
function fill_west_halo!(c, bc::CBC, kernel_size, offset, loc, arch, dep, grid, buffers, args...; kwargs...)

    wait(Oceananigans.Architectures.device(arch), dep)
    
    dst = buffers[bc.condition.rank].west.recv
    src = buffers[bc.condition.from_rank].east.send

    switch_device!(getdevice(c))
    device_copy_to!(dst, src)

    return nothing
end

function fill_east_halo!(c, bc::CBC, kernel_size, offset, loc, arch, dep, grid, buffers, args...; kwargs...)

    wait(Oceananigans.Architectures.device(arch), dep)

    dst = buffers[bc.condition.rank].east.recv
    src = buffers[bc.condition.from_rank].west.send

    switch_device!(getdevice(c))    
    device_copy_to!(dst, src)

    return nothing
end

function fill_south_halo!(c, bc::CBC, kernel_size, offset, loc, arch, dep, grid, buffers, args...; kwargs...)

    wait(Oceananigans.Architectures.device(arch), dep)

    dst = buffers[bc.condition.rank].south.recv
    src = buffers[bc.condition.from_rank].north.send

    switch_device!(getdevice(c))
    device_copy_to!(dst, src)

    return nothing
end

function fill_north_halo!(c, bc::CBC, kernel_size, offset, loc, arch, dep, grid, buffers, args...; kwargs...)

    wait(Oceananigans.Architectures.device(arch), dep)
    
    dst = buffers[bc.condition.rank].north.recv
    src = buffers[bc.condition.from_rank].south.send

    switch_device!(getdevice(c))    
    device_copy_to!(dst, src)

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
