using Oceananigans: instantiated_location
using Oceananigans.Architectures: arch_array, device_copy_to!
using Oceananigans.Operators: assumed_field_location
using Oceananigans.Fields: reduced_dimensions

using Oceananigans.BoundaryConditions:
            ContinuousBoundaryFunction,
            DiscreteBoundaryFunction,
            permute_boundary_conditions,
            fill_halo_event!,
            MCBCT,
            MCBC

import Oceananigans.Fields: tupled_fill_halo_regions!, boundary_conditions, data, fill_send_buffers!

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
@inline function tupled_fill_halo_regions!(full_fields, grid::MultiRegionGrids, args...; kwargs...) 
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
#     apply_regionally!(fill_halo_regions!, c, bcs, loc, mrg, Reference(c.regional_objects), Reference(buffers.regional_objects), args...; kwargs...)

function fill_halo_regions!(c::MultiRegionObject, bcs, indices, loc, mrg::MultiRegionGrid, buffers, args...; kwargs...) 
    arch       = architecture(mrg)
    halo_tuple = construct_regionally(permute_boundary_conditions, bcs)

    for passes in 1:1
        for task in 1:3
            @apply_regionally fill_send_buffers!(c, buffers, mrg, halo_tuple, task)
            buff = Reference(buffers.regional_objects)
            apply_regionally!(fill_halo_event!, task, halo_tuple, 
                            c, indices, loc, arch, mrg, buff, 
                            args...; kwargs...)
        end
    end

    return nothing
end

# Find a better way to do this (this will not work for corners!!)
function fill_send_buffers!(c, buffers, grid, halo_tuple, task)
    bc_left  = halo_tuple[2][task]
    bc_right = halo_tuple[3][task]

    if bc_left isa MCBC || bc_right isa MCBC
        fill_send_buffers!(c, buffers, grid)
    end

    return nothing
end

#####
##### fill_halo! for Communicating boundary condition 
#####
    
## Fill communicating boundary condition halos
for (lside, rside) in zip([:west, :south, :bottom], [:east, :north, :top])
    fill_both_halo!  = Symbol(:fill_, lside, :_and_, rside, :_halo!)
    fill_left_halo!  = Symbol(:fill_, lside, :_halo!)
    fill_right_halo! = Symbol(:fill_, rside, :_halo!)

    @eval begin
        function $fill_both_halo!(c, left_bc::MCBC, right_bc, kernel_size, offset, loc, arch, grid, buffers, args...; kwargs...) 
            $fill_right_halo!(c, right_bc, kernel_size, offset, loc, arch, grid, args...; kwargs...)
            $fill_left_halo!(c, left_bc, kernel_size, offset, loc, arch, grid, buffers, args...; kwargs...)
            return nothing
        end

        function $fill_both_halo!(c, left_bc, right_bc::MCBC, kernel_size, offset, loc, arch, grid, buffers, args...; kwargs...) 
            $fill_left_halo!(c, left_bc, kernel_size, offset, loc, arch, grid, args...; kwargs...)
            $fill_right_halo!(c, right_bc, kernel_size, offset, loc, arch, grid, buffers, args...; kwargs...)
            return nothing
        end
    end
end

getside(x, ::North) = x.north
getside(x, ::South) = x.south
getside(x, ::West)  = x.west
getside(x, ::East)  = x.east

function fill_west_and_east_halo!(c, westbc::MCBC, eastbc::MCBC, kernel_size, offset, loc, arch, grid, buffers, args...; kwargs...)
    H = halo_size(grid)[1]
    N = size(grid)[1]

    westdst = buffers[westbc.condition.rank].west.recv
    eastdst = buffers[eastbc.condition.rank].east.recv

    westsrc = getside(buffers[westbc.condition.from_rank], westbc.condition.from_side).send
    eastsrc = getside(buffers[eastbc.condition.from_rank], eastbc.condition.from_side).send

    devicewest = getdevice(westsrc)
    deviceeast = getdevice(eastsrc)

    switch_device!(devicewest)
    westsrc = flip_west_and_east_indices(westsrc, loc[1], westbc.condition)

    switch_device!(deviceeast)
    eastsrc = flip_west_and_east_indices(eastsrc, loc[1], eastbc.condition)

    switch_device!(getdevice(c))
    device_copy_to!(westdst, westsrc)
    device_copy_to!(eastdst, eastsrc)

    if loc[2] == Face() && westbc.condition isa NonTrivialConnectivity
        Mx, My, _ = size(parent(c))
        view(parent(c), 1:H, 2:My, :) .= view(westdst, :, 1:My-1, :)
    else
        view(parent(c), 1:H, :, :) .= westdst
    end

    if loc[2] == Face() && eastbc.condition isa NonTrivialConnectivity
        Mx, My, _ = size(parent(c))
        view(parent(c), N+1+H:N+2H, 2:My, :) .= view(eastdst, :, 1:My-1, :)
    else
        view(parent(c), N+H+1:N+2H, :, :) .= eastdst
    end

    return nothing
end

function fill_south_and_north_halo!(c, southbc::MCBC, northbc::MCBC, kernel_size, offset, loc, arch, grid, buffers, args...; kwargs...)
    H = halo_size(grid)[2]
    N = size(grid)[2]

    southdst = buffers[southbc.condition.rank].south.recv
    northdst = buffers[northbc.condition.rank].north.recv

    southsrc = getside(buffers[southbc.condition.from_rank], southbc.condition.from_side).send
    northsrc = getside(buffers[northbc.condition.from_rank], northbc.condition.from_side).send

    devicesouth = getdevice(southsrc)
    devicenorth = getdevice(northsrc)

    switch_device!(devicesouth)
    southsrc = flip_south_and_north_indices(southsrc, loc[2], southbc.condition)

    switch_device!(devicenorth)
    northsrc = flip_south_and_north_indices(northsrc, loc[2], northbc.condition)

    switch_device!(getdevice(c))
    device_copy_to!(southdst, southsrc)
    device_copy_to!(northdst, northsrc)

    if loc[1] == Face() && southbc.condition isa NonTrivialConnectivity
        Mx, My, _ = size(parent(c))
        view(parent(c), 2:Mx, 1:H, :) .= view(southdst, 1:Mx-1, :, :)
    else
        view(parent(c), :, 1:H, :) .= southdst
    end

    if loc[1] == Face() && loc[2] == Face() && northbc.condition isa NonTrivialConnectivity
        Mx, My, _ = size(parent(c))
        view(parent(c), 2:Mx, N+H+1:N+2H, :) .= view(northdst, 1:Mx-1, :, :)
    elseif loc[1] == Face() && loc[2] == Center() && northbc.condition isa NonTrivialConnectivity
        Mx, My, _ = size(parent(c))
        view(parent(c), :, N+H+1:N+2H, :) .= view(northdst, :, :, :)
    else
        view(parent(c), :, N+H+1:N+2H, :) .= northdst
    end

    return nothing
end

#####
##### Single fill_halo! for Communicating boundary condition
#####

function fill_west_halo!(c, bc::MCBC, kernel_size, offset, loc, arch, grid, buffers, args...; kwargs...)
    H = halo_size(grid)[1]
    N = size(grid)[1]

    dst = buffers[bc.condition.rank].west.recv
    src = getside(buffers[bc.condition.from_rank], bc.condition.from_side).send

    dev = getdevice(src)
    switch_device!(dev)
    src = flip_west_and_east_indices(src, loc[1], bc.condition)

    switch_device!(getdevice(c))
    device_copy_to!(dst, src)

    p  = view(parent(c), 1:H, :, :)
    p .= dst

    return nothing
end

function fill_east_halo!(c, bc::MCBC, kernel_size, offset, loc, arch, grid, buffers, args...; kwargs...)
    H = halo_size(grid)[1]
    N = size(grid)[1]

    dst = buffers[bc.condition.rank].east.recv
    src = getside(buffers[bc.condition.from_rank], bc.condition.from_side).send

    dev = getdevice(src)
    switch_device!(dev)
    src = flip_west_and_east_indices(src, loc[1], bc.condition)

    switch_device!(getdevice(c))
    device_copy_to!(dst, src)

    p  = view(parent(c), N+H+1:N+2H, :, :)
    p .= dst

    return nothing
end

function fill_south_halo!(c, bc::MCBC, kernel_size, offset, loc, arch, grid, buffers, args...; kwargs...)        
    H = halo_size(grid)[2]
    N = size(grid)[2]

    dst = buffers[bc.condition.rank].south.recv
    src = getside(buffers[bc.condition.from_rank], bc.condition.from_side).send

    dev = getdevice(src)
    switch_device!(dev)
    src = flip_south_and_north_indices(src, loc[2], bc.condition)

    switch_device!(getdevice(c))
    device_copy_to!(dst, src)

    p  = view(parent(c), :, 1:H, :)
    p .= dst

    return nothing
end

function fill_north_halo!(c, bc::MCBC, kernel_size, offset, loc, arch, grid, buffers, args...; kwargs...)    
    H = halo_size(grid)[2]
    N = size(grid)[2]

    dst = buffers[bc.condition.rank].north.recv
    src = getside(buffers[bc.condition.from_rank], bc.condition.from_side).send

    dev = getdevice(src)
    switch_device!(dev)
    src = flip_south_and_north_indices(src, loc[2], bc.condition)

    switch_device!(getdevice(c))
    device_copy_to!(dst, src)

    p = view(parent(c), :, N+H+1:N+2H, :)

    if loc[1] == Center()
        p .= dst
    elseif loc[1] == Face()
        Mx, My, _ = size(p)
        view(p, 2:My, :, :) .= view(dst, 1:My-1, :, :)
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

@inline _getregion(df::DiscreteBoundaryFunction, i) = DiscreteBoundaryFunction(df.func, getregion(df.parameters, i))

# Everything goes for multi-region BC
validate_boundary_condition_location(::MultiRegionObject, ::Center, side) = nothing
validate_boundary_condition_location(::MultiRegionObject, ::Face,   side) = nothing

validate_boundary_condition_topology(::MultiRegionObject, ::Periodic, side) = nothing
validate_boundary_condition_topology(::MultiRegionObject, ::Flat,     side) = nothing

 inject_west_boundary(connectivity, global_bc) = connectivity.west  === nothing ? global_bc : MultiRegionCommunicationBoundaryCondition(connectivity.west)
 inject_east_boundary(connectivity, global_bc) = connectivity.east  === nothing ? global_bc : MultiRegionCommunicationBoundaryCondition(connectivity.east)
inject_south_boundary(connectivity, global_bc) = connectivity.south === nothing ? global_bc : MultiRegionCommunicationBoundaryCondition(connectivity.south)
inject_north_boundary(connectivity, global_bc) = connectivity.north === nothing ? global_bc : MultiRegionCommunicationBoundaryCondition(connectivity.north)
