using Oceananigans: instantiated_location
using Oceananigans.Architectures: on_architecture, device_copy_to!
using Oceananigans.Operators: assumed_field_location
using Oceananigans.Fields: reduced_dimensions
using Oceananigans.DistributedComputations: fill_send_buffers!

using Oceananigans.BoundaryConditions:
    ContinuousBoundaryFunction,
    DiscreteBoundaryFunction,
    fill_halo_event!, get_boundary_kernels,
    MultiRegionFillHalo,
    MCBCT, MCBC

import Oceananigans.BoundaryConditions: fill_halo_regions!, fill_halo_event!

@inline bc_str(::MultiRegionObject) = "MultiRegionObject"

@inline function fill_halo_regions!(fields::NamedTuple, grid::ConformalCubedSphereGridOfSomeKind, args...; kwargs...)
    u = haskey(fields, :u) ? fields.u : nothing
    v = haskey(fields, :v) ? fields.v : nothing

    if !isnothing(u) && !isnothing(v)
        fill_halo_regions!((u, v); kwargs...)
    end

    U = haskey(fields, :U) ? fields.U : nothing
    V = haskey(fields, :V) ? fields.V : nothing

    if !isnothing(U) && !isnothing(V)
        fill_halo_regions!((U, V); kwargs...)
    end

    other_keys = filter(k -> k != :u && k != :v && k != :U && k != :V, keys(fields))
    other_fields = Tuple(fields[k] for k in other_keys)

    for field in other_fields
        fill_halo_regions!(field; kwargs...)
    end

    return nothing
end

fill_halo_regions!(field::MultiRegionField, args...; kwargs...) = 
    fill_halo_regions!(field.data,
                       field.boundary_conditions,
                       field.indices,
                       instantiated_location(field),
                       field.grid,
                       field.communication_buffers,
                       args...;
                       kwargs...)

fill_halo_regions!(c::MultiRegionObject, ::Nothing, args...; kwargs...) = nothing

#####
##### fill_halo_regions! for a MultiRegionObject
#####

# Fill halo regions with a double pass for the moment.
# TODO: Optimize this some way (needs infra-regional-function synchronization between regions).
# The complication here is the possibility of different regions having different number of tasks,
# Which might happen, for example, for a grid that partitioned in a Bounded direction.
function fill_halo_regions!(c::MultiRegionObject, bcs, indices, loc, mrg::MultiRegionGrid, buffers, args...; fill_open_bcs=true, kwargs...)
    arch     = architecture(mrg)
    buff_ref = Reference(buffers.regional_objects)

    apply_regionally!(fill_send_buffers!, c, buffers, mrg)
    apply_regionally!(fill_halo_regions!, c, bcs, indices, loc, mrg, buff_ref, args...; fill_open_bcs, kwargs...)
    apply_regionally!(fill_send_buffers!, c, buffers, mrg)
    apply_regionally!(fill_halo_regions!, c, bcs, indices, loc, mrg, buff_ref, args...; fill_open_bcs, kwargs...)

    return nothing
end

#####
##### fill halo event, splat the args...
#####

fill_halo_event!(c, kernel!::MultiRegionFillHalo, bcs, loc, grid, buffers, args...; kwargs...) = kernel!(c, bcs..., loc, grid, buffers)

getside(x, ::North) = x.north
getside(x, ::South) = x.south
getside(x, ::West)  = x.west
getside(x, ::East)  = x.east

##### 
##### Double-sided MultiRegion filling kernels
#####

function (::MultiRegionFillHalo{<:WestAndEast})(c, westbc, eastbc, loc, grid, buffers)
    H = halo_size(grid)[1]
    N = size(grid)[1]

    westdst = buffers[westbc.condition.rank].west.recv
    eastdst = buffers[eastbc.condition.rank].east.recv

    westsrc = getside(buffers[westbc.condition.from_rank], westbc.condition.from_side).send
    eastsrc = getside(buffers[eastbc.condition.from_rank], eastbc.condition.from_side).send

    device_copy_to!(westdst, westsrc)
    device_copy_to!(eastdst, eastsrc)

    view(parent(c), 1:H, :, :) .= westdst
    view(parent(c), N+H+1:N+2H, :, :) .= eastdst

    return nothing
end

function (::MultiRegionFillHalo{<:SouthAndNorth})(c, southbc, northbc, loc, grid, buffers)
    H = halo_size(grid)[2]
    N = size(grid)[2]

    southdst = buffers[southbc.condition.rank].south.recv
    northdst = buffers[northbc.condition.rank].north.recv

    southsrc = getside(buffers[southbc.condition.from_rank], southbc.condition.from_side).send
    northsrc = getside(buffers[northbc.condition.from_rank], northbc.condition.from_side).send

    device_copy_to!(southdst, southsrc)
    device_copy_to!(northdst, northsrc)

    view(parent(c), :, 1:H, :) .= southdst
    view(parent(c), :, N+H+1:N+2H, :) .= northdst

    return nothing
end

#####
##### Single-sided MultiRegion filling kernels
#####

function (::MultiRegionFillHalo{<:West})(c, bc, loc, grid, buffers)
    H = halo_size(grid)[1]
    N = size(grid)[1]

    dst = buffers[bc.condition.rank].west.recv
    src = getside(buffers[bc.condition.from_rank], bc.condition.from_side).send

    device_copy_to!(dst, src)

    p  = view(parent(c), 1:H, :, :)
    p .= dst

    return nothing
end

function (::MultiRegionFillHalo{<:East})(c, bc, loc, grid, buffers)
    H = halo_size(grid)[1]
    N = size(grid)[1]

    dst = buffers[bc.condition.rank].east.recv
    src = getside(buffers[bc.condition.from_rank], bc.condition.from_side).send

    device_copy_to!(dst, src)

    p  = view(parent(c), N+H+1:N+2H, :, :)
    p .= dst

    return nothing
end

function (::MultiRegionFillHalo{<:South})(c, bc, loc, grid, buffers)
    H = halo_size(grid)[2]
    N = size(grid)[2]

    dst = buffers[bc.condition.rank].south.recv
    src = getside(buffers[bc.condition.from_rank], bc.condition.from_side).send

    device_copy_to!(dst, src)

    p  = view(parent(c), :, 1:H, :)
    p .= dst

    return nothing
end

function (::MultiRegionFillHalo{<:North})(c, bc, loc, grid, buffers)
    H = halo_size(grid)[2]
    N = size(grid)[2]

    dst = buffers[bc.condition.rank].north.recv
    src = getside(buffers[bc.condition.from_rank], bc.condition.from_side).send

    device_copy_to!(dst, src)

    p = view(parent(c), :, N+H+1:N+2H, :)
    p .= dst
    
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
                                    fc.immersed,
                                    fc.kernels,
                                    _getregion(fc.ordered_bcs, i))

@inline getregion(bc::BoundaryCondition, i) = BoundaryCondition(bc.classification, _getregion(bc.condition, i))

@inline getregion(cf::ContinuousBoundaryFunction{X, Y, Z, I}, i) where {X, Y, Z, I} =
            ContinuousBoundaryFunction{X, Y, Z, I}(cf.func,
                                                _getregion(cf.parameters, i),
                                                cf.field_dependencies,
                                                cf.field_dependencies_indices,
                                                cf.field_dependencies_interp)

@inline getregion(df::DiscreteBoundaryFunction, i) =
            DiscreteBoundaryFunction(_getregion(df.func, i), _getregion(df.parameters, i))

@inline _getregion(fc::FieldBoundaryConditions, i) =
            FieldBoundaryConditions(getregion(fc.west, i),
                                    getregion(fc.east, i),
                                    getregion(fc.south, i),
                                    getregion(fc.north, i),
                                    getregion(fc.bottom, i),
                                    getregion(fc.top, i),
                                    fc.immersed,
                                    fc.kernels,
                                    getregion(fc.ordered_bcs, i))

@inline _getregion(bc::BoundaryCondition, i) = BoundaryCondition(bc.classification, getregion(bc.condition, i))

@inline _getregion(cf::ContinuousBoundaryFunction{X, Y, Z, I}, i) where {X, Y, Z, I} =
            ContinuousBoundaryFunction{X, Y, Z, I}(cf.func,
                                                getregion(cf.parameters, i),
                                                cf.field_dependencies,
                                                cf.field_dependencies_indices,
                                                cf.field_dependencies_interp)

@inline _getregion(df::DiscreteBoundaryFunction, i) = DiscreteBoundaryFunction(getregion(df.func, i), getregion(df.parameters, i))

# Everything goes for multi-region BC
validate_boundary_condition_location(::MultiRegionObject, ::Center, side) = nothing
validate_boundary_condition_location(::MultiRegionObject, ::Face,   side) = nothing

validate_boundary_condition_topology(::MultiRegionObject, ::Periodic, side) = nothing
validate_boundary_condition_topology(::MultiRegionObject, ::Flat,     side) = nothing

 inject_west_boundary(connectivity, global_bc) = connectivity.west  === nothing ? global_bc : MultiRegionCommunicationBoundaryCondition(connectivity.west)
 inject_east_boundary(connectivity, global_bc) = connectivity.east  === nothing ? global_bc : MultiRegionCommunicationBoundaryCondition(connectivity.east)
inject_south_boundary(connectivity, global_bc) = connectivity.south === nothing ? global_bc : MultiRegionCommunicationBoundaryCondition(connectivity.south)
inject_north_boundary(connectivity, global_bc) = connectivity.north === nothing ? global_bc : MultiRegionCommunicationBoundaryCondition(connectivity.north)
