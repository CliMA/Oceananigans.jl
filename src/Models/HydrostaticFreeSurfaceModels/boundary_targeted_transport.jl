using Oceananigans.BoundaryConditions: BoundaryCondition, FieldBoundaryConditions, NormalFlow, has_target_transport
using Oceananigans.Fields: Field

using Oceananigans.Architectures: architecture

using ..Models: apply_targeted_left_boundary_correction!, apply_targeted_right_boundary_correction!, bottom_transport,
                east_transport, get_bottom_area, get_east_area, get_north_area, get_south_area, get_top_area,
                get_west_area, initialize_side_target, north_transport, south_transport, top_transport,
                update_open_boundary_transport!, west_transport

const OBC = BoundaryCondition{<:NormalFlow}

#####
##### Helpers for non-Field velocity components
#####

# Only `Field`s carry `boundary_conditions`; `ZeroField`, `FunctionField`,
# `TimeSeriesInterpolation`, etc. do not — return `false` for those.
@inline function _is_targeted(velocity::Field, side)
    velocity.boundary_conditions isa FieldBoundaryConditions || return false
    bc = getproperty(velocity.boundary_conditions, side)
    return bc isa OBC && has_target_transport(bc.classification.scheme)
end
@inline _is_targeted(velocity, side) = false

# Safely retrieve the boundary condition (returns `nothing` for non-Field or MultiRegion types).
@inline _get_bc(velocity::Field, side) =
    velocity.boundary_conditions isa FieldBoundaryConditions ?
    getproperty(velocity.boundary_conditions, side) : nothing
@inline _get_bc(velocity, side) = nothing

#####
##### Initialize only targeted boundaries
#####

"""
$(TYPEDSIGNATURES)

Initialize boundary transport `Field`s only for boundaries with a `target_transport` set
on their scheme. Returns a `NamedTuple` of transport fields and boundary areas, or
`nothing` if no targeted boundaries are present.

Unlike the nonhydrostatic counterpart `initialize_boundary_transport`, no pool-correction
tracking is needed because `HydrostaticFreeSurfaceModel` lets the free surface accommodate
any net imbalance.
"""
function initialize_targeted_boundary_transport(velocities)
    u, v, w = velocities
    arch = architecture(u.grid)
    boundary_transports = NamedTuple()

    if _is_targeted(u, :west)
        boundary_transports = merge(boundary_transports, (; west_transport = west_transport(u),
                                                            west_area = get_west_area(u.grid; locally=true)))
    end

    if _is_targeted(u, :east)
        boundary_transports = merge(boundary_transports, (; east_transport = east_transport(u),
                                                            east_area = get_east_area(u.grid; locally=true)))
    end

    if _is_targeted(v, :south)
        boundary_transports = merge(boundary_transports, (; south_transport = south_transport(v),
                                                            south_area = get_south_area(v.grid; locally=true)))
    end

    if _is_targeted(v, :north)
        boundary_transports = merge(boundary_transports, (; north_transport = north_transport(v),
                                                            north_area = get_north_area(v.grid; locally=true)))
    end

    if _is_targeted(w, :bottom)
        boundary_transports = merge(boundary_transports, (; bottom_transport = bottom_transport(w),
                                                            bottom_area = get_bottom_area(w.grid; locally=true)))
    end

    if _is_targeted(w, :top)
        boundary_transports = merge(boundary_transports, (; top_transport = top_transport(w),
                                                            top_area = get_top_area(w.grid; locally=true)))
    end

    # Every rank joins the communicator split for each side, whether or not it holds that side
    sides = ((u, :west), (u, :east), (v, :south), (v, :north), (w, :bottom), (w, :top))

    for (velocity, side) in sides
        side_target = initialize_side_target(boundary_transports, _get_bc(velocity, side), Val(side), arch)
        boundary_transports = merge(boundary_transports, side_target)
    end

    return isempty(boundary_transports) ? nothing : boundary_transports
end

#####
##### Enforce targeted open boundary transports
#####

enforce_targeted_open_boundary_transport!(model, ::Nothing) = nothing

"""
$(TYPEDSIGNATURES)

Correct boundary velocities to achieve the prescribed `target_transport` for all open
boundaries whose scheme returns `true` from `has_target_transport`.

Unlike the nonhydrostatic version, no global pool correction is applied after the targeted
step: the free surface (η) is free to rise or fall to accommodate any net volume imbalance.
"""
function enforce_targeted_open_boundary_transport!(model, bt)
    u, v, w = model.velocities

    # Compute current fluxes through targeted boundaries.
    update_open_boundary_transport!(bt)

    # Apply per-boundary corrections (shared helpers — non-OBC / non-targeted return nothing).
    apply_targeted_left_boundary_correction!(u, _get_bc(u, :west),   Val(:west),   bt)
    apply_targeted_left_boundary_correction!(v, _get_bc(v, :south),  Val(:south),  bt)
    apply_targeted_left_boundary_correction!(w, _get_bc(w, :bottom), Val(:bottom), bt)
    apply_targeted_right_boundary_correction!(u, _get_bc(u, :east),  Val(:east),   bt)
    apply_targeted_right_boundary_correction!(v, _get_bc(v, :north), Val(:north),  bt)
    apply_targeted_right_boundary_correction!(w, _get_bc(w, :top),   Val(:top),    bt)

    return nothing
end
