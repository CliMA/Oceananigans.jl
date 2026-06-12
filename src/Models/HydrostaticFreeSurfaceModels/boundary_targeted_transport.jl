using Oceananigans.BoundaryConditions: BoundaryCondition, NormalFlow, has_target_transport, FieldBoundaryConditions
using Oceananigans.Fields: Field
using ..Models: west_transport, east_transport, south_transport,
                north_transport, bottom_transport, top_transport,
                get_west_area, get_east_area, get_south_area,
                get_north_area, get_bottom_area, get_top_area,
                update_open_boundary_transport!,
                apply_targeted_left_boundary_correction!,
                apply_targeted_right_boundary_correction!

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
    initialize_targeted_boundary_transport(velocities)

Initialize boundary transport `Field`s only for boundaries with a `target_transport` set
on their scheme. Returns a `NamedTuple` of transport fields and boundary areas, or
`nothing` if no targeted boundaries are present.

Unlike the nonhydrostatic counterpart `initialize_boundary_transport`, no pool-correction
tracking is needed because `HydrostaticFreeSurfaceModel` lets the free surface accommodate
any net imbalance.
"""
function initialize_targeted_boundary_transport(velocities)
    u, v, w = velocities
    boundary_transports = NamedTuple()
    has_targeted = false

    if _is_targeted(u, :west)
        boundary_transports = merge(boundary_transports, (; west_transport = west_transport(u),
                                                            west_area = get_west_area(u.grid)))
        has_targeted = true
    end

    if _is_targeted(u, :east)
        boundary_transports = merge(boundary_transports, (; east_transport = east_transport(u),
                                                            east_area = get_east_area(u.grid)))
        has_targeted = true
    end

    if _is_targeted(v, :south)
        boundary_transports = merge(boundary_transports, (; south_transport = south_transport(v),
                                                            south_area = get_south_area(v.grid)))
        has_targeted = true
    end

    if _is_targeted(v, :north)
        boundary_transports = merge(boundary_transports, (; north_transport = north_transport(v),
                                                            north_area = get_north_area(v.grid)))
        has_targeted = true
    end

    if _is_targeted(w, :bottom)
        boundary_transports = merge(boundary_transports, (; bottom_transport = bottom_transport(w),
                                                            bottom_area = get_bottom_area(w.grid)))
        has_targeted = true
    end

    if _is_targeted(w, :top)
        boundary_transports = merge(boundary_transports, (; top_transport = top_transport(w),
                                                            top_area = get_top_area(w.grid)))
        has_targeted = true
    end

    return has_targeted ? boundary_transports : nothing
end

#####
##### Enforce targeted open boundary transports
#####

enforce_targeted_open_boundary_transport!(model, ::Nothing) = nothing

"""
    enforce_targeted_open_boundary_transport!(model, boundary_transport)

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
