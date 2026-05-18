using Oceananigans.BoundaryConditions: BoundaryCondition, Open, has_target_transport, get_target_transport, FieldBoundaryConditions
using Oceananigans.AbstractOperations: Integral
using Oceananigans.Fields: Field, interior, compute!
using Oceananigans.Models: boundary_total_area
using GPUArraysCore: @allowscalar

const OBC = BoundaryCondition{<:Open}

#####
##### Boundary transport field constructors
#####

@inline west_transport_field(u)   = Field(Integral(view(u, 1, :, :), dims=(2, 3)))
@inline east_transport_field(u)   = Field(Integral(view(u, u.grid.Nx + 1, :, :), dims=(2, 3)))
@inline south_transport_field(v)  = Field(Integral(view(v, :, 1, :), dims=(1, 3)))
@inline north_transport_field(v)  = Field(Integral(view(v, :, v.grid.Ny + 1, :), dims=(1, 3)))
@inline bottom_transport_field(w) = Field(Integral(view(w, :, :, 1), dims=(1, 2)))
@inline top_transport_field(w)    = Field(Integral(view(w, :, :, w.grid.Nz + 1), dims=(1, 2)))

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

Initialize boundary transport `Field`s only for boundaries with a `target_transport` set on
their `PerturbationAdvection` scheme. Returns a `NamedTuple` of flux fields and boundary areas,
or `nothing` if no targeted boundaries are present.

Unlike the nonhydrostatic counterpart, no pool-correction tracking is needed because
`HydrostaticFreeSurfaceModel` lets the free surface accommodate any net imbalance.
"""
function initialize_targeted_boundary_transport(velocities)
    u, v, w = velocities
    boundary_fluxes = NamedTuple()
    has_targeted = false

    if _is_targeted(u, :west)
        boundary_fluxes = merge(boundary_fluxes, (; west_transport  = west_transport_field(u),
                                                    west_area = boundary_total_area(Val(:west), u.grid)))
        has_targeted = true
    end

    if _is_targeted(u, :east)
        boundary_fluxes = merge(boundary_fluxes, (; east_transport  = east_transport_field(u),
                                                    east_area = boundary_total_area(Val(:east), u.grid)))
        has_targeted = true
    end

    if _is_targeted(v, :south)
        boundary_fluxes = merge(boundary_fluxes, (; south_transport = south_transport_field(v),
                                                    south_area = boundary_total_area(Val(:south), v.grid)))
        has_targeted = true
    end

    if _is_targeted(v, :north)
        boundary_fluxes = merge(boundary_fluxes, (; north_transport = north_transport_field(v),
                                                    north_area = boundary_total_area(Val(:north), v.grid)))
        has_targeted = true
    end

    if _is_targeted(w, :bottom)
        boundary_fluxes = merge(boundary_fluxes, (; bottom_transport = bottom_transport_field(w),
                                                    bottom_area = boundary_total_area(Val(:bottom), w.grid)))
        has_targeted = true
    end

    if _is_targeted(w, :top)
        boundary_fluxes = merge(boundary_fluxes, (; top_transport = top_transport_field(w),
                                                    top_area = boundary_total_area(Val(:top), w.grid)))
        has_targeted = true
    end

    return has_targeted ? boundary_fluxes : nothing
end

#####
##### Update targeted boundary transports
#####

update_targeted_boundary_transport!(::Nothing) = nothing
function update_targeted_boundary_transport!(boundary_transport::NamedTuple)
    for val in boundary_transport
        val isa Field && compute!(val)
    end
    return nothing
end

#####
##### Per-boundary targeted corrections
#####

function apply_targeted_left_boundary_correction!(u, bc::OBC, ::Val{:west}, bt)
    has_target_transport(bc.classification.scheme) || return nothing
    west_area = boundary_total_area(Val(:west), u.grid)
    target    = get_target_transport(bc.classification.scheme, u.grid)
    Q_actual  = @allowscalar bt.west_transport[]
    interior(u, 1, :, :) .-= (Q_actual - target) / west_area
    return nothing
end

function apply_targeted_left_boundary_correction!(v, bc::OBC, ::Val{:south}, bt)
    has_target_transport(bc.classification.scheme) || return nothing
    south_area = boundary_total_area(Val(:south), v.grid)
    target     = get_target_transport(bc.classification.scheme, v.grid)
    Q_actual   = @allowscalar bt.south_transport[]
    interior(v, :, 1, :) .-= (Q_actual - target) / south_area
    return nothing
end

function apply_targeted_left_boundary_correction!(w, bc::OBC, ::Val{:bottom}, bt)
    has_target_transport(bc.classification.scheme) || return nothing
    bottom_area = boundary_total_area(Val(:bottom), w.grid)
    target      = get_target_transport(bc.classification.scheme, w.grid)
    Q_actual    = @allowscalar bt.bottom_transport[]
    interior(w, :, :, 1) .-= (Q_actual - target) / bottom_area
    return nothing
end

function apply_targeted_right_boundary_correction!(u, bc::OBC, ::Val{:east}, bt)
    has_target_transport(bc.classification.scheme) || return nothing
    east_area = boundary_total_area(Val(:east), u.grid)
    target    = get_target_transport(bc.classification.scheme, u.grid)
    Q_actual  = @allowscalar bt.east_transport[]
    interior(u, u.grid.Nx + 1, :, :) .-= (Q_actual - target) / east_area
    return nothing
end

function apply_targeted_right_boundary_correction!(v, bc::OBC, ::Val{:north}, bt)
    has_target_transport(bc.classification.scheme) || return nothing
    north_area = boundary_total_area(Val(:north), v.grid)
    target     = get_target_transport(bc.classification.scheme, v.grid)
    Q_actual   = @allowscalar bt.north_transport[]
    interior(v, :, v.grid.Ny + 1, :) .-= (Q_actual - target) / north_area
    return nothing
end

function apply_targeted_right_boundary_correction!(w, bc::OBC, ::Val{:top}, bt)
    has_target_transport(bc.classification.scheme) || return nothing
    top_area = boundary_total_area(Val(:top), w.grid)
    target   = get_target_transport(bc.classification.scheme, w.grid)
    Q_actual = @allowscalar bt.top_transport[]
    interior(w, :, :, w.grid.Nz + 1) .-= (Q_actual - target) / top_area
    return nothing
end

apply_targeted_left_boundary_correction!(velocity, bc, side, bt)  = nothing
apply_targeted_right_boundary_correction!(velocity, bc, side, bt) = nothing

#####
##### Enforce targeted open boundary fluxes
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
    update_targeted_boundary_transport!(bt)

    # Apply per-boundary corrections.
    apply_targeted_left_boundary_correction!(u, _get_bc(u, :west),   Val(:west),   bt)
    apply_targeted_left_boundary_correction!(v, _get_bc(v, :south),  Val(:south),  bt)
    apply_targeted_left_boundary_correction!(w, _get_bc(w, :bottom), Val(:bottom), bt)
    apply_targeted_right_boundary_correction!(u, _get_bc(u, :east),  Val(:east),   bt)
    apply_targeted_right_boundary_correction!(v, _get_bc(v, :north), Val(:north),  bt)
    apply_targeted_right_boundary_correction!(w, _get_bc(w, :top),   Val(:top),    bt)

    return nothing
end
