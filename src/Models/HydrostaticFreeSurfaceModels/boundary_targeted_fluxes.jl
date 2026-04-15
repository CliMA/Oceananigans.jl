using Oceananigans.BoundaryConditions: TargetedPAOBC
using Oceananigans.AbstractOperations: Integral, Ax, Ay, Az, grid_metric_operation
using Oceananigans.Fields: Field, interior, compute!
using GPUArraysCore: @allowscalar

#####
##### Boundary area helpers
#####

function get_west_area(grid)
    dA = grid_metric_operation((Face, Center, Center), Ax, grid)
    ∫dA = sum(dA, dims=(2, 3))
    return @allowscalar ∫dA[1, 1, 1]
end

function get_east_area(grid)
    dA = grid_metric_operation((Face, Center, Center), Ax, grid)
    ∫dA = sum(dA, dims=(2, 3))
    return @allowscalar ∫dA[grid.Nx+1, 1, 1]
end

function get_south_area(grid)
    dA = grid_metric_operation((Center, Face, Center), Ay, grid)
    ∫dA = sum(dA, dims=(1, 3))
    return @allowscalar ∫dA[1, 1, 1]
end

function get_north_area(grid)
    dA = grid_metric_operation((Center, Face, Center), Ay, grid)
    ∫dA = sum(dA, dims=(1, 3))
    return @allowscalar ∫dA[1, grid.Ny+1, 1]
end

function get_bottom_area(grid)
    dA = grid_metric_operation((Center, Center, Face), Az, grid)
    ∫dA = sum(dA, dims=(1, 2))
    return @allowscalar ∫dA[1, 1, 1]
end

function get_top_area(grid)
    dA = grid_metric_operation((Center, Center, Face), Az, grid)
    ∫dA = sum(dA, dims=(1, 2))
    return @allowscalar ∫dA[1, 1, grid.Nz+1]
end

#####
##### Boundary mass flux field constructors
#####

@inline west_mass_flux_field(u)  = Field(Integral(view(u, 1, :, :), dims=(2, 3)))
@inline east_mass_flux_field(u)  = Field(Integral(view(u, u.grid.Nx + 1, :, :), dims=(2, 3)))
@inline south_mass_flux_field(v) = Field(Integral(view(v, :, 1, :), dims=(1, 3)))
@inline north_mass_flux_field(v) = Field(Integral(view(v, :, v.grid.Ny + 1, :), dims=(1, 3)))
@inline bottom_mass_flux_field(w) = Field(Integral(view(w, :, :, 1), dims=(1, 2)))
@inline top_mass_flux_field(w)   = Field(Integral(view(w, :, :, w.grid.Nz + 1), dims=(1, 2)))

#####
##### Helpers for non-Field velocity components
#####

# Only `Field`s carry `boundary_conditions`; `ZeroField`, `FunctionField`,
# `TimeSeriesInterpolation`, etc. do not — return `false` for those.
@inline _is_targeted(velocity::Field, side) =
    getproperty(velocity.boundary_conditions, side) isa TargetedPAOBC
@inline _is_targeted(velocity, side) = false

# Safely retrieve the boundary condition (returns `nothing` for non-Field types).
@inline _get_bc(velocity::Field, side) = getproperty(velocity.boundary_conditions, side)
@inline _get_bc(velocity, side) = nothing

#####
##### Initialize only targeted boundaries
#####

"""
    initialize_targeted_boundary_mass_fluxes(velocities)

Initialize boundary mass flux `Field`s only for boundaries with a `target_mass_flux` set on
their `PerturbationAdvection` scheme. Returns a `NamedTuple` of flux fields and boundary areas,
or `nothing` if no targeted boundaries are present.

Unlike the nonhydrostatic counterpart, no pool-correction tracking is needed because
`HydrostaticFreeSurfaceModel` lets the free surface accommodate any net imbalance.
"""
function initialize_targeted_boundary_mass_fluxes(velocities)
    u, v, w = velocities
    boundary_fluxes = NamedTuple()
    has_targeted = false

    if _is_targeted(u, :west)
        boundary_fluxes = merge(boundary_fluxes, (; west_mass_flux  = west_mass_flux_field(u),
                                                    west_area = get_west_area(u.grid)))
        has_targeted = true
    end

    if _is_targeted(u, :east)
        boundary_fluxes = merge(boundary_fluxes, (; east_mass_flux  = east_mass_flux_field(u),
                                                    east_area = get_east_area(u.grid)))
        has_targeted = true
    end

    if _is_targeted(v, :south)
        boundary_fluxes = merge(boundary_fluxes, (; south_mass_flux = south_mass_flux_field(v),
                                                    south_area = get_south_area(v.grid)))
        has_targeted = true
    end

    if _is_targeted(v, :north)
        boundary_fluxes = merge(boundary_fluxes, (; north_mass_flux = north_mass_flux_field(v),
                                                    north_area = get_north_area(v.grid)))
        has_targeted = true
    end

    if _is_targeted(w, :bottom)
        boundary_fluxes = merge(boundary_fluxes, (; bottom_mass_flux = bottom_mass_flux_field(w),
                                                    bottom_area = get_bottom_area(w.grid)))
        has_targeted = true
    end

    if _is_targeted(w, :top)
        boundary_fluxes = merge(boundary_fluxes, (; top_mass_flux = top_mass_flux_field(w),
                                                    top_area = get_top_area(w.grid)))
        has_targeted = true
    end

    return has_targeted ? boundary_fluxes : nothing
end

#####
##### Update targeted boundary mass fluxes
#####

update_targeted_boundary_mass_fluxes!(::Nothing) = nothing
function update_targeted_boundary_mass_fluxes!(boundary_mass_fluxes::NamedTuple)
    for val in boundary_mass_fluxes
        val isa Field && compute!(val)
    end
    return nothing
end

#####
##### Per-boundary targeted corrections
#####

function apply_targeted_left_boundary_correction!(u, bc::TargetedPAOBC, ::Val{:west}, bmf)
    target   = bc.classification.scheme.target_mass_flux
    Q_actual = @allowscalar bmf.west_mass_flux[]
    interior(u, 1, :, :) .-= (Q_actual - target) / bmf.west_area
    return nothing
end

function apply_targeted_left_boundary_correction!(v, bc::TargetedPAOBC, ::Val{:south}, bmf)
    target   = bc.classification.scheme.target_mass_flux
    Q_actual = @allowscalar bmf.south_mass_flux[]
    interior(v, :, 1, :) .-= (Q_actual - target) / bmf.south_area
    return nothing
end

function apply_targeted_left_boundary_correction!(w, bc::TargetedPAOBC, ::Val{:bottom}, bmf)
    target   = bc.classification.scheme.target_mass_flux
    Q_actual = @allowscalar bmf.bottom_mass_flux[]
    interior(w, :, :, 1) .-= (Q_actual - target) / bmf.bottom_area
    return nothing
end

function apply_targeted_right_boundary_correction!(u, bc::TargetedPAOBC, ::Val{:east}, bmf)
    target   = bc.classification.scheme.target_mass_flux
    Q_actual = @allowscalar bmf.east_mass_flux[]
    interior(u, u.grid.Nx + 1, :, :) .-= (Q_actual - target) / bmf.east_area
    return nothing
end

function apply_targeted_right_boundary_correction!(v, bc::TargetedPAOBC, ::Val{:north}, bmf)
    target   = bc.classification.scheme.target_mass_flux
    Q_actual = @allowscalar bmf.north_mass_flux[]
    interior(v, :, v.grid.Ny + 1, :) .-= (Q_actual - target) / bmf.north_area
    return nothing
end

function apply_targeted_right_boundary_correction!(w, bc::TargetedPAOBC, ::Val{:top}, bmf)
    target   = bc.classification.scheme.target_mass_flux
    Q_actual = @allowscalar bmf.top_mass_flux[]
    interior(w, :, :, w.grid.Nz + 1) .-= (Q_actual - target) / bmf.top_area
    return nothing
end

apply_targeted_left_boundary_correction!(velocity, bc, side, bmf)  = nothing
apply_targeted_right_boundary_correction!(velocity, bc, side, bmf) = nothing

#####
##### Enforce targeted open boundary fluxes
#####

enforce_targeted_open_boundary_fluxes!(model, ::Nothing) = nothing

"""
    enforce_targeted_open_boundary_fluxes!(model, boundary_mass_fluxes)

Correct boundary velocities to achieve the prescribed `target_mass_flux` for all open
boundaries that carry a `TargetedPAOBC`.

Unlike the nonhydrostatic version, no global pool correction is applied after the targeted
step: the free surface (η) is free to rise or fall to accommodate any net mass imbalance.
"""
function enforce_targeted_open_boundary_fluxes!(model, bmf)
    u, v, w = model.velocities

    # Compute current fluxes through targeted boundaries.
    update_targeted_boundary_mass_fluxes!(bmf)

    # Apply per-boundary corrections.
    apply_targeted_left_boundary_correction!(u, _get_bc(u, :west),   Val(:west),   bmf)
    apply_targeted_left_boundary_correction!(v, _get_bc(v, :south),  Val(:south),  bmf)
    apply_targeted_left_boundary_correction!(w, _get_bc(w, :bottom), Val(:bottom), bmf)
    apply_targeted_right_boundary_correction!(u, _get_bc(u, :east),  Val(:east),   bmf)
    apply_targeted_right_boundary_correction!(v, _get_bc(v, :north), Val(:north),  bmf)
    apply_targeted_right_boundary_correction!(w, _get_bc(w, :top),   Val(:top),    bmf)

    return nothing
end
