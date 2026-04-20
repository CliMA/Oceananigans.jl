using Oceananigans.BoundaryConditions: BoundaryCondition, Open, has_target_mass_flux, get_target_mass_flux
using Oceananigans.AbstractOperations: Integral, Ax, Ay, Az, grid_metric_operation
using Oceananigans.Fields: Field, interior
using GPUArraysCore: @allowscalar

const OBC  = BoundaryCondition{<:Open} # OpenBoundaryCondition
const IOBC = BoundaryCondition{<:Open{<:Nothing}} # "Imposed-velocity" OpenBoundaryCondition (with no scheme)
const FIOBC = BoundaryCondition{<:Open{<:Nothing}, <:Number} # "Fixed-imposed-velocity" OpenBoundaryCondition
const ZIOBC = BoundaryCondition{<:Open{<:Nothing}, <:Nothing} # "Zero-imposed-velocity" OpenBoundaryCondition (no-inflow)

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

# Left boundary integrals for normal velocity components
@inline west_mass_flux(u)   = Field(Integral(view(u, 1, :, :), dims=(2, 3)))
@inline south_mass_flux(v)  = Field(Integral(view(v, :, 1, :), dims=(1, 3)))
@inline bottom_mass_flux(w) = Field(Integral(view(w, :, :, 1), dims=(1, 2)))

# Right boundary integrals for normal velocity components
@inline east_mass_flux(u)   = Field(Integral(view(u, u.grid.Nx + 1, :, :), dims=(2, 3)))
@inline north_mass_flux(v)  = Field(Integral(view(v, :, v.grid.Ny + 1, :), dims=(1, 3)))
@inline top_mass_flux(w)    = Field(Integral(view(w, :, :, w.grid.Nz + 1), dims=(1, 2)))

initialize_boundary_mass_flux(u, bc::OBC, ::Val{:west})   = (; west_mass_flux = west_mass_flux(u), west_area = get_west_area(u.grid))
initialize_boundary_mass_flux(u, bc::OBC, ::Val{:east})   = (; east_mass_flux = east_mass_flux(u), east_area = get_east_area(u.grid))
initialize_boundary_mass_flux(v, bc::OBC, ::Val{:south})  = (; south_mass_flux = south_mass_flux(v), south_area = get_south_area(v.grid))
initialize_boundary_mass_flux(v, bc::OBC, ::Val{:north})  = (; north_mass_flux = north_mass_flux(v), north_area = get_north_area(v.grid))
initialize_boundary_mass_flux(w, bc::OBC, ::Val{:bottom}) = (; bottom_mass_flux = bottom_mass_flux(w), bottom_area = get_bottom_area(w.grid))
initialize_boundary_mass_flux(w, bc::OBC, ::Val{:top})    = (; top_mass_flux = top_mass_flux(w), top_area = get_top_area(w.grid))

initialize_boundary_mass_flux(u, bc::ZIOBC, ::Val{:west})   = NamedTuple()
initialize_boundary_mass_flux(u, bc::ZIOBC, ::Val{:east})   = NamedTuple()
initialize_boundary_mass_flux(v, bc::ZIOBC, ::Val{:south})  = NamedTuple()
initialize_boundary_mass_flux(v, bc::ZIOBC, ::Val{:north})  = NamedTuple()
initialize_boundary_mass_flux(w, bc::ZIOBC, ::Val{:bottom}) = NamedTuple()
initialize_boundary_mass_flux(w, bc::ZIOBC, ::Val{:top})    = NamedTuple()

initialize_boundary_mass_flux(u, bc::FIOBC, ::Val{:west})   = (; west_mass_flux = bc.condition * get_west_area(u.grid), west_area = get_west_area(u.grid))
initialize_boundary_mass_flux(u, bc::FIOBC, ::Val{:east})   = (; east_mass_flux = bc.condition * get_east_area(u.grid), east_area = get_east_area(u.grid))
initialize_boundary_mass_flux(v, bc::FIOBC, ::Val{:south})  = (; south_mass_flux = bc.condition * get_south_area(v.grid), south_area = get_south_area(v.grid))
initialize_boundary_mass_flux(v, bc::FIOBC, ::Val{:north})  = (; north_mass_flux = bc.condition * get_north_area(v.grid), north_area = get_north_area(v.grid))
initialize_boundary_mass_flux(w, bc::FIOBC, ::Val{:bottom}) = (; bottom_mass_flux = bc.condition * get_bottom_area(w.grid), bottom_area = get_bottom_area(w.grid))
initialize_boundary_mass_flux(w, bc::FIOBC, ::Val{:top})    = (; top_mass_flux = bc.condition * get_top_area(w.grid), top_area = get_top_area(w.grid))

initialize_boundary_mass_flux(velocity, ::Nothing, side) = NamedTuple()
initialize_boundary_mass_flux(velocity, bc, side) = NamedTuple()

needs_mass_flux_correction(::IOBC) = false
needs_mass_flux_correction(::OBC) = true
needs_mass_flux_correction(::Nothing) = false
needs_mass_flux_correction(bc) = false

# True for scheme boundaries that participate in the global pool correction.
# Boundaries with a target_mass_flux are corrected independently and excluded from the pool.
needs_pool_correction(::IOBC) = false
needs_pool_correction(bc::OBC) = !has_target_mass_flux(bc.classification.scheme)
needs_pool_correction(::Nothing) = false
needs_pool_correction(bc) = false

"""
    initialize_boundary_mass_fluxes(velocities::NamedTuple)

Initialize boundary mass fluxes for boundaries with OpenBoundaryConditions,
returning a NamedTuple of boundary fluxes.
"""
function initialize_boundary_mass_fluxes(velocities::NamedTuple)

    u, v, w = velocities
    u_bcs = u.boundary_conditions
    v_bcs = v.boundary_conditions
    w_bcs = w.boundary_conditions

    boundary_fluxes = NamedTuple()
    right_scheme_boundaries = Symbol[]
    left_scheme_boundaries = Symbol[]
    total_area_pool_boundaries = zero(eltype(u))

    # Check west boundary (u velocity)
    west_flux_and_area = initialize_boundary_mass_flux(u, u_bcs.west, Val(:west))
    boundary_fluxes = merge(boundary_fluxes, west_flux_and_area)
    if needs_mass_flux_correction(u_bcs.west)
        push!(left_scheme_boundaries, :west)
        if needs_pool_correction(u_bcs.west)
            total_area_pool_boundaries += boundary_fluxes.west_area
        end
    end

    # Check east boundary (u velocity)
    east_flux_and_area = initialize_boundary_mass_flux(u, u_bcs.east, Val(:east))
    boundary_fluxes = merge(boundary_fluxes, east_flux_and_area)
    if needs_mass_flux_correction(u_bcs.east)
        push!(right_scheme_boundaries, :east)
        if needs_pool_correction(u_bcs.east)
            total_area_pool_boundaries += boundary_fluxes.east_area
        end
    end

    # Check south boundary (v velocity)
    south_flux_and_area = initialize_boundary_mass_flux(v, v_bcs.south, Val(:south))
    boundary_fluxes = merge(boundary_fluxes, south_flux_and_area)
    if needs_mass_flux_correction(v_bcs.south)
        push!(left_scheme_boundaries, :south)
        if needs_pool_correction(v_bcs.south)
            total_area_pool_boundaries += boundary_fluxes.south_area
        end
    end

    # Check north boundary (v velocity)
    north_flux_and_area = initialize_boundary_mass_flux(v, v_bcs.north, Val(:north))
    boundary_fluxes = merge(boundary_fluxes, north_flux_and_area)
    if needs_mass_flux_correction(v_bcs.north)
        push!(right_scheme_boundaries, :north)
        if needs_pool_correction(v_bcs.north)
            total_area_pool_boundaries += boundary_fluxes.north_area
        end
    end

    # Check bottom boundary (w velocity)
    bottom_flux_and_area = initialize_boundary_mass_flux(w, w_bcs.bottom, Val(:bottom))
    boundary_fluxes = merge(boundary_fluxes, bottom_flux_and_area)
    if needs_mass_flux_correction(w_bcs.bottom)
        push!(left_scheme_boundaries, :bottom)
        if needs_pool_correction(w_bcs.bottom)
            total_area_pool_boundaries += boundary_fluxes.bottom_area
        end
    end

    # Check top boundary (w velocity)
    top_flux_and_area = initialize_boundary_mass_flux(w, w_bcs.top, Val(:top))
    boundary_fluxes = merge(boundary_fluxes, top_flux_and_area)
    if needs_mass_flux_correction(w_bcs.top)
        push!(right_scheme_boundaries, :top)
        if needs_pool_correction(w_bcs.top)
            total_area_pool_boundaries += boundary_fluxes.top_area
        end
    end

    boundary_fluxes = merge(boundary_fluxes, (; left_scheme_boundaries = Tuple(left_scheme_boundaries),
                                                right_scheme_boundaries = Tuple(right_scheme_boundaries),
                                                total_area_pool_boundaries))

    if length(boundary_fluxes.left_scheme_boundaries) == 0 && length(boundary_fluxes.right_scheme_boundaries) == 0
        return nothing
    else
        return boundary_fluxes
    end
end

update_open_boundary_mass_fluxes!(model) = map(compute!, model.boundary_mass_fluxes)

open_boundary_mass_flux(model, bc::OBC, ::Val{:west}, u) = @allowscalar model.boundary_mass_fluxes.west_mass_flux[]
open_boundary_mass_flux(model, bc::OBC, ::Val{:east}, u) = @allowscalar model.boundary_mass_fluxes.east_mass_flux[]
open_boundary_mass_flux(model, bc::OBC, ::Val{:south}, v) = @allowscalar model.boundary_mass_fluxes.south_mass_flux[]
open_boundary_mass_flux(model, bc::OBC, ::Val{:north}, v) = @allowscalar model.boundary_mass_fluxes.north_mass_flux[]
open_boundary_mass_flux(model, bc::OBC, ::Val{:bottom}, w) = @allowscalar model.boundary_mass_fluxes.bottom_mass_flux[]
open_boundary_mass_flux(model, bc::OBC, ::Val{:top}, w) = @allowscalar model.boundary_mass_fluxes.top_mass_flux[]

open_boundary_mass_flux(model, bc::ZIOBC, ::Val{:west}, u) = zero(model.grid)
open_boundary_mass_flux(model, bc::ZIOBC, ::Val{:east}, u) = zero(model.grid)
open_boundary_mass_flux(model, bc::ZIOBC, ::Val{:south}, v) = zero(model.grid)
open_boundary_mass_flux(model, bc::ZIOBC, ::Val{:north}, v) = zero(model.grid)
open_boundary_mass_flux(model, bc::ZIOBC, ::Val{:bottom}, w) = zero(model.grid)
open_boundary_mass_flux(model, bc::ZIOBC, ::Val{:top}, w) = zero(model.grid)

open_boundary_mass_flux(model, bc, side, velocity) = zero(model.grid)

function open_boundary_mass_inflow(model)
    update_open_boundary_mass_fluxes!(model)

    u, v, w = model.velocities
    total_flux = zero(model.grid)

    # Add flux through left boundaries
    total_flux += open_boundary_mass_flux(model, u.boundary_conditions.west, Val(:west), u)
    total_flux += open_boundary_mass_flux(model, v.boundary_conditions.south, Val(:south), v)
    total_flux += open_boundary_mass_flux(model, w.boundary_conditions.bottom, Val(:bottom), w)

    # Subtract flux through right boundaries.
    total_flux -= open_boundary_mass_flux(model, u.boundary_conditions.east, Val(:east), u)
    total_flux -= open_boundary_mass_flux(model, v.boundary_conditions.north, Val(:north), v)
    total_flux -= open_boundary_mass_flux(model, w.boundary_conditions.top, Val(:top), w)

    return total_flux
end

# Boundaries with a target_mass_flux are corrected independently; skip them in the pool correction.
function correct_left_boundary_mass_flux!(u, bc::OBC, ::Val{:west}, A⁻¹_∮udA)
    has_target_mass_flux(bc.classification.scheme) && return nothing
    interior(u, 1, :, :) .-= A⁻¹_∮udA
end
function correct_left_boundary_mass_flux!(v, bc::OBC, ::Val{:south}, A⁻¹_∮udA)
    has_target_mass_flux(bc.classification.scheme) && return nothing
    interior(v, :, 1, :) .-= A⁻¹_∮udA
end
function correct_left_boundary_mass_flux!(w, bc::OBC, ::Val{:bottom}, A⁻¹_∮udA)
    has_target_mass_flux(bc.classification.scheme) && return nothing
    interior(w, :, :, 1) .-= A⁻¹_∮udA
end
correct_left_boundary_mass_flux!(u, bc::IOBC, ::Val{:west},   A⁻¹_∮udA) = nothing
correct_left_boundary_mass_flux!(v, bc::IOBC, ::Val{:south},  A⁻¹_∮udA) = nothing
correct_left_boundary_mass_flux!(w, bc::IOBC, ::Val{:bottom}, A⁻¹_∮udA) = nothing
correct_left_boundary_mass_flux!(u, bc, side, A⁻¹_∮udA) = nothing

function correct_right_boundary_mass_flux!(u, bc::OBC, ::Val{:east}, A⁻¹_∮udA)
    has_target_mass_flux(bc.classification.scheme) && return nothing
    interior(u, u.grid.Nx + 1, :, :) .+= A⁻¹_∮udA
end
function correct_right_boundary_mass_flux!(v, bc::OBC, ::Val{:north}, A⁻¹_∮udA)
    has_target_mass_flux(bc.classification.scheme) && return nothing
    interior(v, :, v.grid.Ny + 1, :) .+= A⁻¹_∮udA
end
function correct_right_boundary_mass_flux!(w, bc::OBC, ::Val{:top}, A⁻¹_∮udA)
    has_target_mass_flux(bc.classification.scheme) && return nothing
    interior(w, :, :, w.grid.Nz + 1) .+= A⁻¹_∮udA
end
correct_right_boundary_mass_flux!(u, bc::IOBC, ::Val{:east},  A⁻¹_∮udA) = nothing
correct_right_boundary_mass_flux!(v, bc::IOBC, ::Val{:north}, A⁻¹_∮udA) = nothing
correct_right_boundary_mass_flux!(w, bc::IOBC, ::Val{:top},   A⁻¹_∮udA) = nothing
correct_right_boundary_mass_flux!(u, bc, side, A⁻¹_∮udA) = nothing

# Apply a per-boundary correction to reach the prescribed target_mass_flux.
# target_mass_flux is the desired integral of the normal velocity in the positive
# coordinate direction (e.g., eastward for west/east boundaries).
function apply_targeted_left_boundary_correction!(u, bc::OBC, ::Val{:west}, boundary_mass_fluxes)
    has_target_mass_flux(bc.classification.scheme) || return nothing
    target = get_target_mass_flux(bc.classification.scheme)
    Q_actual = @allowscalar boundary_mass_fluxes.west_mass_flux[]
    interior(u, 1, :, :) .-= (Q_actual - target) / boundary_mass_fluxes.west_area
    return nothing
end

function apply_targeted_left_boundary_correction!(v, bc::OBC, ::Val{:south}, boundary_mass_fluxes)
    has_target_mass_flux(bc.classification.scheme) || return nothing
    target = get_target_mass_flux(bc.classification.scheme)
    Q_actual = @allowscalar boundary_mass_fluxes.south_mass_flux[]
    interior(v, :, 1, :) .-= (Q_actual - target) / boundary_mass_fluxes.south_area
    return nothing
end

function apply_targeted_left_boundary_correction!(w, bc::OBC, ::Val{:bottom}, boundary_mass_fluxes)
    has_target_mass_flux(bc.classification.scheme) || return nothing
    target = get_target_mass_flux(bc.classification.scheme)
    Q_actual = @allowscalar boundary_mass_fluxes.bottom_mass_flux[]
    interior(w, :, :, 1) .-= (Q_actual - target) / boundary_mass_fluxes.bottom_area
    return nothing
end

function apply_targeted_right_boundary_correction!(u, bc::OBC, ::Val{:east}, boundary_mass_fluxes)
    has_target_mass_flux(bc.classification.scheme) || return nothing
    target = get_target_mass_flux(bc.classification.scheme)
    Q_actual = @allowscalar boundary_mass_fluxes.east_mass_flux[]
    interior(u, u.grid.Nx + 1, :, :) .-= (Q_actual - target) / boundary_mass_fluxes.east_area
    return nothing
end

function apply_targeted_right_boundary_correction!(v, bc::OBC, ::Val{:north}, boundary_mass_fluxes)
    has_target_mass_flux(bc.classification.scheme) || return nothing
    target = get_target_mass_flux(bc.classification.scheme)
    Q_actual = @allowscalar boundary_mass_fluxes.north_mass_flux[]
    interior(v, :, v.grid.Ny + 1, :) .-= (Q_actual - target) / boundary_mass_fluxes.north_area
    return nothing
end

function apply_targeted_right_boundary_correction!(w, bc::OBC, ::Val{:top}, boundary_mass_fluxes)
    has_target_mass_flux(bc.classification.scheme) || return nothing
    target = get_target_mass_flux(bc.classification.scheme)
    Q_actual = @allowscalar boundary_mass_fluxes.top_mass_flux[]
    interior(w, :, :, w.grid.Nz + 1) .-= (Q_actual - target) / boundary_mass_fluxes.top_area
    return nothing
end

apply_targeted_left_boundary_correction!(velocity, bc, side, boundary_mass_fluxes) = nothing
apply_targeted_right_boundary_correction!(velocity, bc, side, boundary_mass_fluxes) = nothing

enforce_open_boundary_mass_conservation!(model, ::Nothing) = nothing

"""
    enforce_open_boundary_mass_conservation!(model, boundary_mass_fluxes)

Correct boundary velocities for open boundary conditions to enforce mass conservation.

Boundaries with a `target_mass_flux` set on their scheme are corrected first: a uniform
velocity adjustment is applied so that the net volume flux through that boundary equals
`target_mass_flux`. The remaining net imbalance across all open boundaries is then
distributed uniformly over the pool boundaries (those without a `target_mass_flux`).
"""
function enforce_open_boundary_mass_conservation!(model, boundary_mass_fluxes)
    u, v, w = model.velocities

    # Step 1: compute all boundary fluxes before any corrections.
    update_open_boundary_mass_fluxes!(model)

    # Step 2: apply per-boundary corrections for targeted boundaries.
    apply_targeted_left_boundary_correction!(u, u.boundary_conditions.west,   Val(:west),   boundary_mass_fluxes)
    apply_targeted_left_boundary_correction!(v, v.boundary_conditions.south,  Val(:south),  boundary_mass_fluxes)
    apply_targeted_left_boundary_correction!(w, w.boundary_conditions.bottom, Val(:bottom), boundary_mass_fluxes)
    apply_targeted_right_boundary_correction!(u, u.boundary_conditions.east,  Val(:east),   boundary_mass_fluxes)
    apply_targeted_right_boundary_correction!(v, v.boundary_conditions.north, Val(:north),  boundary_mass_fluxes)
    apply_targeted_right_boundary_correction!(w, w.boundary_conditions.top,   Val(:top),    boundary_mass_fluxes)

    # Step 3: distribute remaining imbalance over pool boundaries.
    A = boundary_mass_fluxes.total_area_pool_boundaries
    iszero(A) && return nothing

    ∮udA = open_boundary_mass_inflow(model)
    A⁻¹_∮udA = ∮udA / A

    correct_left_boundary_mass_flux!(u, u.boundary_conditions.west,   Val(:west),   A⁻¹_∮udA)
    correct_left_boundary_mass_flux!(v, v.boundary_conditions.south,  Val(:south),  A⁻¹_∮udA)
    correct_left_boundary_mass_flux!(w, w.boundary_conditions.bottom, Val(:bottom), A⁻¹_∮udA)
    correct_right_boundary_mass_flux!(u, u.boundary_conditions.east,  Val(:east),   A⁻¹_∮udA)
    correct_right_boundary_mass_flux!(v, v.boundary_conditions.north, Val(:north),  A⁻¹_∮udA)
    correct_right_boundary_mass_flux!(w, w.boundary_conditions.top,   Val(:top),    A⁻¹_∮udA)
end
