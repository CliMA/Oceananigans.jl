using Oceananigans.BoundaryConditions: BoundaryCondition, Open
using Oceananigans.AbstractOperations: Integral, Ax, Ay, Az, grid_metric_operation
using Oceananigans.Fields: Field, interior, compute!
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

# Left boundary integrals for normal velocity / momentum components
@inline west_mass_flux(u)   = Field(Integral(view(u, 1, :, :), dims=(2, 3)))
@inline south_mass_flux(v)  = Field(Integral(view(v, :, 1, :), dims=(1, 3)))
@inline bottom_mass_flux(w) = Field(Integral(view(w, :, :, 1), dims=(1, 2)))

# Right boundary integrals for normal velocity / momentum components
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

"""
    initialize_boundary_mass_fluxes(velocities::NamedTuple)

Initialize boundary mass fluxes for boundaries with `OpenBoundaryCondition`s,
returning a `NamedTuple` of boundary fluxes and areas, or `nothing` when no
open boundary requires correction.

The `velocities` argument is a `NamedTuple` of three face-normal fields.
Destructuring is positional, so any 3-field `NamedTuple` works:
`(; u, v, w)` for incompressible / volume-flux models like `NonhydrostaticModel`,
or `(; ρu, ρv, ρw)` for anelastic / mass-flux models.
"""
function initialize_boundary_mass_fluxes(velocities::NamedTuple)

    u, v, w = velocities
    u_bcs = u.boundary_conditions
    v_bcs = v.boundary_conditions
    w_bcs = w.boundary_conditions

    boundary_fluxes = NamedTuple()
    right_scheme_boundaries = Symbol[]
    left_scheme_boundaries = Symbol[]
    total_area_scheme_boundaries = zero(eltype(u))

    # Check west boundary (u velocity)
    west_flux_and_area = initialize_boundary_mass_flux(u, u_bcs.west, Val(:west))
    boundary_fluxes = merge(boundary_fluxes, west_flux_and_area)
    if needs_mass_flux_correction(u_bcs.west)
        push!(left_scheme_boundaries, :west)
        total_area_scheme_boundaries += boundary_fluxes.west_area
    end

    # Check east boundary (u velocity)
    east_flux_and_area = initialize_boundary_mass_flux(u, u_bcs.east, Val(:east))
    boundary_fluxes = merge(boundary_fluxes, east_flux_and_area)
    if needs_mass_flux_correction(u_bcs.east)
        push!(right_scheme_boundaries, :east)
        total_area_scheme_boundaries += boundary_fluxes.east_area
    end

    # Check south boundary (v velocity)
    south_flux_and_area = initialize_boundary_mass_flux(v, v_bcs.south, Val(:south))
    boundary_fluxes = merge(boundary_fluxes, south_flux_and_area)
    if needs_mass_flux_correction(v_bcs.south)
        push!(left_scheme_boundaries, :south)
        total_area_scheme_boundaries += boundary_fluxes.south_area
    end

    # Check north boundary (v velocity)
    north_flux_and_area = initialize_boundary_mass_flux(v, v_bcs.north, Val(:north))
    boundary_fluxes = merge(boundary_fluxes, north_flux_and_area)
    if needs_mass_flux_correction(v_bcs.north)
        push!(right_scheme_boundaries, :north)
        total_area_scheme_boundaries += boundary_fluxes.north_area
    end

    # Check bottom boundary (w velocity)
    bottom_flux_and_area = initialize_boundary_mass_flux(w, w_bcs.bottom, Val(:bottom))
    boundary_fluxes = merge(boundary_fluxes, bottom_flux_and_area)
    if needs_mass_flux_correction(w_bcs.bottom)
        push!(left_scheme_boundaries, :bottom)
        total_area_scheme_boundaries += boundary_fluxes.bottom_area
    end

    # Check top boundary (w velocity)
    top_flux_and_area = initialize_boundary_mass_flux(w, w_bcs.top, Val(:top))
    boundary_fluxes = merge(boundary_fluxes, top_flux_and_area)
    if needs_mass_flux_correction(w_bcs.top)
        push!(right_scheme_boundaries, :top)
        total_area_scheme_boundaries += boundary_fluxes.top_area
    end

    boundary_fluxes = merge(boundary_fluxes, (; left_scheme_boundaries = Tuple(left_scheme_boundaries),
                                                right_scheme_boundaries = Tuple(right_scheme_boundaries),
                                                total_area_scheme_boundaries))

    if length(boundary_fluxes.left_scheme_boundaries) == 0 && length(boundary_fluxes.right_scheme_boundaries) == 0
        return nothing
    else
        return boundary_fluxes
    end
end

update_open_boundary_mass_fluxes!(boundary_mass_fluxes) = map(compute!, boundary_mass_fluxes)

open_boundary_mass_flux(bmf, ::OBC, ::Val{:west})   = @allowscalar bmf.west_mass_flux[]
open_boundary_mass_flux(bmf, ::OBC, ::Val{:east})   = @allowscalar bmf.east_mass_flux[]
open_boundary_mass_flux(bmf, ::OBC, ::Val{:south})  = @allowscalar bmf.south_mass_flux[]
open_boundary_mass_flux(bmf, ::OBC, ::Val{:north})  = @allowscalar bmf.north_mass_flux[]
open_boundary_mass_flux(bmf, ::OBC, ::Val{:bottom}) = @allowscalar bmf.bottom_mass_flux[]
open_boundary_mass_flux(bmf, ::OBC, ::Val{:top})    = @allowscalar bmf.top_mass_flux[]

# Imposed-velocity Open BCs (no scheme) and non-open BCs contribute zero to the
# net boundary mass flux: imposed-velocity outflow is already fixed, and closed
# boundaries carry no mass flux by definition. Method-per-side avoids ambiguity
# with the `::OBC, ::Val{:<side>}` methods above (ZIOBC <: IOBC <: OBC).
open_boundary_mass_flux(bmf, ::ZIOBC, ::Val{:west})   = zero(bmf.total_area_scheme_boundaries)
open_boundary_mass_flux(bmf, ::ZIOBC, ::Val{:east})   = zero(bmf.total_area_scheme_boundaries)
open_boundary_mass_flux(bmf, ::ZIOBC, ::Val{:south})  = zero(bmf.total_area_scheme_boundaries)
open_boundary_mass_flux(bmf, ::ZIOBC, ::Val{:north})  = zero(bmf.total_area_scheme_boundaries)
open_boundary_mass_flux(bmf, ::ZIOBC, ::Val{:bottom}) = zero(bmf.total_area_scheme_boundaries)
open_boundary_mass_flux(bmf, ::ZIOBC, ::Val{:top})    = zero(bmf.total_area_scheme_boundaries)

open_boundary_mass_flux(bmf, bc, side) = zero(bmf.total_area_scheme_boundaries)

function open_boundary_mass_inflow(boundary_mass_fluxes, velocities)
    update_open_boundary_mass_fluxes!(boundary_mass_fluxes)

    u, v, w = velocities
    total_flux = zero(boundary_mass_fluxes.total_area_scheme_boundaries)

    # Add flux through left boundaries
    total_flux += open_boundary_mass_flux(boundary_mass_fluxes, u.boundary_conditions.west,   Val(:west))
    total_flux += open_boundary_mass_flux(boundary_mass_fluxes, v.boundary_conditions.south,  Val(:south))
    total_flux += open_boundary_mass_flux(boundary_mass_fluxes, w.boundary_conditions.bottom, Val(:bottom))

    # Subtract flux through right boundaries.
    total_flux -= open_boundary_mass_flux(boundary_mass_fluxes, u.boundary_conditions.east,  Val(:east))
    total_flux -= open_boundary_mass_flux(boundary_mass_fluxes, v.boundary_conditions.north, Val(:north))
    total_flux -= open_boundary_mass_flux(boundary_mass_fluxes, w.boundary_conditions.top,   Val(:top))

    return total_flux
end

correct_left_boundary_mass_flux!(u, bc::OBC, ::Val{:west},    A⁻¹_∮udA) = interior(u, 1, :, :) .-= A⁻¹_∮udA
correct_left_boundary_mass_flux!(v, bc::OBC, ::Val{:south},   A⁻¹_∮udA) = interior(v, :, 1, :) .-= A⁻¹_∮udA
correct_left_boundary_mass_flux!(w, bc::OBC, ::Val{:bottom},  A⁻¹_∮udA) = interior(w, :, :, 1) .-= A⁻¹_∮udA
correct_left_boundary_mass_flux!(u, bc::IOBC, ::Val{:west},   A⁻¹_∮udA) = nothing
correct_left_boundary_mass_flux!(v, bc::IOBC, ::Val{:south},  A⁻¹_∮udA) = nothing
correct_left_boundary_mass_flux!(w, bc::IOBC, ::Val{:bottom}, A⁻¹_∮udA) = nothing
correct_left_boundary_mass_flux!(u, bc, side, A⁻¹_∮udA) = nothing

correct_right_boundary_mass_flux!(u, bc::OBC, ::Val{:east},   A⁻¹_∮udA) = interior(u, u.grid.Nx + 1, :, :) .+= A⁻¹_∮udA
correct_right_boundary_mass_flux!(v, bc::OBC, ::Val{:north},  A⁻¹_∮udA) = interior(v, :, v.grid.Ny + 1, :) .+= A⁻¹_∮udA
correct_right_boundary_mass_flux!(w, bc::OBC, ::Val{:top},    A⁻¹_∮udA) = interior(w, :, :, w.grid.Nz + 1) .+= A⁻¹_∮udA
correct_right_boundary_mass_flux!(u, bc::IOBC, ::Val{:east},  A⁻¹_∮udA) = nothing
correct_right_boundary_mass_flux!(v, bc::IOBC, ::Val{:north}, A⁻¹_∮udA) = nothing
correct_right_boundary_mass_flux!(w, bc::IOBC, ::Val{:top},   A⁻¹_∮udA) = nothing
correct_right_boundary_mass_flux!(u, bc, side, A⁻¹_∮udA) = nothing

enforce_open_boundary_mass_conservation!(velocities, ::Nothing) = nothing

"""
    enforce_open_boundary_mass_conservation!(velocities, boundary_mass_fluxes)

Correct boundary values in `velocities` so that the combined net mass flux through all
`OpenBoundaryCondition` boundaries with a radiation scheme vanishes — the
solvability condition for an incompressible / anelastic pressure Poisson problem.

`velocities` is a `NamedTuple` of three face-normal fields: volume fluxes
`(; u, v, w)` for an incompressible model like `NonhydrostaticModel`, or
density-weighted momenta `(; ρu, ρv, ρw)` for an anelastic model.
`boundary_mass_fluxes` is the container returned by
[`initialize_boundary_mass_fluxes`](@ref), or `nothing` when no open boundaries
require correction (in which case this is a no-op).
"""
function enforce_open_boundary_mass_conservation!(velocities, boundary_mass_fluxes)
    u, v, w = velocities

    ∮udA = open_boundary_mass_inflow(boundary_mass_fluxes, velocities)
    A = boundary_mass_fluxes.total_area_scheme_boundaries

    A⁻¹_∮udA = ∮udA / A

    correct_left_boundary_mass_flux!(u, u.boundary_conditions.west,   Val(:west),   A⁻¹_∮udA)
    correct_left_boundary_mass_flux!(v, v.boundary_conditions.south,  Val(:south),  A⁻¹_∮udA)
    correct_left_boundary_mass_flux!(w, w.boundary_conditions.bottom, Val(:bottom), A⁻¹_∮udA)

    correct_right_boundary_mass_flux!(u, u.boundary_conditions.east,  Val(:east),  A⁻¹_∮udA)
    correct_right_boundary_mass_flux!(v, v.boundary_conditions.north, Val(:north), A⁻¹_∮udA)
    correct_right_boundary_mass_flux!(w, w.boundary_conditions.top,   Val(:top),   A⁻¹_∮udA)

    return nothing
end
