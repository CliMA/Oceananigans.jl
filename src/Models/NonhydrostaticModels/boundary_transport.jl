using Oceananigans.BoundaryConditions: BoundaryCondition, Open, has_target_transport, get_target_transport
using Oceananigans.AbstractOperations: Integral
using Oceananigans.Fields: Field, interior
using Oceananigans.Models: boundary_total_area
using GPUArraysCore: @allowscalar

const OBC  = BoundaryCondition{<:Open} # OpenBoundaryCondition
const IOBC = BoundaryCondition{<:Open{<:Nothing}} # "Imposed-velocity" OpenBoundaryCondition (with no scheme)
const FIOBC = BoundaryCondition{<:Open{<:Nothing}, <:Number} # "Fixed-imposed-velocity" OpenBoundaryCondition
const ZIOBC = BoundaryCondition{<:Open{<:Nothing}, <:Nothing} # "Zero-imposed-velocity" OpenBoundaryCondition (no-inflow)

# Left boundary integrals for normal velocity components
@inline west_transport(u)   = Field(Integral(view(u, 1, :, :), dims=(2, 3)))
@inline south_transport(v)  = Field(Integral(view(v, :, 1, :), dims=(1, 3)))
@inline bottom_transport(w) = Field(Integral(view(w, :, :, 1), dims=(1, 2)))

# Right boundary integrals for normal velocity components
@inline east_transport(u)   = Field(Integral(view(u, u.grid.Nx + 1, :, :), dims=(2, 3)))
@inline north_transport(v)  = Field(Integral(view(v, :, v.grid.Ny + 1, :), dims=(1, 3)))
@inline top_transport(w)    = Field(Integral(view(w, :, :, w.grid.Nz + 1), dims=(1, 2)))

initialize_boundary_transport(u, bc::OBC, ::Val{:west})   = (; west_transport = west_transport(u), west_area = boundary_total_area(Val(:west), u.grid))
initialize_boundary_transport(u, bc::OBC, ::Val{:east})   = (; east_transport = east_transport(u), east_area = boundary_total_area(Val(:east), u.grid))
initialize_boundary_transport(v, bc::OBC, ::Val{:south})  = (; south_transport = south_transport(v), south_area = boundary_total_area(Val(:south), v.grid))
initialize_boundary_transport(v, bc::OBC, ::Val{:north})  = (; north_transport = north_transport(v), north_area = boundary_total_area(Val(:north), v.grid))
initialize_boundary_transport(w, bc::OBC, ::Val{:bottom}) = (; bottom_transport = bottom_transport(w), bottom_area = boundary_total_area(Val(:bottom), w.grid))
initialize_boundary_transport(w, bc::OBC, ::Val{:top})    = (; top_transport = top_transport(w), top_area = boundary_total_area(Val(:top), w.grid))

initialize_boundary_transport(u, bc::ZIOBC, ::Val{:west})   = NamedTuple()
initialize_boundary_transport(u, bc::ZIOBC, ::Val{:east})   = NamedTuple()
initialize_boundary_transport(v, bc::ZIOBC, ::Val{:south})  = NamedTuple()
initialize_boundary_transport(v, bc::ZIOBC, ::Val{:north})  = NamedTuple()
initialize_boundary_transport(w, bc::ZIOBC, ::Val{:bottom}) = NamedTuple()
initialize_boundary_transport(w, bc::ZIOBC, ::Val{:top})    = NamedTuple()

initialize_boundary_transport(u, bc::FIOBC, ::Val{:west})   = (; west_transport = bc.condition * boundary_total_area(Val(:west), u.grid), west_area = boundary_total_area(Val(:west), u.grid))
initialize_boundary_transport(u, bc::FIOBC, ::Val{:east})   = (; east_transport = bc.condition * boundary_total_area(Val(:east), u.grid), east_area = boundary_total_area(Val(:east), u.grid))
initialize_boundary_transport(v, bc::FIOBC, ::Val{:south})  = (; south_transport = bc.condition * boundary_total_area(Val(:south), v.grid), south_area = boundary_total_area(Val(:south), v.grid))
initialize_boundary_transport(v, bc::FIOBC, ::Val{:north})  = (; north_transport = bc.condition * boundary_total_area(Val(:north), v.grid), north_area = boundary_total_area(Val(:north), v.grid))
initialize_boundary_transport(w, bc::FIOBC, ::Val{:bottom}) = (; bottom_transport = bc.condition * boundary_total_area(Val(:bottom), w.grid), bottom_area = boundary_total_area(Val(:bottom), w.grid))
initialize_boundary_transport(w, bc::FIOBC, ::Val{:top})    = (; top_transport = bc.condition * boundary_total_area(Val(:top), w.grid), top_area = boundary_total_area(Val(:top), w.grid))

initialize_boundary_transport(velocity, ::Nothing, side) = NamedTuple()
initialize_boundary_transport(velocity, bc, side) = NamedTuple()

needs_transport_correction(::IOBC) = false
needs_transport_correction(::OBC) = true
needs_transport_correction(::Nothing) = false
needs_transport_correction(bc) = false

# True for scheme boundaries that participate in the global pool correction.
# Boundaries with a target_transport are corrected independently and excluded from the pool.
needs_pool_correction(::IOBC) = false
needs_pool_correction(bc::OBC) = !has_target_transport(bc.classification.scheme)
needs_pool_correction(::Nothing) = false
needs_pool_correction(bc) = false

"""
    initialize_boundary_transport(velocities::NamedTuple)

Initialize boundary transports for boundaries with OpenBoundaryConditions,
returning a NamedTuple of boundary fluxes.
"""
function initialize_boundary_transport(velocities::NamedTuple)

    u, v, w = velocities
    u_bcs = u.boundary_conditions
    v_bcs = v.boundary_conditions
    w_bcs = w.boundary_conditions

    boundary_fluxes = NamedTuple()
    right_scheme_boundaries = Symbol[]
    left_scheme_boundaries = Symbol[]
    total_area_pool_boundaries = zero(eltype(u))

    # Check west boundary (u velocity)
    west_flux_and_area = initialize_boundary_transport(u, u_bcs.west, Val(:west))
    boundary_fluxes = merge(boundary_fluxes, west_flux_and_area)
    if needs_transport_correction(u_bcs.west)
        push!(left_scheme_boundaries, :west)
        if needs_pool_correction(u_bcs.west)
            total_area_pool_boundaries += boundary_fluxes.west_area
        end
    end

    # Check east boundary (u velocity)
    east_flux_and_area = initialize_boundary_transport(u, u_bcs.east, Val(:east))
    boundary_fluxes = merge(boundary_fluxes, east_flux_and_area)
    if needs_transport_correction(u_bcs.east)
        push!(right_scheme_boundaries, :east)
        if needs_pool_correction(u_bcs.east)
            total_area_pool_boundaries += boundary_fluxes.east_area
        end
    end

    # Check south boundary (v velocity)
    south_flux_and_area = initialize_boundary_transport(v, v_bcs.south, Val(:south))
    boundary_fluxes = merge(boundary_fluxes, south_flux_and_area)
    if needs_transport_correction(v_bcs.south)
        push!(left_scheme_boundaries, :south)
        if needs_pool_correction(v_bcs.south)
            total_area_pool_boundaries += boundary_fluxes.south_area
        end
    end

    # Check north boundary (v velocity)
    north_flux_and_area = initialize_boundary_transport(v, v_bcs.north, Val(:north))
    boundary_fluxes = merge(boundary_fluxes, north_flux_and_area)
    if needs_transport_correction(v_bcs.north)
        push!(right_scheme_boundaries, :north)
        if needs_pool_correction(v_bcs.north)
            total_area_pool_boundaries += boundary_fluxes.north_area
        end
    end

    # Check bottom boundary (w velocity)
    bottom_flux_and_area = initialize_boundary_transport(w, w_bcs.bottom, Val(:bottom))
    boundary_fluxes = merge(boundary_fluxes, bottom_flux_and_area)
    if needs_transport_correction(w_bcs.bottom)
        push!(left_scheme_boundaries, :bottom)
        if needs_pool_correction(w_bcs.bottom)
            total_area_pool_boundaries += boundary_fluxes.bottom_area
        end
    end

    # Check top boundary (w velocity)
    top_flux_and_area = initialize_boundary_transport(w, w_bcs.top, Val(:top))
    boundary_fluxes = merge(boundary_fluxes, top_flux_and_area)
    if needs_transport_correction(w_bcs.top)
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

update_open_boundary_transport!(model) = map(compute!, model.boundary_transport)

open_boundary_transport(model, bc::OBC, ::Val{:west}, u) = @allowscalar model.boundary_transport.west_transport[]
open_boundary_transport(model, bc::OBC, ::Val{:east}, u) = @allowscalar model.boundary_transport.east_transport[]
open_boundary_transport(model, bc::OBC, ::Val{:south}, v) = @allowscalar model.boundary_transport.south_transport[]
open_boundary_transport(model, bc::OBC, ::Val{:north}, v) = @allowscalar model.boundary_transport.north_transport[]
open_boundary_transport(model, bc::OBC, ::Val{:bottom}, w) = @allowscalar model.boundary_transport.bottom_transport[]
open_boundary_transport(model, bc::OBC, ::Val{:top}, w) = @allowscalar model.boundary_transport.top_transport[]

open_boundary_transport(model, bc::ZIOBC, ::Val{:west}, u) = zero(model.grid)
open_boundary_transport(model, bc::ZIOBC, ::Val{:east}, u) = zero(model.grid)
open_boundary_transport(model, bc::ZIOBC, ::Val{:south}, v) = zero(model.grid)
open_boundary_transport(model, bc::ZIOBC, ::Val{:north}, v) = zero(model.grid)
open_boundary_transport(model, bc::ZIOBC, ::Val{:bottom}, w) = zero(model.grid)
open_boundary_transport(model, bc::ZIOBC, ::Val{:top}, w) = zero(model.grid)

open_boundary_transport(model, bc, side, velocity) = zero(model.grid)

function open_boundary_inflow_transport(model)
    update_open_boundary_transport!(model)

    u, v, w = model.velocities
    total_flux = zero(model.grid)

    # Add flux through left boundaries
    total_flux += open_boundary_transport(model, u.boundary_conditions.west, Val(:west), u)
    total_flux += open_boundary_transport(model, v.boundary_conditions.south, Val(:south), v)
    total_flux += open_boundary_transport(model, w.boundary_conditions.bottom, Val(:bottom), w)

    # Subtract flux through right boundaries.
    total_flux -= open_boundary_transport(model, u.boundary_conditions.east, Val(:east), u)
    total_flux -= open_boundary_transport(model, v.boundary_conditions.north, Val(:north), v)
    total_flux -= open_boundary_transport(model, w.boundary_conditions.top, Val(:top), w)

    return total_flux
end

# Boundaries with a target_transport are corrected independently; skip them in the pool correction.
function correct_left_boundary_transport!(u, bc::OBC, ::Val{:west}, A⁻¹_∮udA)
    has_target_transport(bc.classification.scheme) && return nothing
    interior(u, 1, :, :) .-= A⁻¹_∮udA
end
function correct_left_boundary_transport!(v, bc::OBC, ::Val{:south}, A⁻¹_∮udA)
    has_target_transport(bc.classification.scheme) && return nothing
    interior(v, :, 1, :) .-= A⁻¹_∮udA
end
function correct_left_boundary_transport!(w, bc::OBC, ::Val{:bottom}, A⁻¹_∮udA)
    has_target_transport(bc.classification.scheme) && return nothing
    interior(w, :, :, 1) .-= A⁻¹_∮udA
end
correct_left_boundary_transport!(u, bc::IOBC, ::Val{:west},   A⁻¹_∮udA) = nothing
correct_left_boundary_transport!(v, bc::IOBC, ::Val{:south},  A⁻¹_∮udA) = nothing
correct_left_boundary_transport!(w, bc::IOBC, ::Val{:bottom}, A⁻¹_∮udA) = nothing
correct_left_boundary_transport!(u, bc, side, A⁻¹_∮udA) = nothing

function correct_right_boundary_transport!(u, bc::OBC, ::Val{:east}, A⁻¹_∮udA)
    has_target_transport(bc.classification.scheme) && return nothing
    interior(u, u.grid.Nx + 1, :, :) .+= A⁻¹_∮udA
end
function correct_right_boundary_transport!(v, bc::OBC, ::Val{:north}, A⁻¹_∮udA)
    has_target_transport(bc.classification.scheme) && return nothing
    interior(v, :, v.grid.Ny + 1, :) .+= A⁻¹_∮udA
end
function correct_right_boundary_transport!(w, bc::OBC, ::Val{:top}, A⁻¹_∮udA)
    has_target_transport(bc.classification.scheme) && return nothing
    interior(w, :, :, w.grid.Nz + 1) .+= A⁻¹_∮udA
end
correct_right_boundary_transport!(u, bc::IOBC, ::Val{:east},  A⁻¹_∮udA) = nothing
correct_right_boundary_transport!(v, bc::IOBC, ::Val{:north}, A⁻¹_∮udA) = nothing
correct_right_boundary_transport!(w, bc::IOBC, ::Val{:top},   A⁻¹_∮udA) = nothing
correct_right_boundary_transport!(u, bc, side, A⁻¹_∮udA) = nothing

# Apply a per-boundary correction to reach the prescribed target_transport.
# target_transport is the desired integral of the normal velocity in the positive
# coordinate direction (e.g., eastward for west/east boundaries).
function apply_targeted_left_boundary_correction!(u, bc::OBC, ::Val{:west}, boundary_transport)
    has_target_transport(bc.classification.scheme) || return nothing
    west_area = boundary_total_area(Val(:west), u.grid)
    target    = get_target_transport(bc.classification.scheme, u.grid)
    Q_actual  = @allowscalar boundary_transport.west_transport[]
    interior(u, 1, :, :) .-= (Q_actual - target) / west_area
    return nothing
end

function apply_targeted_left_boundary_correction!(v, bc::OBC, ::Val{:south}, boundary_transport)
    has_target_transport(bc.classification.scheme) || return nothing
    south_area = boundary_total_area(Val(:south), v.grid)
    target     = get_target_transport(bc.classification.scheme, v.grid)
    Q_actual   = @allowscalar boundary_transport.south_transport[]
    interior(v, :, 1, :) .-= (Q_actual - target) / south_area
    return nothing
end

function apply_targeted_left_boundary_correction!(w, bc::OBC, ::Val{:bottom}, boundary_transport)
    has_target_transport(bc.classification.scheme) || return nothing
    bottom_area = boundary_total_area(Val(:bottom), w.grid)
    target      = get_target_transport(bc.classification.scheme, w.grid)
    Q_actual    = @allowscalar boundary_transport.bottom_transport[]
    interior(w, :, :, 1) .-= (Q_actual - target) / bottom_area
    return nothing
end

function apply_targeted_right_boundary_correction!(u, bc::OBC, ::Val{:east}, boundary_transport)
    has_target_transport(bc.classification.scheme) || return nothing
    east_area = boundary_total_area(Val(:east), u.grid)
    target    = get_target_transport(bc.classification.scheme, u.grid)
    Q_actual  = @allowscalar boundary_transport.east_transport[]
    interior(u, u.grid.Nx + 1, :, :) .-= (Q_actual - target) / east_area
    return nothing
end

function apply_targeted_right_boundary_correction!(v, bc::OBC, ::Val{:north}, boundary_transport)
    has_target_transport(bc.classification.scheme) || return nothing
    north_area = boundary_total_area(Val(:north), v.grid)
    target     = get_target_transport(bc.classification.scheme, v.grid)
    Q_actual   = @allowscalar boundary_transport.north_transport[]
    interior(v, :, v.grid.Ny + 1, :) .-= (Q_actual - target) / north_area
    return nothing
end

function apply_targeted_right_boundary_correction!(w, bc::OBC, ::Val{:top}, boundary_transport)
    has_target_transport(bc.classification.scheme) || return nothing
    top_area = boundary_total_area(Val(:top), w.grid)
    target   = get_target_transport(bc.classification.scheme, w.grid)
    Q_actual = @allowscalar boundary_transport.top_transport[]
    interior(w, :, :, w.grid.Nz + 1) .-= (Q_actual - target) / top_area
    return nothing
end

apply_targeted_left_boundary_correction!(velocity, bc, side, boundary_transport) = nothing
apply_targeted_right_boundary_correction!(velocity, bc, side, boundary_transport) = nothing

enforce_net_zero_transport!(model, ::Nothing) = nothing

"""
    enforce_net_zero_transport!(model, boundary_transport)

Correct boundary velocities for open boundary conditions to enforce volume conservation.

Boundaries with a `target_transport` set on their scheme are corrected first: a uniform
velocity adjustment is applied so that the net transport through that boundary equals
`target_transport`. The remaining net imbalance across all open boundaries is then
distributed uniformly over the pool boundaries (those without a `target_transport`).
"""
function enforce_net_zero_transport!(model, boundary_transport)
    u, v, w = model.velocities

    # Step 1: compute all boundary fluxes before any corrections.
    update_open_boundary_transport!(model)

    # Step 2: apply per-boundary corrections for targeted boundaries.
    apply_targeted_left_boundary_correction!(u, u.boundary_conditions.west,   Val(:west),   boundary_transport)
    apply_targeted_left_boundary_correction!(v, v.boundary_conditions.south,  Val(:south),  boundary_transport)
    apply_targeted_left_boundary_correction!(w, w.boundary_conditions.bottom, Val(:bottom), boundary_transport)
    apply_targeted_right_boundary_correction!(u, u.boundary_conditions.east,  Val(:east),   boundary_transport)
    apply_targeted_right_boundary_correction!(v, v.boundary_conditions.north, Val(:north),  boundary_transport)
    apply_targeted_right_boundary_correction!(w, w.boundary_conditions.top,   Val(:top),    boundary_transport)

    # Step 3: distribute remaining imbalance over pool boundaries.
    A = boundary_transport.total_area_pool_boundaries
    iszero(A) && return nothing

    ∮udA = open_boundary_inflow_transport(model)
    A⁻¹_∮udA = ∮udA / A

    correct_left_boundary_transport!(u, u.boundary_conditions.west,   Val(:west),   A⁻¹_∮udA)
    correct_left_boundary_transport!(v, v.boundary_conditions.south,  Val(:south),  A⁻¹_∮udA)
    correct_left_boundary_transport!(w, w.boundary_conditions.bottom, Val(:bottom), A⁻¹_∮udA)
    correct_right_boundary_transport!(u, u.boundary_conditions.east,  Val(:east),   A⁻¹_∮udA)
    correct_right_boundary_transport!(v, v.boundary_conditions.north, Val(:north),  A⁻¹_∮udA)
    correct_right_boundary_transport!(w, w.boundary_conditions.top,   Val(:top),    A⁻¹_∮udA)
end
