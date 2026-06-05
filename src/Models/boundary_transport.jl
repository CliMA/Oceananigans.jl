using Oceananigans.BoundaryConditions: BoundaryCondition, NormalFlow, has_target_transport, get_target_transport
using Oceananigans.AbstractOperations: Integral, Ax, Ay, Az, grid_metric_operation
using Oceananigans.Fields: Field, interior, compute!
using GPUArraysCore: @allowscalar

const NFBC = BoundaryCondition{<:NormalFlow} # "Radiantion" NormalFlowBoundaryCondition (with a matching scheme)
const INFBC = BoundaryCondition{<:NormalFlow{<:Nothing}} # "Imposed-velocity" NormalFlowBoundaryCondition (with no scheme)
const FINFBC = BoundaryCondition{<:NormalFlow{<:Nothing}, <:Number} # "Fixed-imposed-velocity" NormalFlowBoundaryCondition
const ZINFBC = BoundaryCondition{<:NormalFlow{<:Nothing}, <:Nothing} # "Zero-imposed-velocity" NormalFlowBoundaryCondition (no-inflow)

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
@inline west_transport(u)   = Field(Integral(view(u, 1, :, :), dims=(2, 3)))
@inline south_transport(v)  = Field(Integral(view(v, :, 1, :), dims=(1, 3)))
@inline bottom_transport(w) = Field(Integral(view(w, :, :, 1), dims=(1, 2)))

# Right boundary integrals for normal velocity / momentum components
@inline east_transport(u)   = Field(Integral(view(u, u.grid.Nx + 1, :, :), dims=(2, 3)))
@inline north_transport(v)  = Field(Integral(view(v, :, v.grid.Ny + 1, :), dims=(1, 3)))
@inline top_transport(w)    = Field(Integral(view(w, :, :, w.grid.Nz + 1), dims=(1, 2)))

initialize_side_transport(u, bc::NFBC, ::Val{:west})   = (; west_transport = west_transport(u), west_area = get_west_area(u.grid))
initialize_side_transport(u, bc::NFBC, ::Val{:east})   = (; east_transport = east_transport(u), east_area = get_east_area(u.grid))
initialize_side_transport(v, bc::NFBC, ::Val{:south})  = (; south_transport = south_transport(v), south_area = get_south_area(v.grid))
initialize_side_transport(v, bc::NFBC, ::Val{:north})  = (; north_transport = north_transport(v), north_area = get_north_area(v.grid))
initialize_side_transport(w, bc::NFBC, ::Val{:bottom}) = (; bottom_transport = bottom_transport(w), bottom_area = get_bottom_area(w.grid))
initialize_side_transport(w, bc::NFBC, ::Val{:top})    = (; top_transport = top_transport(w), top_area = get_top_area(w.grid))

initialize_side_transport(u, bc::ZINFBC, ::Val{:west})   = NamedTuple()
initialize_side_transport(u, bc::ZINFBC, ::Val{:east})   = NamedTuple()
initialize_side_transport(v, bc::ZINFBC, ::Val{:south})  = NamedTuple()
initialize_side_transport(v, bc::ZINFBC, ::Val{:north})  = NamedTuple()
initialize_side_transport(w, bc::ZINFBC, ::Val{:bottom}) = NamedTuple()
initialize_side_transport(w, bc::ZINFBC, ::Val{:top})    = NamedTuple()

initialize_side_transport(u, bc::FINFBC, ::Val{:west})   = (; west_transport = bc.condition * get_west_area(u.grid), west_area = get_west_area(u.grid))
initialize_side_transport(u, bc::FINFBC, ::Val{:east})   = (; east_transport = bc.condition * get_east_area(u.grid), east_area = get_east_area(u.grid))
initialize_side_transport(v, bc::FINFBC, ::Val{:south})  = (; south_transport = bc.condition * get_south_area(v.grid), south_area = get_south_area(v.grid))
initialize_side_transport(v, bc::FINFBC, ::Val{:north})  = (; north_transport = bc.condition * get_north_area(v.grid), north_area = get_north_area(v.grid))
initialize_side_transport(w, bc::FINFBC, ::Val{:bottom}) = (; bottom_transport = bc.condition * get_bottom_area(w.grid), bottom_area = get_bottom_area(w.grid))
initialize_side_transport(w, bc::FINFBC, ::Val{:top})    = (; top_transport = bc.condition * get_top_area(w.grid), top_area = get_top_area(w.grid))

initialize_side_transport(velocity, ::Nothing, side) = NamedTuple()
initialize_side_transport(velocity, bc, side) = NamedTuple()

needs_transport_correction(::INFBC) = false
needs_transport_correction(::NFBC) = true
needs_transport_correction(::Nothing) = false
needs_transport_correction(bc) = false

# True for scheme boundaries that participate in the global pool (net-zero) correction.
# Boundaries with a `target_transport` are corrected independently and excluded from the pool.
needs_pool_correction(::INFBC) = false
needs_pool_correction(bc::NFBC) = !has_target_transport(bc.classification.scheme)
needs_pool_correction(::Nothing) = false
needs_pool_correction(bc) = false

"""
    initialize_boundary_transport(velocities::NamedTuple)

Initialize boundary transports for boundaries with `NormalFlowBoundaryCondition`s,
returning a `NamedTuple` of boundary transports and areas, or `nothing` when no
open boundary requires correction.

The `velocities` argument is a `NamedTuple` of three face-normal fields.
Destructuring is positional, so any 3-field `NamedTuple` works:
`(; u, v, w)` for incompressible / volume-flux models like `NonhydrostaticModel`,
or `(; ρu, ρv, ρw)` for anelastic / mass-flux models.
"""
function initialize_boundary_transport(velocities::NamedTuple)

    u, v, w = velocities
    u_bcs = u.boundary_conditions
    v_bcs = v.boundary_conditions
    w_bcs = w.boundary_conditions

    boundary_transports = NamedTuple()
    right_scheme_boundaries = Symbol[]
    left_scheme_boundaries = Symbol[]
    total_area_scheme_boundaries = zero(eltype(u))
    total_area_pool_boundaries = zero(eltype(u))

    # Check west boundary (u velocity)
    west_transport_and_area = initialize_side_transport(u, u_bcs.west, Val(:west))
    boundary_transports = merge(boundary_transports, west_transport_and_area)
    if needs_transport_correction(u_bcs.west)
        push!(left_scheme_boundaries, :west)
        total_area_scheme_boundaries += boundary_transports.west_area
        needs_pool_correction(u_bcs.west) && (total_area_pool_boundaries += boundary_transports.west_area)
    end

    # Check east boundary (u velocity)
    east_transport_and_area = initialize_side_transport(u, u_bcs.east, Val(:east))
    boundary_transports = merge(boundary_transports, east_transport_and_area)
    if needs_transport_correction(u_bcs.east)
        push!(right_scheme_boundaries, :east)
        total_area_scheme_boundaries += boundary_transports.east_area
        needs_pool_correction(u_bcs.east) && (total_area_pool_boundaries += boundary_transports.east_area)
    end

    # Check south boundary (v velocity)
    south_transport_and_area = initialize_side_transport(v, v_bcs.south, Val(:south))
    boundary_transports = merge(boundary_transports, south_transport_and_area)
    if needs_transport_correction(v_bcs.south)
        push!(left_scheme_boundaries, :south)
        total_area_scheme_boundaries += boundary_transports.south_area
        needs_pool_correction(v_bcs.south) && (total_area_pool_boundaries += boundary_transports.south_area)
    end

    # Check north boundary (v velocity)
    north_transport_and_area = initialize_side_transport(v, v_bcs.north, Val(:north))
    boundary_transports = merge(boundary_transports, north_transport_and_area)
    if needs_transport_correction(v_bcs.north)
        push!(right_scheme_boundaries, :north)
        total_area_scheme_boundaries += boundary_transports.north_area
        needs_pool_correction(v_bcs.north) && (total_area_pool_boundaries += boundary_transports.north_area)
    end

    # Check bottom boundary (w velocity)
    bottom_transport_and_area = initialize_side_transport(w, w_bcs.bottom, Val(:bottom))
    boundary_transports = merge(boundary_transports, bottom_transport_and_area)
    if needs_transport_correction(w_bcs.bottom)
        push!(left_scheme_boundaries, :bottom)
        total_area_scheme_boundaries += boundary_transports.bottom_area
        needs_pool_correction(w_bcs.bottom) && (total_area_pool_boundaries += boundary_transports.bottom_area)
    end

    # Check top boundary (w velocity)
    top_transport_and_area = initialize_side_transport(w, w_bcs.top, Val(:top))
    boundary_transports = merge(boundary_transports, top_transport_and_area)
    if needs_transport_correction(w_bcs.top)
        push!(right_scheme_boundaries, :top)
        total_area_scheme_boundaries += boundary_transports.top_area
        needs_pool_correction(w_bcs.top) && (total_area_pool_boundaries += boundary_transports.top_area)
    end

    boundary_transports = merge(boundary_transports, (; left_scheme_boundaries = Tuple(left_scheme_boundaries),
                                                        right_scheme_boundaries = Tuple(right_scheme_boundaries),
                                                        total_area_scheme_boundaries,
                                                        total_area_pool_boundaries))

    if length(boundary_transports.left_scheme_boundaries) == 0 && length(boundary_transports.right_scheme_boundaries) == 0
        return nothing
    else
        return boundary_transports
    end
end

update_open_boundary_transport!(boundary_transport) = map(compute!, boundary_transport)

open_boundary_transport(bt, ::NFBC, ::Val{:west})   = @allowscalar bt.west_transport[]
open_boundary_transport(bt, ::NFBC, ::Val{:east})   = @allowscalar bt.east_transport[]
open_boundary_transport(bt, ::NFBC, ::Val{:south})  = @allowscalar bt.south_transport[]
open_boundary_transport(bt, ::NFBC, ::Val{:north})  = @allowscalar bt.north_transport[]
open_boundary_transport(bt, ::NFBC, ::Val{:bottom}) = @allowscalar bt.bottom_transport[]
open_boundary_transport(bt, ::NFBC, ::Val{:top})    = @allowscalar bt.top_transport[]

# Imposed-velocity NormalFlow BCs (no scheme) and non-open BCs contribute zero to the
# net boundary transport: imposed-velocity outflow is already fixed, and closed
# boundaries carry no transport by definition. Method-per-side avoids ambiguity
# with the `::NFBC, ::Val{:<side>}` methods above (ZINFBC <: INFBC <: NFBC).
open_boundary_transport(bt, ::ZINFBC, ::Val{:west})   = zero(bt.total_area_scheme_boundaries)
open_boundary_transport(bt, ::ZINFBC, ::Val{:east})   = zero(bt.total_area_scheme_boundaries)
open_boundary_transport(bt, ::ZINFBC, ::Val{:south})  = zero(bt.total_area_scheme_boundaries)
open_boundary_transport(bt, ::ZINFBC, ::Val{:north})  = zero(bt.total_area_scheme_boundaries)
open_boundary_transport(bt, ::ZINFBC, ::Val{:bottom}) = zero(bt.total_area_scheme_boundaries)
open_boundary_transport(bt, ::ZINFBC, ::Val{:top})    = zero(bt.total_area_scheme_boundaries)

open_boundary_transport(bt, bc, side) = zero(bt.total_area_scheme_boundaries)

function open_boundary_inflow_transport(boundary_transport, velocities)
    update_open_boundary_transport!(boundary_transport)

    u, v, w = velocities
    total_transport = zero(boundary_transport.total_area_scheme_boundaries)

    # Add transport through left boundaries
    total_transport += open_boundary_transport(boundary_transport, u.boundary_conditions.west,   Val(:west))
    total_transport += open_boundary_transport(boundary_transport, v.boundary_conditions.south,  Val(:south))
    total_transport += open_boundary_transport(boundary_transport, w.boundary_conditions.bottom, Val(:bottom))

    # Subtract transport through right boundaries.
    total_transport -= open_boundary_transport(boundary_transport, u.boundary_conditions.east,  Val(:east))
    total_transport -= open_boundary_transport(boundary_transport, v.boundary_conditions.north, Val(:north))
    total_transport -= open_boundary_transport(boundary_transport, w.boundary_conditions.top,   Val(:top))

    return total_transport
end

# Pool-correction methods. Boundaries with a `target_transport` are corrected
# independently (see `apply_targeted_*_boundary_correction!` below); skip them
# here so the pool correction only touches non-targeted boundaries.
function correct_left_boundary_transport!(u, bc::NFBC, ::Val{:west}, A⁻¹_∮udA)
    has_target_transport(bc.classification.scheme) && return nothing
    interior(u, 1, :, :) .-= A⁻¹_∮udA
end
function correct_left_boundary_transport!(v, bc::NFBC, ::Val{:south}, A⁻¹_∮udA)
    has_target_transport(bc.classification.scheme) && return nothing
    interior(v, :, 1, :) .-= A⁻¹_∮udA
end
function correct_left_boundary_transport!(w, bc::NFBC, ::Val{:bottom}, A⁻¹_∮udA)
    has_target_transport(bc.classification.scheme) && return nothing
    interior(w, :, :, 1) .-= A⁻¹_∮udA
end
correct_left_boundary_transport!(u, bc::INFBC, ::Val{:west},   A⁻¹_∮udA) = nothing
correct_left_boundary_transport!(v, bc::INFBC, ::Val{:south},  A⁻¹_∮udA) = nothing
correct_left_boundary_transport!(w, bc::INFBC, ::Val{:bottom}, A⁻¹_∮udA) = nothing
correct_left_boundary_transport!(u, bc, side, A⁻¹_∮udA) = nothing

function correct_right_boundary_transport!(u, bc::NFBC, ::Val{:east}, A⁻¹_∮udA)
    has_target_transport(bc.classification.scheme) && return nothing
    interior(u, u.grid.Nx + 1, :, :) .+= A⁻¹_∮udA
end
function correct_right_boundary_transport!(v, bc::NFBC, ::Val{:north}, A⁻¹_∮udA)
    has_target_transport(bc.classification.scheme) && return nothing
    interior(v, :, v.grid.Ny + 1, :) .+= A⁻¹_∮udA
end
function correct_right_boundary_transport!(w, bc::NFBC, ::Val{:top}, A⁻¹_∮udA)
    has_target_transport(bc.classification.scheme) && return nothing
    interior(w, :, :, w.grid.Nz + 1) .+= A⁻¹_∮udA
end
correct_right_boundary_transport!(u, bc::INFBC, ::Val{:east},  A⁻¹_∮udA) = nothing
correct_right_boundary_transport!(v, bc::INFBC, ::Val{:north}, A⁻¹_∮udA) = nothing
correct_right_boundary_transport!(w, bc::INFBC, ::Val{:top},   A⁻¹_∮udA) = nothing
correct_right_boundary_transport!(u, bc, side, A⁻¹_∮udA) = nothing

# Apply a per-boundary correction to reach the prescribed `target_transport`.
# The boundary velocity is adjusted by a uniform shift so that the integral of the
# normal velocity over the boundary matches `target_transport` (positive in the
# positive coordinate direction — e.g., eastward through west/east boundaries).
function apply_targeted_left_boundary_correction!(u, bc::NFBC, ::Val{:west}, boundary_transport)
    has_target_transport(bc.classification.scheme) || return nothing
    west_area = get_west_area(u.grid)
    target    = get_target_transport(bc.classification.scheme, u.grid)
    Q_actual  = @allowscalar boundary_transport.west_transport[]
    interior(u, 1, :, :) .-= (Q_actual - target) / west_area
    return nothing
end

function apply_targeted_left_boundary_correction!(v, bc::NFBC, ::Val{:south}, boundary_transport)
    has_target_transport(bc.classification.scheme) || return nothing
    south_area = get_south_area(v.grid)
    target     = get_target_transport(bc.classification.scheme, v.grid)
    Q_actual   = @allowscalar boundary_transport.south_transport[]
    interior(v, :, 1, :) .-= (Q_actual - target) / south_area
    return nothing
end

function apply_targeted_left_boundary_correction!(w, bc::NFBC, ::Val{:bottom}, boundary_transport)
    has_target_transport(bc.classification.scheme) || return nothing
    bottom_area = get_bottom_area(w.grid)
    target      = get_target_transport(bc.classification.scheme, w.grid)
    Q_actual    = @allowscalar boundary_transport.bottom_transport[]
    interior(w, :, :, 1) .-= (Q_actual - target) / bottom_area
    return nothing
end

function apply_targeted_right_boundary_correction!(u, bc::NFBC, ::Val{:east}, boundary_transport)
    has_target_transport(bc.classification.scheme) || return nothing
    east_area = get_east_area(u.grid)
    target    = get_target_transport(bc.classification.scheme, u.grid)
    Q_actual  = @allowscalar boundary_transport.east_transport[]
    interior(u, u.grid.Nx + 1, :, :) .-= (Q_actual - target) / east_area
    return nothing
end

function apply_targeted_right_boundary_correction!(v, bc::NFBC, ::Val{:north}, boundary_transport)
    has_target_transport(bc.classification.scheme) || return nothing
    north_area = get_north_area(v.grid)
    target     = get_target_transport(bc.classification.scheme, v.grid)
    Q_actual   = @allowscalar boundary_transport.north_transport[]
    interior(v, :, v.grid.Ny + 1, :) .-= (Q_actual - target) / north_area
    return nothing
end

function apply_targeted_right_boundary_correction!(w, bc::NFBC, ::Val{:top}, boundary_transport)
    has_target_transport(bc.classification.scheme) || return nothing
    top_area = get_top_area(w.grid)
    target   = get_target_transport(bc.classification.scheme, w.grid)
    Q_actual = @allowscalar boundary_transport.top_transport[]
    interior(w, :, :, w.grid.Nz + 1) .-= (Q_actual - target) / top_area
    return nothing
end

apply_targeted_left_boundary_correction!(velocity, bc, side, boundary_transport)  = nothing
apply_targeted_right_boundary_correction!(velocity, bc, side, boundary_transport) = nothing
