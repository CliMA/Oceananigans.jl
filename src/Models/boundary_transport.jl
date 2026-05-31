using Oceananigans.BoundaryConditions: BoundaryCondition, NormalFlow
using Oceananigans.AbstractOperations: Integral, Ax, Ay, Az, grid_metric_operation
using Oceananigans.Fields: Field, interior, compute!
using GPUArraysCore: @allowscalar

const OBC  = BoundaryCondition{<:NormalFlow} # OpenBoundaryCondition
const IOBC = BoundaryCondition{<:NormalFlow{<:Nothing}} # "Imposed-velocity" OpenBoundaryCondition (with no scheme)
const FIOBC = BoundaryCondition{<:NormalFlow{<:Nothing}, <:Number} # "Fixed-imposed-velocity" OpenBoundaryCondition
const ZIOBC = BoundaryCondition{<:NormalFlow{<:Nothing}, <:Nothing} # "Zero-imposed-velocity" OpenBoundaryCondition (no-inflow)

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

initialize_side_transport(u, bc::OBC, ::Val{:west})   = (; west_transport = west_transport(u), west_area = get_west_area(u.grid))
initialize_side_transport(u, bc::OBC, ::Val{:east})   = (; east_transport = east_transport(u), east_area = get_east_area(u.grid))
initialize_side_transport(v, bc::OBC, ::Val{:south})  = (; south_transport = south_transport(v), south_area = get_south_area(v.grid))
initialize_side_transport(v, bc::OBC, ::Val{:north})  = (; north_transport = north_transport(v), north_area = get_north_area(v.grid))
initialize_side_transport(w, bc::OBC, ::Val{:bottom}) = (; bottom_transport = bottom_transport(w), bottom_area = get_bottom_area(w.grid))
initialize_side_transport(w, bc::OBC, ::Val{:top})    = (; top_transport = top_transport(w), top_area = get_top_area(w.grid))

initialize_side_transport(u, bc::ZIOBC, ::Val{:west})   = NamedTuple()
initialize_side_transport(u, bc::ZIOBC, ::Val{:east})   = NamedTuple()
initialize_side_transport(v, bc::ZIOBC, ::Val{:south})  = NamedTuple()
initialize_side_transport(v, bc::ZIOBC, ::Val{:north})  = NamedTuple()
initialize_side_transport(w, bc::ZIOBC, ::Val{:bottom}) = NamedTuple()
initialize_side_transport(w, bc::ZIOBC, ::Val{:top})    = NamedTuple()

initialize_side_transport(u, bc::FIOBC, ::Val{:west})   = (; west_transport = bc.condition * get_west_area(u.grid), west_area = get_west_area(u.grid))
initialize_side_transport(u, bc::FIOBC, ::Val{:east})   = (; east_transport = bc.condition * get_east_area(u.grid), east_area = get_east_area(u.grid))
initialize_side_transport(v, bc::FIOBC, ::Val{:south})  = (; south_transport = bc.condition * get_south_area(v.grid), south_area = get_south_area(v.grid))
initialize_side_transport(v, bc::FIOBC, ::Val{:north})  = (; north_transport = bc.condition * get_north_area(v.grid), north_area = get_north_area(v.grid))
initialize_side_transport(w, bc::FIOBC, ::Val{:bottom}) = (; bottom_transport = bc.condition * get_bottom_area(w.grid), bottom_area = get_bottom_area(w.grid))
initialize_side_transport(w, bc::FIOBC, ::Val{:top})    = (; top_transport = bc.condition * get_top_area(w.grid), top_area = get_top_area(w.grid))

initialize_side_transport(velocity, ::Nothing, side) = NamedTuple()
initialize_side_transport(velocity, bc, side) = NamedTuple()

needs_transport_correction(::IOBC) = false
needs_transport_correction(::OBC) = true
needs_transport_correction(::Nothing) = false
needs_transport_correction(bc) = false

"""
    initialize_boundary_transport(velocities::NamedTuple)

Initialize boundary transports for boundaries with `OpenBoundaryCondition`s,
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

    # Check west boundary (u velocity)
    west_transport_and_area = initialize_side_transport(u, u_bcs.west, Val(:west))
    boundary_transports = merge(boundary_transports, west_transport_and_area)
    if needs_transport_correction(u_bcs.west)
        push!(left_scheme_boundaries, :west)
        total_area_scheme_boundaries += boundary_transports.west_area
    end

    # Check east boundary (u velocity)
    east_transport_and_area = initialize_side_transport(u, u_bcs.east, Val(:east))
    boundary_transports = merge(boundary_transports, east_transport_and_area)
    if needs_transport_correction(u_bcs.east)
        push!(right_scheme_boundaries, :east)
        total_area_scheme_boundaries += boundary_transports.east_area
    end

    # Check south boundary (v velocity)
    south_transport_and_area = initialize_side_transport(v, v_bcs.south, Val(:south))
    boundary_transports = merge(boundary_transports, south_transport_and_area)
    if needs_transport_correction(v_bcs.south)
        push!(left_scheme_boundaries, :south)
        total_area_scheme_boundaries += boundary_transports.south_area
    end

    # Check north boundary (v velocity)
    north_transport_and_area = initialize_side_transport(v, v_bcs.north, Val(:north))
    boundary_transports = merge(boundary_transports, north_transport_and_area)
    if needs_transport_correction(v_bcs.north)
        push!(right_scheme_boundaries, :north)
        total_area_scheme_boundaries += boundary_transports.north_area
    end

    # Check bottom boundary (w velocity)
    bottom_transport_and_area = initialize_side_transport(w, w_bcs.bottom, Val(:bottom))
    boundary_transports = merge(boundary_transports, bottom_transport_and_area)
    if needs_transport_correction(w_bcs.bottom)
        push!(left_scheme_boundaries, :bottom)
        total_area_scheme_boundaries += boundary_transports.bottom_area
    end

    # Check top boundary (w velocity)
    top_transport_and_area = initialize_side_transport(w, w_bcs.top, Val(:top))
    boundary_transports = merge(boundary_transports, top_transport_and_area)
    if needs_transport_correction(w_bcs.top)
        push!(right_scheme_boundaries, :top)
        total_area_scheme_boundaries += boundary_transports.top_area
    end

    boundary_transports = merge(boundary_transports, (; left_scheme_boundaries = Tuple(left_scheme_boundaries),
                                                        right_scheme_boundaries = Tuple(right_scheme_boundaries),
                                                        total_area_scheme_boundaries))

    if length(boundary_transports.left_scheme_boundaries) == 0 && length(boundary_transports.right_scheme_boundaries) == 0
        return nothing
    else
        return boundary_transports
    end
end

update_open_boundary_transport!(boundary_transport) = map(compute!, boundary_transport)

open_boundary_transport(bt, ::OBC, ::Val{:west})   = @allowscalar bt.west_transport[]
open_boundary_transport(bt, ::OBC, ::Val{:east})   = @allowscalar bt.east_transport[]
open_boundary_transport(bt, ::OBC, ::Val{:south})  = @allowscalar bt.south_transport[]
open_boundary_transport(bt, ::OBC, ::Val{:north})  = @allowscalar bt.north_transport[]
open_boundary_transport(bt, ::OBC, ::Val{:bottom}) = @allowscalar bt.bottom_transport[]
open_boundary_transport(bt, ::OBC, ::Val{:top})    = @allowscalar bt.top_transport[]

# Imposed-velocity Open BCs (no scheme) and non-open BCs contribute zero to the
# net boundary transport: imposed-velocity outflow is already fixed, and closed
# boundaries carry no transport by definition. Method-per-side avoids ambiguity
# with the `::OBC, ::Val{:<side>}` methods above (ZIOBC <: IOBC <: OBC).
open_boundary_transport(bt, ::ZIOBC, ::Val{:west})   = zero(bt.total_area_scheme_boundaries)
open_boundary_transport(bt, ::ZIOBC, ::Val{:east})   = zero(bt.total_area_scheme_boundaries)
open_boundary_transport(bt, ::ZIOBC, ::Val{:south})  = zero(bt.total_area_scheme_boundaries)
open_boundary_transport(bt, ::ZIOBC, ::Val{:north})  = zero(bt.total_area_scheme_boundaries)
open_boundary_transport(bt, ::ZIOBC, ::Val{:bottom}) = zero(bt.total_area_scheme_boundaries)
open_boundary_transport(bt, ::ZIOBC, ::Val{:top})    = zero(bt.total_area_scheme_boundaries)

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

correct_left_boundary_transport!(u, bc::OBC, ::Val{:west},    A⁻¹_∮udA) = interior(u, 1, :, :) .-= A⁻¹_∮udA
correct_left_boundary_transport!(v, bc::OBC, ::Val{:south},   A⁻¹_∮udA) = interior(v, :, 1, :) .-= A⁻¹_∮udA
correct_left_boundary_transport!(w, bc::OBC, ::Val{:bottom},  A⁻¹_∮udA) = interior(w, :, :, 1) .-= A⁻¹_∮udA
correct_left_boundary_transport!(u, bc::IOBC, ::Val{:west},   A⁻¹_∮udA) = nothing
correct_left_boundary_transport!(v, bc::IOBC, ::Val{:south},  A⁻¹_∮udA) = nothing
correct_left_boundary_transport!(w, bc::IOBC, ::Val{:bottom}, A⁻¹_∮udA) = nothing
correct_left_boundary_transport!(u, bc, side, A⁻¹_∮udA) = nothing

correct_right_boundary_transport!(u, bc::OBC, ::Val{:east},   A⁻¹_∮udA) = interior(u, u.grid.Nx + 1, :, :) .+= A⁻¹_∮udA
correct_right_boundary_transport!(v, bc::OBC, ::Val{:north},  A⁻¹_∮udA) = interior(v, :, v.grid.Ny + 1, :) .+= A⁻¹_∮udA
correct_right_boundary_transport!(w, bc::OBC, ::Val{:top},    A⁻¹_∮udA) = interior(w, :, :, w.grid.Nz + 1) .+= A⁻¹_∮udA
correct_right_boundary_transport!(u, bc::IOBC, ::Val{:east},  A⁻¹_∮udA) = nothing
correct_right_boundary_transport!(v, bc::IOBC, ::Val{:north}, A⁻¹_∮udA) = nothing
correct_right_boundary_transport!(w, bc::IOBC, ::Val{:top},   A⁻¹_∮udA) = nothing
correct_right_boundary_transport!(u, bc, side, A⁻¹_∮udA) = nothing
