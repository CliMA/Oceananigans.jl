using GPUArraysCore: @allowscalar
using KernelAbstractions: @index, @kernel

using Oceananigans.AbstractOperations: Ax, Ay, Az, Integral, grid_metric_operation
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: BoundaryCondition, NormalFlow, get_target_transport, has_target_transport
using Oceananigans.Fields: Field, compute!
using Oceananigans.Grids: Center, Face, immersed_peripheral_node
using Oceananigans.Utils: KernelParameters, launch!

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
$(TYPEDSIGNATURES)

Initialize boundary transports for boundaries with `NormalFlowBoundaryCondition`s,
returning a `NamedTuple` of boundary transports and areas, or `nothing` when no
boundary requires correction. The result is the `boundary_transport` argument of
[`enforce_net_zero_transport!`](@ref).

The `velocities` argument is a `NamedTuple` of three face-normal fields.
Destructuring is positional, so any 3-field `NamedTuple` works:
`(; u, v, w)` for incompressible / volume-flux models like `NonhydrostaticModel`,
or `(; ρu, ρv, ρw)` for anelastic / mass-flux models, whose transports are then mass fluxes.

A boundary requires correction when its `NormalFlowBoundaryCondition` has a radiation `scheme`
or was built with `rebalance=true`. An imposed `NormalFlowBoundaryCondition` without either is
never adjusted: a fixed value contributes its transport to the balance, and a zero (impenetrable)
one contributes nothing. On an `ImmersedBoundaryGrid` the transports and areas count only the
faces that are not on the immersed boundary.
"""
function initialize_boundary_transport(velocities::NamedTuple)

    u, v, w = velocities
    u_bcs = u.boundary_conditions
    v_bcs = v.boundary_conditions
    w_bcs = w.boundary_conditions

    boundary_transports = NamedTuple()
    right_scheme_boundaries = ()
    left_scheme_boundaries = ()
    total_area_scheme_boundaries = zero(eltype(u))
    total_area_pool_boundaries = zero(eltype(u))

    # Check west boundary (u velocity)
    west_transport_and_area = initialize_side_transport(u, u_bcs.west, Val(:west))
    boundary_transports = merge(boundary_transports, west_transport_and_area)
    if needs_transport_correction(u_bcs.west)
        left_scheme_boundaries = (left_scheme_boundaries..., :west)
        total_area_scheme_boundaries += boundary_transports.west_area
        needs_pool_correction(u_bcs.west) && (total_area_pool_boundaries += boundary_transports.west_area)
    end

    # Check east boundary (u velocity)
    east_transport_and_area = initialize_side_transport(u, u_bcs.east, Val(:east))
    boundary_transports = merge(boundary_transports, east_transport_and_area)
    if needs_transport_correction(u_bcs.east)
        right_scheme_boundaries = (right_scheme_boundaries..., :east)
        total_area_scheme_boundaries += boundary_transports.east_area
        needs_pool_correction(u_bcs.east) && (total_area_pool_boundaries += boundary_transports.east_area)
    end

    # Check south boundary (v velocity)
    south_transport_and_area = initialize_side_transport(v, v_bcs.south, Val(:south))
    boundary_transports = merge(boundary_transports, south_transport_and_area)
    if needs_transport_correction(v_bcs.south)
        left_scheme_boundaries = (left_scheme_boundaries..., :south)
        total_area_scheme_boundaries += boundary_transports.south_area
        needs_pool_correction(v_bcs.south) && (total_area_pool_boundaries += boundary_transports.south_area)
    end

    # Check north boundary (v velocity)
    north_transport_and_area = initialize_side_transport(v, v_bcs.north, Val(:north))
    boundary_transports = merge(boundary_transports, north_transport_and_area)
    if needs_transport_correction(v_bcs.north)
        right_scheme_boundaries = (right_scheme_boundaries..., :north)
        total_area_scheme_boundaries += boundary_transports.north_area
        needs_pool_correction(v_bcs.north) && (total_area_pool_boundaries += boundary_transports.north_area)
    end

    # Check bottom boundary (w velocity)
    bottom_transport_and_area = initialize_side_transport(w, w_bcs.bottom, Val(:bottom))
    boundary_transports = merge(boundary_transports, bottom_transport_and_area)
    if needs_transport_correction(w_bcs.bottom)
        left_scheme_boundaries = (left_scheme_boundaries..., :bottom)
        total_area_scheme_boundaries += boundary_transports.bottom_area
        needs_pool_correction(w_bcs.bottom) && (total_area_pool_boundaries += boundary_transports.bottom_area)
    end

    # Check top boundary (w velocity)
    top_transport_and_area = initialize_side_transport(w, w_bcs.top, Val(:top))
    boundary_transports = merge(boundary_transports, top_transport_and_area)
    if needs_transport_correction(w_bcs.top)
        right_scheme_boundaries = (right_scheme_boundaries..., :top)
        total_area_scheme_boundaries += boundary_transports.top_area
        needs_pool_correction(w_bcs.top) && (total_area_pool_boundaries += boundary_transports.top_area)
    end

    boundary_transports = merge(boundary_transports, (; left_scheme_boundaries,
                                                        right_scheme_boundaries,
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

#####
##### Boundary-face shifts
#####

# Shift the boundary-normal component on a boundary plane by `-δ`. Faces on an immersed boundary are
# skipped: they are walls, `mask_immersed_field!` keeps them at zero, and the transport integrals and
# areas above exclude them, so a shift that touched them would both leave a spurious normal velocity
# on the immersed boundary and (for a solver that treats every cell of the underlying grid as active)
# break the solvability condition it was meant to restore.
@kernel function _shift_x_boundary_faces!(u, grid, i, δ)
    j, k = @index(Global, NTuple)
    active = !immersed_peripheral_node(i, j, k, grid, Face(), Center(), Center())
    @inbounds u[i, j, k] -= ifelse(active, δ, zero(δ))
end

@kernel function _shift_y_boundary_faces!(v, grid, j, δ)
    i, k = @index(Global, NTuple)
    active = !immersed_peripheral_node(i, j, k, grid, Center(), Face(), Center())
    @inbounds v[i, j, k] -= ifelse(active, δ, zero(δ))
end

@kernel function _shift_z_boundary_faces!(w, grid, k, δ)
    i, j = @index(Global, NTuple)
    active = !immersed_peripheral_node(i, j, k, grid, Center(), Center(), Face())
    @inbounds w[i, j, k] -= ifelse(active, δ, zero(δ))
end

function shift_x_boundary_faces!(u, i, δ)
    grid = u.grid
    launch!(architecture(grid), grid, KernelParameters(1:grid.Ny, 1:grid.Nz), _shift_x_boundary_faces!, u, grid, i, δ)
    return nothing
end

function shift_y_boundary_faces!(v, j, δ)
    grid = v.grid
    launch!(architecture(grid), grid, KernelParameters(1:grid.Nx, 1:grid.Nz), _shift_y_boundary_faces!, v, grid, j, δ)
    return nothing
end

function shift_z_boundary_faces!(w, k, δ)
    grid = w.grid
    launch!(architecture(grid), grid, KernelParameters(1:grid.Nx, 1:grid.Ny), _shift_z_boundary_faces!, w, grid, k, δ)
    return nothing
end

# Subtract `δ` from the boundary-normal component on the active faces of a boundary plane
shift_boundary_faces!(u, ::Val{:west},   δ) = shift_x_boundary_faces!(u, 1, δ)
shift_boundary_faces!(u, ::Val{:east},   δ) = shift_x_boundary_faces!(u, u.grid.Nx + 1, δ)
shift_boundary_faces!(v, ::Val{:south},  δ) = shift_y_boundary_faces!(v, 1, δ)
shift_boundary_faces!(v, ::Val{:north},  δ) = shift_y_boundary_faces!(v, v.grid.Ny + 1, δ)
shift_boundary_faces!(w, ::Val{:bottom}, δ) = shift_z_boundary_faces!(w, 1, δ)
shift_boundary_faces!(w, ::Val{:top},    δ) = shift_z_boundary_faces!(w, w.grid.Nz + 1, δ)

const LeftSide  = Union{Val{:west}, Val{:south}, Val{:bottom}}
const RightSide = Union{Val{:east}, Val{:north}, Val{:top}}

#####
##### Pool correction
#####

# Boundaries with a `target_transport` are corrected independently (see
# `apply_targeted_boundary_correction!` below); skip them here so the pool
# correction only touches non-targeted boundaries. The net transport is
# positive into the domain, so the left (inflow) faces are lowered and
# the right (outflow) faces are raised by the same amount per unit area.
function correct_left_boundary_transport!(u, bc::NFBC, side::LeftSide, A⁻¹_∮udA)
    has_target_transport(bc.classification.scheme) && return nothing
    return shift_boundary_faces!(u, side, A⁻¹_∮udA)
end

function correct_right_boundary_transport!(u, bc::NFBC, side::RightSide, A⁻¹_∮udA)
    has_target_transport(bc.classification.scheme) && return nothing
    return shift_boundary_faces!(u, side, -A⁻¹_∮udA)
end

# Imposed boundaries without `rebalance=true`, and closed boundaries, keep their values.
correct_left_boundary_transport!(u, ::INFBC, ::LeftSide, A⁻¹_∮udA) = nothing
correct_right_boundary_transport!(u, ::INFBC, ::RightSide, A⁻¹_∮udA) = nothing
correct_left_boundary_transport!(u, bc, side, A⁻¹_∮udA) = nothing
correct_right_boundary_transport!(u, bc, side, A⁻¹_∮udA) = nothing

#####
##### Targeted correction
#####

# Apply a per-boundary correction to reach the prescribed `target_transport`.
# The boundary velocity is adjusted by a uniform shift so that the integral of the
# normal velocity over the boundary matches `target_transport` (positive in the
# positive coordinate direction — e.g., eastward through west/east boundaries).
side_area(bt, ::Val{:west})   = bt.west_area
side_area(bt, ::Val{:east})   = bt.east_area
side_area(bt, ::Val{:south})  = bt.south_area
side_area(bt, ::Val{:north})  = bt.north_area
side_area(bt, ::Val{:bottom}) = bt.bottom_area
side_area(bt, ::Val{:top})    = bt.top_area

function apply_targeted_boundary_correction!(u, bc::NFBC, side, boundary_transport)
    has_target_transport(bc.classification.scheme) || return nothing
    area     = side_area(boundary_transport, side)
    target   = get_target_transport(bc.classification.scheme, u.grid)
    Q_actual = open_boundary_transport(boundary_transport, bc, side)
    return shift_boundary_faces!(u, side, (Q_actual - target) / area)
end

apply_targeted_boundary_correction!(velocity, bc, side, boundary_transport) = nothing

apply_targeted_left_boundary_correction!(velocity, bc, side::LeftSide, boundary_transport) =
    apply_targeted_boundary_correction!(velocity, bc, side, boundary_transport)

apply_targeted_right_boundary_correction!(velocity, bc, side::RightSide, boundary_transport) =
    apply_targeted_boundary_correction!(velocity, bc, side, boundary_transport)

#####
##### Net-zero transport
#####

enforce_net_zero_transport!(velocities, ::Nothing) = nothing

"""
$(TYPEDSIGNATURES)

Correct the boundary values of `velocities` so that the combined net transport through every
boundary that participates in the correction vanishes — the solvability condition of a pressure
Poisson problem with Neumann conditions, such as the incompressible-pressure problem of
`NonhydrostaticModel` without a free surface or the anelastic problem of a mass-flux model.

`velocities` is a `NamedTuple` of three face-normal fields, `(; u, v, w)` or `(; ρu, ρv, ρw)`.
`boundary_transport` is the container returned by [`initialize_boundary_transport`](@ref) for the
same fields, or `nothing` when no boundary requires correction (in which case this is a no-op).

Participating boundaries are the `NormalFlowBoundaryCondition`s with a radiation `scheme` and
the imposed ones built with `NormalFlowBoundaryCondition(value; rebalance=true)`. Boundaries whose
scheme carries a `target_transport` are corrected first by a per-boundary shift so that their net
transport matches the target exactly; the remaining net imbalance is then distributed uniformly
over the pool boundaries (those without a `target_transport`). Faces on an immersed boundary are
left at zero.

The shift is uniform in the given field over each boundary's active area, so for momentum fields
it is uniform in mass flux, and the velocity increment scales as `1/ρ`.

The correction is a function of the boundary data alone, and applying it to already-balanced
data is a fixed point, so it may be repeated freely. It must, however, follow every halo fill that
writes the boundary-normal faces (`fill_halo_regions!` with `fill_normal_flow_bcs=true`, the
default), because such a fill rewrites imposed boundaries from their prescribed values and
reverts an earlier correction. `NonhydrostaticModel` calls it once per stage in
`compute_pressure_correction!`, immediately after its only normal-flow fill.
"""
function enforce_net_zero_transport!(velocities, boundary_transport)
    u, v, w = velocities

    # Step 1: compute current fluxes before any corrections.
    update_open_boundary_transport!(boundary_transport)

    # Step 2: per-boundary corrections for targeted boundaries.
    apply_targeted_boundary_correction!(u, u.boundary_conditions.west,   Val(:west),   boundary_transport)
    apply_targeted_boundary_correction!(v, v.boundary_conditions.south,  Val(:south),  boundary_transport)
    apply_targeted_boundary_correction!(w, w.boundary_conditions.bottom, Val(:bottom), boundary_transport)
    apply_targeted_boundary_correction!(u, u.boundary_conditions.east,   Val(:east),   boundary_transport)
    apply_targeted_boundary_correction!(v, v.boundary_conditions.north,  Val(:north),  boundary_transport)
    apply_targeted_boundary_correction!(w, w.boundary_conditions.top,    Val(:top),    boundary_transport)

    # Step 3: distribute the remaining imbalance over pool boundaries.
    A = boundary_transport.total_area_pool_boundaries
    ∮udA = open_boundary_inflow_transport(boundary_transport, velocities)

    if iszero(A)
        FT = eltype(u)
        net_zero_tolerance = sqrt(eps(FT)) * boundary_transport.total_area_scheme_boundaries
        if abs(∮udA) > net_zero_tolerance
            error("Every open boundary carries a `target_transport`, but the targets do not sum " *
                  "to a net-zero transport across the domain (net inflow = $∮udA). The pressure " *
                  "Poisson problem has no solution in this configuration. Make sure inflow targets " *
                  "balance outflow targets, or leave at least one open boundary without a " *
                  "`target_transport` so the global pool correction can absorb the residual.")
        end
        return nothing
    end

    A⁻¹_∮udA = ∮udA / A

    correct_left_boundary_transport!(u, u.boundary_conditions.west,   Val(:west),   A⁻¹_∮udA)
    correct_left_boundary_transport!(v, v.boundary_conditions.south,  Val(:south),  A⁻¹_∮udA)
    correct_left_boundary_transport!(w, w.boundary_conditions.bottom, Val(:bottom), A⁻¹_∮udA)

    correct_right_boundary_transport!(u, u.boundary_conditions.east,  Val(:east),  A⁻¹_∮udA)
    correct_right_boundary_transport!(v, v.boundary_conditions.north, Val(:north), A⁻¹_∮udA)
    correct_right_boundary_transport!(w, w.boundary_conditions.top,   Val(:top),   A⁻¹_∮udA)

    return nothing
end
