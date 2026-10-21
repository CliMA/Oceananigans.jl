using KernelAbstractions: @kernel, @index
using Oceananigans.BoundaryConditions: BoundaryCondition, Gradient, Mixed, Value
using Oceananigans.Fields: Field
using Oceananigans.Grids: Closed, Open, Sides, inactive_node, topology
using Oceananigans.ImmersedBoundaries: labeled_indices
using Oceananigans.Operators: Vᶜᶜᶜ
using Oceananigans.Utils: KernelParameters, launch!

#####
##### Wall values of `TransportedWallValue`, stored in the tracer at the node beyond each wall: the inactive node beyond an
##### immersed wall, or the halo node beyond a side of the domain whose condition is a flux
#####

# The tendencies and caches of the wall nodes, at the positions of `nodes`. The cache holds cʷ at the start of a split
# Runge-Kutta step, or the previous tendency of Adams-Bashforth and Runge-Kutta 3
struct WallValues{N, T}
       nodes :: N
    tendency :: T
       cache :: T
end


# The storage of the wall values of a tracer's advection, which its directions share, or nothing
wall_values(advection) = nothing
wall_values(scheme::Union{WENO, UpwindBiased, Centered}) = wall_values(scheme.extrapolation)
wall_values(extrapolation::TransportedWallValue) = extrapolation.wall_values
wall_values(advection::FluxFormAdvection) = something(wall_values(advection.x), wall_values(advection.y), wall_values(advection.z), Some(nothing))

@inline function stores_wall_value(i, j, k, grid, d, s)
    di, dj, dk = unit_vector(d)
    immersed = inactive_node(i, j, k, grid, Center(), Center(), Center())
    neighbor_active = !inactive_node(i - s * di, j - s * dj, k - s * dk, grid, Center(), Center(), Center())
    return immersed & neighbor_active
end

@inline stores_wall_value_along(i, j, k, grid, ::Type{Flat}, d) = false
@inline stores_wall_value_along(i, j, k, grid, T, d) = stores_wall_value(i, j, k, grid, d, -1) | stores_wall_value(i, j, k, grid, d, +1)

# `sides` = (west, east, south, north, bottom, top): whether the halo node beyond each side can store a wall value
@kernel function _label_wall_nodes!(labels, grid, sides)
    i, j, k = @index(Global, NTuple)
    Nx, Ny, Nz = size(grid)
    west, east, south, north, bottom, top = sides
    beyond_transporting_side = (west | (i ≥ 1)) & (east | (i ≤ Nx)) & (south | (j ≥ 1)) & (north | (j ≤ Ny)) & (bottom | (k ≥ 1)) & (top | (k ≤ Nz))
    @inbounds labels[i, j, k] = beyond_transporting_side & (stores_wall_value_along(i, j, k, grid, topology(grid, 1), Val(1)) |
                                                            stores_wall_value_along(i, j, k, grid, topology(grid, 2), Val(2)) |
                                                            stores_wall_value_along(i, j, k, grid, topology(grid, 3), Val(3)))
end

# A Closed side whose condition is a flux transports its wall value; a Value, Gradient or Mixed side fills it in the halo
transports_wall_value(side, bc) = false
transports_wall_value(::Type{Closed}, bc) = true
transports_wall_value(::Type{Closed}, ::BoundaryCondition{<:Union{Value, Gradient, Mixed}}) = false

transporting_sides(T, bcᴸ, bcᴿ) = (false, false)
transporting_sides(::Type{Sides{L, R}}, bcᴸ, bcᴿ) where {L, R} = (transports_wall_value(L, bcᴸ), transports_wall_value(R, bcᴿ))

# The interior and the first halo layer, which holds the wall nodes of the sides of the domain
wall_node_range(::Type{Flat}, N) = 1:1
wall_node_range(T, N) = 0:N+1

function WallValues(grid, bcs)
    sides = (transporting_sides(topology(grid, 1), bcs.west,   bcs.east)...,
             transporting_sides(topology(grid, 2), bcs.south,  bcs.north)...,
             transporting_sides(topology(grid, 3), bcs.bottom, bcs.top)...)

    labels = Field{Center, Center, Center}(grid, Int8)
    ranges = map(wall_node_range, topology(grid), size(grid))
    launch!(architecture(grid), grid, KernelParameters(ranges...), _label_wall_nodes!, labels, grid, sides)
    nodes, _ = labeled_indices(grid, labels.data, (1,))

    isempty(nodes) && return nothing
    wall_vector() = on_architecture(architecture(grid), zeros(eltype(grid), length(nodes)))
    return WallValues(nodes, wall_vector(), wall_vector())
end

"""
    materialize_wall_values(advection, grid, boundary_conditions)

Return `advection` with the storage of its wall values on `grid`, shared by the reconstructions whose `extrapolation` is
`TransportedWallValue()`; without storage, `TransportedWallValue` extrapolates as `HierarchicalExtrapolation`. The walls of
the domain with a Value, Gradient or Mixed condition in `boundary_conditions` transport nothing: their condition fills the
wall value in the halo.
"""
materialize_wall_values(advection, grid, boundary_conditions) =
    with_wall_values(advection, TransportedWallValue(WallValues(grid, boundary_conditions)))


with_wall_values(advection, stored) = advection
with_wall_values(weno::WENO{N, FT, WCT, <:Any, <:Any, <:TransportedWallValue}, stored) where {N, FT, WCT} = WENO{N, FT, WCT}(weno.bounds, stored, weno.advecting_velocity_scheme, weno.time_discretization)
with_wall_values(scheme::UpwindBiased{N, FT, <:Any, <:TransportedWallValue},   stored) where {N, FT}      = UpwindBiased{N, FT}(stored, scheme.advecting_velocity_scheme, scheme.time_discretization)
with_wall_values(scheme::Centered{N, FT, <:Any, <:TransportedWallValue},       stored) where {N, FT}      = Centered{N, FT}(stored, scheme.time_discretization)

with_wall_values(advection::FluxFormAdvection, stored) = FluxFormAdvection(with_wall_values(advection.x, stored), with_wall_values(advection.y, stored), with_wall_values(advection.z, stored))

# Two-node reconstructions take the node with data and read no wall value
with_wall_values(scheme::Union{UpwindBiased{1}, Centered{1}}, stored) = scheme

"""
    materialize_wall_values(advection::NamedTuple, grid, boundary_conditions, names)

Return `advection` with the storage of the wall values of the tracers `names`, whose `boundary_conditions` are regularized.
"""
function materialize_wall_values(advection::NamedTuple, grid, boundary_conditions, names)
    stored = map(name -> materialize_wall_values(advection[name], grid, boundary_conditions[name]), names)
    return merge(advection, NamedTuple{names}(stored))
end

Adapt.adapt_structure(to, w::WallValues) = WallValues(Adapt.adapt(to, w.nodes), Adapt.adapt(to, w.tendency), Adapt.adapt(to, w.cache))
Architectures.on_architecture(to, w::WallValues) = WallValues(on_architecture(to, w.nodes), on_architecture(to, w.tendency), on_architecture(to, w.cache))

Adapt.adapt_structure(to, extrapolation::TransportedWallValue) = TransportedWallValue(Adapt.adapt(to, extrapolation.wall_values))
Architectures.on_architecture(to, extrapolation::TransportedWallValue) = TransportedWallValue(on_architecture(to, extrapolation.wall_values))

#####
##### The wall values a reconstruction along a direction reads from its stencil, at the node beyond the last node with data
#####

# Beyond the exterior cell of an Open side there is no wall value, so a direction with an Open side extrapolates hierarchically
@inline extrapolation_along(extrapolation, grid, d) = HierarchicalExtrapolation()
@inline extrapolation_along(extrapolation::TransportedWallValue{<:WallValues}, grid, ::Val{d}) where d =
    ifelse(open_direction(topology(grid, d)), HierarchicalExtrapolation(), extrapolation)

@inline open_direction(T) = false
@inline open_direction(::Type{Sides{L, R}}) where {L, R} = (L === Open) | (R === Open)

#####
##### Tendency
#####

@inline face_transport(i, j, k, grid, U, ::Val{1}) = Ax_qᶠᶜᶜ(i, j, k, grid, U.u)
@inline face_transport(i, j, k, grid, U, ::Val{2}) = Ay_qᶜᶠᶜ(i, j, k, grid, U.v)
@inline face_transport(i, j, k, grid, U, ::Val{3}) = Az_qᶜᶜᶠ(i, j, k, grid, U.w)

# The node 1, 1, 1 keeps the discarded evaluations in bounds where there is no wall cell
@inline function adjacent_fluid_cell(i, j, k, grid, d, s)
    di, dj, dk = unit_vector(d)
    wall = stores_wall_value(i, j, k, grid, d, s)
    return ifelse(wall, i - s * di, 1), ifelse(wall, j - s * dj, 1), ifelse(wall, k - s * dk, 1), wall
end

@inline function extrapolated_wall_value(i, j, k, grid, scheme::AbstractAdvectionScheme{N}, c, d, s) where N
    e = unit_vector(d)
    di, dj, dk = e
    f = (1 - s) ÷ 2
    iᶠ, jᶠ, kᶠ = i + f * di, j + f * dj, k + f * dk
    WCT = weight_computation(scheme)
    S  = index_stencil(iᶠ, jᶠ, kᶠ, grid, scheme, e, c)
    ΔS = node_differences(S)
    nᴸ = boundary_distance(iᶠ - di, jᶠ - dj, kᶠ - dk, grid, d, Center(), Center(), Center(), -1, Val(N))
    nᴿ = boundary_distance(iᶠ,      jᶠ,      kᶠ,      grid, d, Center(), Center(), Center(), +1, Val(N))
    n  = ifelse(s == 1, nᴿ, nᴸ)
    nᵒ = ifelse(s == 1, nᴸ, nᴿ)
    σ, _ = ghost_extrapolation(WCT, S, ΔS, n, nᵒ, s, HierarchicalExtrapolation(), Val(N))
    # a cell walled on both sides along d has no line beyond it
    opposite = inactive_node(i - s * di, j - s * dj, k - s * dk, grid, Center(), Center(), Center())
    return @inbounds c[i, j, k] + ifelse(opposite, zero(σ), s * σ / 2)
end

@inline function van_albada_agreement(WCT, Δ₁, Δ₂)
    φ = newton_div(WCT, 2 * Δ₁ * Δ₂, Δ₁^2 + Δ₂^2)
    return ifelse(Δ₁ * Δ₂ > 0, φ, zero(Δ₁))
end

# The upwind wall value, or the fluid where the wall ends, plus half its van Albada limited slope along the wall; the tracer
# holds both the fluid and the wall values
@inline function tangential_face_value(i, j, k, grid, WCT, c, d, s, t, q)
    ti, tj, tk = unit_vector(t)
    sᵗ = ifelse(q > 0, 1, -1)
    oᵘ = (1 + sᵗ) ÷ 2
    iᵁ, jᵁ, kᵁ = i - oᵘ * ti, j - oᵘ * tj, k - oᵘ * tk
    iᴰ, jᴰ, kᴰ = iᵁ + sᵗ * ti, jᵁ + sᵗ * tj, kᵁ + sᵗ * tk
    iᵁᵁ, jᵁᵁ, kᵁᵁ = iᵁ - sᵗ * ti, jᵁ - sᵗ * tj, kᵁ - sᵗ * tk

    cᵁ  = @inbounds c[iᵁ, jᵁ, kᵁ]
    cᴰ  = @inbounds c[iᴰ, jᴰ, kᴰ]
    cᵁᵁ = @inbounds c[iᵁᵁ, jᵁᵁ, kᵁᵁ]

    along = stores_wall_value(iᵁ,  jᵁ,  kᵁ,  grid, d, s) &
            stores_wall_value(iᴰ,  jᴰ,  kᴰ,  grid, d, s) &
            stores_wall_value(iᵁᵁ, jᵁᵁ, kᵁᵁ, grid, d, s)

    Δ⁻ = ifelse(along, cᵁ - cᵁᵁ, zero(cᵁ))
    Δ⁺ = cᴰ - cᵁ

    return cᵁ + van_albada_agreement(WCT, Δ⁻, Δ⁺) * Δ⁻ / 2
end

@inline tangential_tendency(i, j, k, iᴿ, jᴿ, kᴿ, grid, ::Type{Flat}, WCT, U, c, d, s, t) = zero(grid)

# A face whose cell across has no data, the end of a wall at a side of the domain, carries no transport
@inline function tangential_tendency(i, j, k, iᴿ, jᴿ, kᴿ, grid, T, WCT, U, c, d, s, t)
    ti, tj, tk = unit_vector(t)
    q⁻ = face_transport(iᴿ,      jᴿ,      kᴿ,      grid, U, t)
    q⁺ = face_transport(iᴿ + ti, jᴿ + tj, kᴿ + tk, grid, U, t)
    ĉ⁻ = tangential_face_value(i,      j,      k,      grid, WCT, c, d, s, t, q⁻)
    ĉ⁺ = tangential_face_value(i + ti, j + tj, k + tk, grid, WCT, c, d, s, t, q⁺)
    d⁻ = node_has_data(iᴿ - ti, jᴿ - tj, kᴿ - tk, grid, t, Center(), Center(), Center())
    d⁺ = node_has_data(iᴿ + ti, jᴿ + tj, kᴿ + tk, grid, t, Center(), Center(), Center())
    G⁻ = @inbounds ifelse(d⁻, q⁻ * (ĉ⁻ - c[i, j, k]), zero(q⁻))
    G⁺ = @inbounds ifelse(d⁺, q⁺ * (ĉ⁺ - c[i, j, k]), zero(q⁺))
    return G⁻ - G⁺
end

@inline tangential_directions(::Val{d}) where d = (Val(ifelse(d == 1, 2, 1)), Val(ifelse(d == 3, 2, 3)))

@inline advection_along(advection, ::Val{1}) = x_advection(advection)
@inline advection_along(advection, ::Val{2}) = y_advection(advection)
@inline advection_along(advection, ::Val{3}) = z_advection(advection)

# V ∂ₜcʷ = qⁿ (ĉʷ - cʷ) + Σₜ [q⁻ (ĉ⁻ - cʷ) - q⁺ (ĉ⁺ - cʷ)], with qⁿ the inflow into R through its face opposite the wall
@inline function single_wall_value_tendency(i, j, k, grid, advection, U, c, d, s)
    iᴿ, jᴿ, kᴿ, wall = adjacent_fluid_cell(i, j, k, grid, d, s)
    scheme = advection_along(advection, d)
    WCT = weight_computation(scheme)

    di, dj, dk = unit_vector(d)
    f  = (1 - s) ÷ 2
    qⁿ = max(s * face_transport(iᴿ + f * di, jᴿ + f * dj, kᴿ + f * dk, grid, U, d), 0)
    Gʷ = @inbounds qⁿ * (extrapolated_wall_value(iᴿ, jᴿ, kᴿ, grid, scheme, c, d, s) - c[i, j, k])

    t₁, t₂ = tangential_directions(d)
    Gʷ += tangential_tendency(i, j, k, iᴿ, jᴿ, kᴿ, grid, topology(grid, t₁), WCT, U, c, d, s, t₁)
    Gʷ += tangential_tendency(i, j, k, iᴿ, jᴿ, kᴿ, grid, topology(grid, t₂), WCT, U, c, d, s, t₂)

    return ifelse(wall, Gʷ / Vᶜᶜᶜ(iᴿ, jᴿ, kᴿ, grid), zero(Gʷ)), wall
end

@inline directional_wall_value_tendency(i, j, k, grid, ::Type{Flat}, scheme, U, c, d) = zero(grid), zero(grid)

@inline function directional_wall_value_tendency(i, j, k, grid, T, scheme, U, c, d)
    G⁻, w⁻ = single_wall_value_tendency(i, j, k, grid, scheme, U, c, d, -1)
    G⁺, w⁺ = single_wall_value_tendency(i, j, k, grid, scheme, U, c, d, +1)
    return G⁻ + G⁺, w⁻ + w⁺
end

# A node that stores several walls follows the average of their tendencies; cʷ is a point value at the wall and steps as cʷ
@inline function wall_value_tendency(i, j, k, grid, scheme, U, c)
    Gˣ, wˣ = directional_wall_value_tendency(i, j, k, grid, topology(grid, 1), scheme, U, c, Val(1))
    Gʸ, wʸ = directional_wall_value_tendency(i, j, k, grid, topology(grid, 2), scheme, U, c, Val(2))
    Gᶻ, wᶻ = directional_wall_value_tendency(i, j, k, grid, topology(grid, 3), scheme, U, c, Val(3))
    return (Gˣ + Gʸ + Gᶻ) / max(wˣ + wʸ + wᶻ, one(grid))
end

@inline initial_wall_value(i, j, k, grid, ::Type{Flat}, scheme, c, d) = zero(grid), zero(grid)

@inline function initial_wall_value(i, j, k, grid, T, scheme, c, d)
    total = zero(grid)
    wall_count = zero(grid)
    for s in (-1, +1)
        iᴿ, jᴿ, kᴿ, wall = adjacent_fluid_cell(i, j, k, grid, d, s)
        total += ifelse(wall, extrapolated_wall_value(iᴿ, jᴿ, kᴿ, grid, advection_along(scheme, d), c, d, s), zero(grid))
        wall_count += wall
    end
    return total, wall_count
end

@inline function initial_wall_value(i, j, k, grid, scheme, c)
    cˣ, wˣ = initial_wall_value(i, j, k, grid, topology(grid, 1), scheme, c, Val(1))
    cʸ, wʸ = initial_wall_value(i, j, k, grid, topology(grid, 2), scheme, c, Val(2))
    cᶻ, wᶻ = initial_wall_value(i, j, k, grid, topology(grid, 3), scheme, c, Val(3))
    return (cˣ + cʸ + cᶻ) / max(wˣ + wʸ + wᶻ, one(grid))
end

#####
##### Kernels over the wall nodes
#####

@kernel function _compute_wall_value_tendency!(walls, grid, scheme, U, c)
    n, = @index(Global, NTuple)
    i, j, k = @inbounds walls.nodes[n]
    @inbounds walls.tendency[n] = wall_value_tendency(i, j, k, grid, scheme, U, c)
end

@kernel function _initialize_wall_values!(walls, grid, scheme, c)
    n, = @index(Global, NTuple)
    i, j, k = @inbounds walls.nodes[n]
    @inbounds c[i, j, k] = initial_wall_value(i, j, k, grid, scheme, c)
end

@kernel function _cache_wall_values!(walls, grid, c)
    n, = @index(Global, NTuple)
    i, j, k = @inbounds walls.nodes[n]
    @inbounds walls.cache[n] = c[i, j, k]
end

@kernel function _rk_substep_wall_values!(walls, grid, c, Δt)
    n, = @index(Global, NTuple)
    i, j, k = @inbounds walls.nodes[n]
    @inbounds c[i, j, k] = walls.cache[n] + Δt * walls.tendency[n]
end

@kernel function _ab2_step_wall_values!(walls, grid, c, Δt, χ)
    n, = @index(Global, NTuple)
    i, j, k = @inbounds walls.nodes[n]

    FT = eltype(χ)
    α = 3 * one(FT) / 2 + χ
    β = 1 * one(FT) / 2 + χ
    not_euler = χ != convert(FT, -0.5) # use to prevent corruption by leftover NaNs in G⁻

    @inbounds begin
        Gⁿ = walls.tendency[n]
        c[i, j, k] += Δt * (α * Gⁿ - β * walls.cache[n] * not_euler)
        walls.cache[n] = Gⁿ
    end
end

@kernel function _rk3_substep_wall_values!(walls, grid, c, Δt, γⁿ, ζⁿ)
    n, = @index(Global, NTuple)
    i, j, k = @inbounds walls.nodes[n]
    @inbounds begin
        Gⁿ = walls.tendency[n]
        c[i, j, k] += Δt * (γⁿ * Gⁿ + ζⁿ * walls.cache[n])
        walls.cache[n] = Gⁿ
    end
end

@kernel function _rk3_substep_wall_values!(walls, grid, c, Δt, γ¹, ::Nothing)
    n, = @index(Global, NTuple)
    i, j, k = @inbounds walls.nodes[n]
    @inbounds begin
        Gⁿ = walls.tendency[n]
        c[i, j, k] += Δt * γ¹ * Gⁿ
        walls.cache[n] = Gⁿ
    end
end

# A map over the positions in `nodes` launches the same kernel for any number of nodes, whose count is known only at runtime
launch_over_wall_values!(::Nothing, grid, kernel!::K, args::Vararg{Any, N}) where {K, N} = nothing

function launch_over_wall_values!(walls::WallValues, grid, kernel!::K, args::Vararg{Any, N}) where {K, N}
    launch!(architecture(grid), grid, Base.OneTo(length(walls.nodes)), kernel!, walls, grid, args...)
    return nothing
end

#####
##### Interface for the models: `c` is the tracer, which holds its wall values
#####

compute_wall_value_tendency!(advection, U, c) = launch_over_wall_values!(wall_values(advection), c.grid, _compute_wall_value_tendency!, advection, U, c)
initialize_wall_values!(advection, c) = launch_over_wall_values!(wall_values(advection), c.grid, _initialize_wall_values!, advection, c)
cache_wall_values!(advection, c) = launch_over_wall_values!(wall_values(advection), c.grid, _cache_wall_values!, c)
rk_substep_wall_values!(advection, c, Δt) = launch_over_wall_values!(wall_values(advection), c.grid, _rk_substep_wall_values!, c, Δt)
ab2_step_wall_values!(advection, c, Δt, χ) = launch_over_wall_values!(wall_values(advection), c.grid, _ab2_step_wall_values!, c, Δt, χ)
rk3_substep_wall_values!(advection, c, Δt, γⁿ, ζⁿ) = launch_over_wall_values!(wall_values(advection), c.grid, _rk3_substep_wall_values!, c, Δt, γⁿ, ζⁿ)

# The tracers of a model, and the wall values that the advection of each stores.
# map over the values: map over a NamedTuple does not specialize on the function, and boxes the advection schemes
compute_wall_value_tendencies!(advection::NamedTuple, U, tracers) = (map((a, c) -> compute_wall_value_tendency!(a, U, c), values(NamedTuple{keys(tracers)}(advection)), values(tracers)); nothing)
initialize_wall_values!(advection::NamedTuple, tracers) = (map(initialize_wall_values!, values(NamedTuple{keys(tracers)}(advection)), values(tracers)); nothing)
cache_wall_values!(advection::NamedTuple, tracers) = (map(cache_wall_values!, values(NamedTuple{keys(tracers)}(advection)), values(tracers)); nothing)
rk_substep_wall_values!(advection::NamedTuple, tracers, Δt) = (map((a, c) -> rk_substep_wall_values!(a, c, Δt), values(NamedTuple{keys(tracers)}(advection)), values(tracers)); nothing)
ab2_step_wall_values!(advection::NamedTuple, tracers, Δt, χ) = (map((a, c) -> ab2_step_wall_values!(a, c, Δt, χ), values(NamedTuple{keys(tracers)}(advection)), values(tracers)); nothing)
rk3_substep_wall_values!(advection::NamedTuple, tracers, Δt, γⁿ, ζⁿ) = (map((a, c) -> rk3_substep_wall_values!(a, c, Δt, γⁿ, ζⁿ), values(NamedTuple{keys(tracers)}(advection)), values(tracers)); nothing)

#####
##### Checkpointing: the wall values are part of the tracer; the caches are saved here
#####

function Oceananigans.prognostic_state(advection::AbstractAdvectionScheme)
    walls = wall_values(advection)
    return isnothing(walls) ? nothing : (; cache = on_architecture(CPU(), walls.cache))
end

Oceananigans.restore_prognostic_state!(advection::AbstractAdvectionScheme, ::Nothing) = advection

function Oceananigans.restore_prognostic_state!(advection::AbstractAdvectionScheme, from)
    copyto!(wall_values(advection).cache, from.cache)
    return advection
end
