using KernelAbstractions: @kernel, @index
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: Field, CenterField
using Oceananigans.Grids: Bounded, LeftConnected, RightConnected, inactive_node, topology
using Oceananigans.ImmersedBoundaries: MutableGridOfSomeKind, labeled_indices
using Oceananigans.Operators: Vᶜᶜᶜ, σⁿ, ∂t_σ
using Oceananigans.Utils: launch!

#####
##### Stored wall values of `TransportedWallValue`
#####

# Values at the inactive nodes beyond the immersed walls; tendencies, previous tendencies and caches at the positions of `nodes`
struct ImmersedWallValues{N, V, T}
                nodes :: N
               values :: V
             tendency :: T
    previous_tendency :: T
                cache :: T
end

# Fields reduced along d, indexed by the halo nodes beyond the wall of the domain normal to d on the side s
struct DomainWallValues{d, s, V}
               values :: V
             tendency :: V
    previous_tendency :: V
                cache :: V
end

DomainWallValues{d, s}(values::V, tendency::V, previous_tendency::V, cache::V) where {d, s, V} =
    DomainWallValues{d, s, V}(values, tendency, previous_tendency, cache)

const StoredWallValueWENO = WENO{<:Any, <:Any, <:Any, <:Any, <:Any, <:GhostCells{<:Any, <:TransportedWallValue{<:ImmersedWallValues}}}

transported_wall_value(advection::StoredWallValueWENO) = advection.buffer_scheme.extrapolation

@inline function stores_wall_value(i, j, k, grid, d, s)
    di, dj, dk = unit_vector(d)
    immersed = inactive_node(i, j, k, grid, Center(), Center(), Center())
    neighbor_active = !inactive_node(i - s * di, j - s * dj, k - s * dk, grid, Center(), Center(), Center())
    return immersed & neighbor_active
end

@inline stores_wall_value_along(i, j, k, grid, ::Type{Flat}, d) = false
@inline stores_wall_value_along(i, j, k, grid, T, d) = stores_wall_value(i, j, k, grid, d, -1) | stores_wall_value(i, j, k, grid, d, +1)

@kernel function _label_wall_nodes!(labels, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds labels[i, j, k] = stores_wall_value_along(i, j, k, grid, topology(grid, 1), Val(1)) |
                                stores_wall_value_along(i, j, k, grid, topology(grid, 2), Val(2)) |
                                stores_wall_value_along(i, j, k, grid, topology(grid, 3), Val(3))
end

function ImmersedWallValues(grid)
    labels = Field{Center, Center, Center}(grid, Int8)
    launch!(architecture(grid), grid, size(grid), _label_wall_nodes!, labels, grid)  # size(grid), since :xyz skips the inactive nodes on grids with an active cells map
    nodes, _ = labeled_indices(grid, labels.data, (1,))
    wall_vector() = on_architecture(architecture(grid), zeros(eltype(grid), length(nodes)))
    return ImmersedWallValues(nodes, CenterField(grid), wall_vector(), wall_vector(), wall_vector())
end

wall_field(grid, ::Val{1}) = Field{Nothing, Center, Center}(grid)
wall_field(grid, ::Val{2}) = Field{Center, Nothing, Center}(grid)
wall_field(grid, ::Val{3}) = Field{Center, Center, Nothing}(grid)

DomainWallValues(grid, ::Val{d}, s) where d = DomainWallValues{d, s}(wall_field(grid, Val(d)), wall_field(grid, Val(d)),
                                                                     wall_field(grid, Val(d)), wall_field(grid, Val(d)))

domain_wall_values(grid, d, topology)               = (left = nothing,                       right = nothing)
domain_wall_values(grid, d, ::Type{Bounded})        = (left = DomainWallValues(grid, d, -1), right = DomainWallValues(grid, d, +1))
domain_wall_values(grid, d, ::Type{RightConnected}) = (left = DomainWallValues(grid, d, -1), right = nothing)
domain_wall_values(grid, d, ::Type{LeftConnected})  = (left = nothing,                       right = DomainWallValues(grid, d, +1))

"""
    materialize_wall_values(advection, grid)

Return `advection` with the storage of its wall values on `grid`, when its boundary scheme is
`GhostCells(extrapolation = TransportedWallValue())`; without storage, `TransportedWallValue` extrapolates as
`HierarchicalExtrapolation`.
"""
materialize_wall_values(advection, grid) = advection

function materialize_wall_values(weno::WENO{N, FT, WCT, <:Any, <:Any, <:GhostCells{<:Any, <:TransportedWallValue}}, grid) where {N, FT, WCT}
    domain_walls  = (x = domain_wall_values(grid, Val(1), topology(grid, 1)),
                     y = domain_wall_values(grid, Val(2), topology(grid, 2)),
                     z = domain_wall_values(grid, Val(3), topology(grid, 3)))
    extrapolation = TransportedWallValue(ImmersedWallValues(grid), domain_walls)
    buffer_scheme = GhostCells(weno.buffer_scheme.scheme, extrapolation)
    return WENO{N, FT, WCT}(weno.bounds, buffer_scheme, weno.advecting_velocity_scheme, weno.time_discretization)
end

"""
    materialize_wall_values(advection::NamedTuple, grid, names)

Return `advection` with the storage of the wall values of the tracers `names`.
"""
function materialize_wall_values(advection::NamedTuple, grid, names)
    stored = map(name -> materialize_wall_values(advection[name], grid), names)
    return merge(advection, NamedTuple{names}(stored))
end

Adapt.adapt_structure(to, w::ImmersedWallValues) =
    ImmersedWallValues(Adapt.adapt(to, w.nodes), Adapt.adapt(to, w.values), Adapt.adapt(to, w.tendency),
                       Adapt.adapt(to, w.previous_tendency), Adapt.adapt(to, w.cache))

Adapt.adapt_structure(to, w::DomainWallValues{d, s}) where {d, s} =
    DomainWallValues{d, s}(Adapt.adapt(to, w.values), Adapt.adapt(to, w.tendency), Adapt.adapt(to, w.previous_tendency), Adapt.adapt(to, w.cache))

Architectures.on_architecture(to, w::ImmersedWallValues) =
    ImmersedWallValues(on_architecture(to, w.nodes), on_architecture(to, w.values), on_architecture(to, w.tendency),
                       on_architecture(to, w.previous_tendency), on_architecture(to, w.cache))

Architectures.on_architecture(to, w::DomainWallValues{d, s}) where {d, s} =
    DomainWallValues{d, s}(on_architecture(to, w.values), on_architecture(to, w.tendency),
                           on_architecture(to, w.previous_tendency), on_architecture(to, w.cache))

# The reconstruction reads only the wall values
adapt_wall_values(to, ::Nothing) = nothing
adapt_wall_values(to, w::Union{ImmersedWallValues, DomainWallValues}) = Adapt.adapt(to, w.values)
adapt_wall_values(to, walls::NamedTuple) = map(w -> adapt_wall_values(to, w), walls)

Adapt.adapt_structure(to, extrapolation::TransportedWallValue) = TransportedWallValue(adapt_wall_values(to, extrapolation.immersed_walls), adapt_wall_values(to, extrapolation.domain_walls))
Architectures.on_architecture(to, extrapolation::TransportedWallValue) = TransportedWallValue(on_architecture(to, extrapolation.immersed_walls), on_architecture(to, extrapolation.domain_walls))

#####
##### Reading the wall values
#####

@inline wall_index(i, j, k, ::Val{1}) = (1, j, k)
@inline wall_index(i, j, k, ::Val{2}) = (i, 1, k)
@inline wall_index(i, j, k, ::Val{3}) = (i, j, 1)

@inline Base.getindex(w::ImmersedWallValues, i, j, k)     = @inbounds w.values[i, j, k]
@inline Base.setindex!(w::ImmersedWallValues, v, i, j, k) = @inbounds w.values[i, j, k] = v

# wall_index maps the halo nodes beyond the wall, and the nodes of the reduced fields, to the nodes of the reduced fields
@inline Base.getindex(w::DomainWallValues{d}, i, j, k)     where d = @inbounds w.values[wall_index(i, j, k, Val(d))...]
@inline Base.setindex!(w::DomainWallValues{d}, v, i, j, k) where d = @inbounds w.values[wall_index(i, j, k, Val(d))...] = v

@inline domain_wall(::Nothing, i, j, k, grid, d) = (false, zero(grid))
@inline domain_wall(cʷ,        i, j, k, grid, d) = (true, @inbounds(cʷ[wall_index(i, j, k, d)...]))

@inline stored_wall_value(extrapolation, i, j, k, grid, d, nᴸ, nᴿ, bias, ::Val{N}) where N = nothing

# pᵈ is the index along d of the upwind peripheral node, which stores the wall value of an immersed wall
@inline function stored_wall_value(extrapolation::TransportedWallValue{<:Any, <:NamedTuple}, i, j, k, grid, ::Val{d}, nᴸ, nᴿ, bias, ::Val{N}) where {d, N}
    left = bias === LeftBias
    m  = ifelse(left, nᴸ, nᴿ) + 1
    iᵈ = (i, j, k)[d]
    pᵈ = ifelse(left, iᵈ - 1 - m, iᵈ + m)
    cᵖ = @inbounds extrapolation.immersed_walls[Base.setindex((i, j, k), pᵈ, d)...]

    cᴸ, cᴿ     = extrapolation.domain_walls[d]
    wallᴸ, cʷᴸ = domain_wall(cᴸ, i, j, k, grid, Val(d))
    wallᴿ, cʷᴿ = domain_wall(cᴿ, i, j, k, grid, Val(d))
    beyond     = ifelse(left, wallᴸ & (pᵈ < 1), wallᴿ & (pᵈ > size(grid, d)))

    return ifelse(beyond, ifelse(left, cʷᴸ, cʷᴿ), cᵖ)
end

@inline domain_wall_node(a, b, grid, ::Val{1}, s) = (ifelse(s == -1, 0, size(grid, 1) + 1), a, b)
@inline domain_wall_node(a, b, grid, ::Val{2}, s) = (a, ifelse(s == -1, 0, size(grid, 2) + 1), b)
@inline domain_wall_node(a, b, grid, ::Val{3}, s) = (a, b, ifelse(s == -1, 0, size(grid, 3) + 1))

#####
##### Tendency
#####

@inline face_transport(i, j, k, grid, U, ::Val{1}) = @inbounds U.u[i, j, k] * Axᶠᶜᶜ(i, j, k, grid)
@inline face_transport(i, j, k, grid, U, ::Val{2}) = @inbounds U.v[i, j, k] * Ayᶜᶠᶜ(i, j, k, grid)
@inline face_transport(i, j, k, grid, U, ::Val{3}) = @inbounds U.w[i, j, k] * Azᶜᶜᶠ(i, j, k, grid)

# The node 1, 1, 1 keeps the discarded evaluations in bounds where there is no wall cell
@inline function adjacent_fluid_cell(i, j, k, grid, d, s)
    di, dj, dk = unit_vector(d)
    wall = stores_wall_value(i, j, k, grid, d, s)
    return ifelse(wall, i - s * di, 1), ifelse(wall, j - s * dj, 1), ifelse(wall, k - s * dk, 1), wall
end

@inline function normal_face_value(i, j, k, grid, weno::WENO{N, FT, WCT}, c, d, s) where {N, FT, WCT}
    e = unit_vector(d)
    di, dj, dk = e
    f = (1 - s) ÷ 2
    iᶠ, jᶠ, kᶠ = i + f * di, j + f * dj, k + f * dk
    S  = upwind_stencil(iᶠ, jᶠ, kᶠ, grid, weno, LeftBias, e, c)
    ΔS = node_differences(S)
    nᴸ = boundary_distance(iᶠ - di, jᶠ - dj, kᶠ - dk, grid, Center(), Center(), Center(), d, -1, Val(N))
    nᴿ = boundary_distance(iᶠ,      jᶠ,      kᶠ,      grid, Center(), Center(), Center(), d, +1, Val(N))
    n  = ifelse(s == 1, nᴿ, nᴸ)
    nᵒ = ifelse(s == 1, nᴸ, nᴿ)
    σ, _ = ghost_extrapolation(WCT, S, ΔS, n, nᵒ, s, nothing, Val(N))
    # a cell walled on both sides along d has no line beyond it
    opposite = inactive_node(i - s * di, j - s * dj, k - s * dk, grid, Center(), Center(), Center())
    return @inbounds c[i, j, k] + ifelse(opposite, zero(σ), s * σ / 2)
end

@inline function van_albada_agreement(WCT, Δ₁, Δ₂)
    φ = newton_div(WCT, 2 * Δ₁ * Δ₂, Δ₁^2 + Δ₂^2)
    return ifelse(Δ₁ * Δ₂ > 0, φ, zero(Δ₁))
end

# The upwind wall value, or the fluid where the wall ends, plus half its van Albada limited slope along the wall
@inline function tangential_face_value(i, j, k, grid, WCT, c, cʷ, d, s, t, q)
    ti, tj, tk = unit_vector(t)
    σ = ifelse(q > 0, 1, -1)
    u = (1 + σ) ÷ 2
    iᵁ, jᵁ, kᵁ = i - u * ti, j - u * tj, k - u * tk
    iᴰ, jᴰ, kᴰ = iᵁ + σ * ti, jᵁ + σ * tj, kᵁ + σ * tk
    iᵁᵁ, jᵁᵁ, kᵁᵁ = iᵁ - σ * ti, jᵁ - σ * tj, kᵁ - σ * tk

    cᵁ  = @inbounds ifelse(inactive_node(iᵁ, jᵁ, kᵁ, grid, Center(), Center(), Center()), cʷ[iᵁ, jᵁ, kᵁ], c[iᵁ, jᵁ, kᵁ])
    cᴰ  = @inbounds cʷ[iᴰ, jᴰ, kᴰ]
    cᵁᵁ = @inbounds cʷ[iᵁᵁ, jᵁᵁ, kᵁᵁ]

    along = stores_wall_value(iᵁ,  jᵁ,  kᵁ,  grid, d, s) &
            stores_wall_value(iᴰ,  jᴰ,  kᴰ,  grid, d, s) &
            stores_wall_value(iᵁᵁ, jᵁᵁ, kᵁᵁ, grid, d, s)

    Δ⁻ = ifelse(along, cᵁ - cᵁᵁ, zero(cᵁ))
    Δ⁺ = cᴰ - cᵁ

    return cᵁ + van_albada_agreement(WCT, Δ⁻, Δ⁺) * Δ⁻ / 2
end

@inline tangential_tendency(i, j, k, iᴿ, jᴿ, kᴿ, grid, ::Type{Flat}, WCT, U, c, cʷ, d, s, t) = zero(grid)

@inline function tangential_tendency(i, j, k, iᴿ, jᴿ, kᴿ, grid, T, WCT, U, c, cʷ, d, s, t)
    ti, tj, tk = unit_vector(t)
    q⁻ = face_transport(iᴿ,      jᴿ,      kᴿ,      grid, U, t)
    q⁺ = face_transport(iᴿ + ti, jᴿ + tj, kᴿ + tk, grid, U, t)
    ĉ⁻ = tangential_face_value(i,      j,      k,      grid, WCT, c, cʷ, d, s, t, q⁻)
    ĉ⁺ = tangential_face_value(i + ti, j + tj, k + tk, grid, WCT, c, cʷ, d, s, t, q⁺)
    return @inbounds q⁻ * (ĉ⁻ - cʷ[i, j, k]) - q⁺ * (ĉ⁺ - cʷ[i, j, k])
end

@inline tangential_directions(::Val{1}) = (Val(2), Val(3))
@inline tangential_directions(::Val{2}) = (Val(1), Val(3))
@inline tangential_directions(::Val{3}) = (Val(1), Val(2))

# V ∂ₜcʷ = qⁿ (ĉʷ - cʷ) + Σₜ [q⁻ (ĉ⁻ - cʷ) - q⁺ (ĉ⁺ - cʷ)], with qⁿ the inflow into R through its face opposite the wall
@inline function single_wall_value_tendency(i, j, k, grid, weno::WENO{N, FT, WCT}, U, c, cʷ, d, s) where {N, FT, WCT}
    iᴿ, jᴿ, kᴿ, wall = adjacent_fluid_cell(i, j, k, grid, d, s)

    di, dj, dk = unit_vector(d)
    f  = (1 - s) ÷ 2
    qⁿ = max(s * face_transport(iᴿ + f * di, jᴿ + f * dj, kᴿ + f * dk, grid, U, d), 0)
    Gʷ = @inbounds qⁿ * (normal_face_value(iᴿ, jᴿ, kᴿ, grid, weno, c, d, s) - cʷ[i, j, k])

    t₁, t₂ = tangential_directions(d)
    Gʷ += tangential_tendency(i, j, k, iᴿ, jᴿ, kᴿ, grid, topology(grid, t₁), WCT, U, c, cʷ, d, s, t₁)
    Gʷ += tangential_tendency(i, j, k, iᴿ, jᴿ, kᴿ, grid, topology(grid, t₂), WCT, U, c, cʷ, d, s, t₂)

    return ifelse(wall, Gʷ / Vᶜᶜᶜ(iᴿ, jᴿ, kᴿ, grid), zero(Gʷ)), wall
end

@inline directional_wall_value_tendency(i, j, k, grid, ::Type{Flat}, weno, U, c, cʷ, d) = zero(grid), zero(grid)

@inline function directional_wall_value_tendency(i, j, k, grid, T, weno, U, c, cʷ, d)
    G⁻, w⁻ = single_wall_value_tendency(i, j, k, grid, weno, U, c, cʷ, d, -1)
    G⁺, w⁺ = single_wall_value_tendency(i, j, k, grid, weno, U, c, cʷ, d, +1)
    return G⁻ + G⁺, w⁻ + w⁺
end

# The stored value is stepped as σ cʷ, whose tendency is σ (Gʷ + cʷ ∂ₜσ / σ)
@inline thickness_weighted_tendency(i, j, k, grid, Gʷ, cʷ) = Gʷ
@inline thickness_weighted_tendency(i, j, k, grid::MutableGridOfSomeKind, Gʷ, cʷ) =
    @inbounds (Gʷ + cʷ[i, j, k] * ∂t_σ(i, j, k, grid) / σⁿ(i, j, k, grid, Center(), Center(), Center())) * σⁿ(i, j, k, grid, Center(), Center(), Center())

# A node that stores several walls follows the average of their tendencies
@inline function wall_value_tendency(cʷ::ImmersedWallValues, i, j, k, grid, weno, U, c)
    Gˣ, wˣ = directional_wall_value_tendency(i, j, k, grid, topology(grid, 1), weno, U, c, cʷ, Val(1))
    Gʸ, wʸ = directional_wall_value_tendency(i, j, k, grid, topology(grid, 2), weno, U, c, cʷ, Val(2))
    Gᶻ, wᶻ = directional_wall_value_tendency(i, j, k, grid, topology(grid, 3), weno, U, c, cʷ, Val(3))
    walls = wˣ + wʸ + wᶻ
    return thickness_weighted_tendency(i, j, k, grid, (Gˣ + Gʸ + Gᶻ) / max(walls, one(walls)), cʷ)
end

@inline wall_value_tendency(cʷ::DomainWallValues{d, s}, i, j, k, grid, weno, U, c) where {d, s} = first(single_wall_value_tendency(i, j, k, grid, weno, U, c, cʷ, Val(d), s))

@inline initial_wall_value(i, j, k, grid, ::Type{Flat}, weno, c, d) = zero(grid), zero(grid)

@inline function initial_wall_value(i, j, k, grid, T, weno, c, d)
    total = zero(grid)
    walls = zero(grid)
    for s in (-1, +1)
        iᴿ, jᴿ, kᴿ, wall = adjacent_fluid_cell(i, j, k, grid, d, s)
        total += ifelse(wall, normal_face_value(iᴿ, jᴿ, kᴿ, grid, weno, c, d, s), zero(grid))
        walls += wall
    end
    return total, walls
end

@inline function initial_wall_value(::ImmersedWallValues, i, j, k, grid, weno, c)
    cˣ, wˣ = initial_wall_value(i, j, k, grid, topology(grid, 1), weno, c, Val(1))
    cʸ, wʸ = initial_wall_value(i, j, k, grid, topology(grid, 2), weno, c, Val(2))
    cᶻ, wᶻ = initial_wall_value(i, j, k, grid, topology(grid, 3), weno, c, Val(3))
    return (cˣ + cʸ + cᶻ) / max(wˣ + wʸ + wᶻ, one(grid))
end

@inline function initial_wall_value(::DomainWallValues{d, s}, i, j, k, grid, weno, c) where {d, s}
    iᴿ, jᴿ, kᴿ, wall = adjacent_fluid_cell(i, j, k, grid, Val(d), s)
    return ifelse(wall, normal_face_value(iᴿ, jᴿ, kᴿ, grid, weno, c, Val(d), s), zero(grid))
end

#####
##### Kernels over the stored wall values
#####

# I is the position in `nodes` of an immersed wall, and the position along a wall of the domain
@inline wall_node(walls::ImmersedWallValues, I, grid) = @inbounds walls.nodes[I[1]]
@inline wall_node(::DomainWallValues{d, s}, I, grid) where {d, s} = domain_wall_node(I[1], I[2], grid, Val(d), s)

@inline wall_slot(::ImmersedWallValues, I, i, j, k) = I
@inline wall_slot(::DomainWallValues{d}, I, i, j, k) where d = wall_index(i, j, k, Val(d))

# The walls of the domain store cʷ, the immersed walls σ cʷ
@inline stretching(::ImmersedWallValues, i, j, k, grid) = σⁿ(i, j, k, grid, Center(), Center(), Center())
@inline stretching(::DomainWallValues,   i, j, k, grid) = one(grid)

@inline previous_stretching(::ImmersedWallValues, i, j, k, grid) = Oceananigans.Operators.σ⁻(i, j, k, grid, Center(), Center(), Center())
@inline previous_stretching(::DomainWallValues,   i, j, k, grid) = one(grid)

@kernel function _compute_wall_value_tendency!(walls, grid, weno, U, c)
    I = @index(Global, NTuple)
    i, j, k = wall_node(walls, I, grid)
    @inbounds walls.tendency[wall_slot(walls, I, i, j, k)...] = wall_value_tendency(walls, i, j, k, grid, weno, U, c)
end

@kernel function _initialize_wall_values!(walls, grid, weno, c)
    I = @index(Global, NTuple)
    i, j, k = wall_node(walls, I, grid)
    @inbounds walls[i, j, k] = initial_wall_value(walls, i, j, k, grid, weno, c)
end

@kernel function _cache_wall_values!(walls, grid)
    I = @index(Global, NTuple)
    i, j, k = wall_node(walls, I, grid)
    @inbounds walls.cache[wall_slot(walls, I, i, j, k)...] = walls[i, j, k] * stretching(walls, i, j, k, grid)
end

@kernel function _split_runge_kutta_substep_wall_values!(walls, grid, Δt)
    I = @index(Global, NTuple)
    i, j, k = wall_node(walls, I, grid)
    n = wall_slot(walls, I, i, j, k)
    @inbounds walls[i, j, k] = (walls.cache[n...] + Δt * walls.tendency[n...]) / stretching(walls, i, j, k, grid)
end

@kernel function _adams_bashforth_2_step_wall_values!(walls, grid, Δt, χ)
    I = @index(Global, NTuple)
    i, j, k = wall_node(walls, I, grid)
    n = wall_slot(walls, I, i, j, k)

    FT = eltype(χ)
    α = 3 * one(FT) / 2 + χ
    β = 1 * one(FT) / 2 + χ
    not_euler = χ != convert(FT, -0.5) # use to prevent corruption by leftover NaNs in G⁻

    σᶜᶜⁿ = stretching(walls, i, j, k, grid)
    σᶜᶜ⁻ = previous_stretching(walls, i, j, k, grid)

    @inbounds begin
        Gⁿ = walls.tendency[n...]
        ∂t_σcʷ = α * Gⁿ - β * walls.previous_tendency[n...] * not_euler
        walls[i, j, k] = (σᶜᶜ⁻ * walls[i, j, k] + Δt * ∂t_σcʷ) / σᶜᶜⁿ
        walls.previous_tendency[n...] = Gⁿ
    end
end

@kernel function _runge_kutta_3_substep_wall_values!(walls, grid, Δt, γⁿ, ζⁿ)
    I = @index(Global, NTuple)
    i, j, k = wall_node(walls, I, grid)
    n = wall_slot(walls, I, i, j, k)
    @inbounds begin
        Gⁿ = walls.tendency[n...]
        walls[i, j, k] += Δt * (γⁿ * Gⁿ + ζⁿ * walls.previous_tendency[n...])
        walls.previous_tendency[n...] = Gⁿ
    end
end

@kernel function _runge_kutta_3_substep_wall_values!(walls, grid, Δt, γ¹, ::Nothing)
    I = @index(Global, NTuple)
    i, j, k = wall_node(walls, I, grid)
    n = wall_slot(walls, I, i, j, k)
    @inbounds begin
        Gⁿ = walls.tendency[n...]
        walls[i, j, k] += Δt * γ¹ * Gⁿ
        walls.previous_tendency[n...] = Gⁿ
    end
end

launch_over_wall_values!(::Nothing, kernel!::K, args::Vararg{Any, N}) where {K, N} = nothing

# A map over the positions in `nodes` launches the same kernel for any number of nodes, whose count is known only at runtime
function launch_over_wall_values!(walls::ImmersedWallValues, kernel!::K, args::Vararg{Any, N}) where {K, N}
    grid = walls.values.grid
    launch!(architecture(grid), grid, Base.OneTo(length(walls.nodes)), kernel!, walls, grid, args...)
    return nothing
end

# The positions on the walls of the domain normal to d run along the other two directions
launch_over_wall_values!(walls::DomainWallValues{1}, kernel!::K, args::Vararg{Any, N}) where {K, N} = launch!(architecture(walls.values.grid), walls.values.grid, Val(:yz), kernel!, walls, walls.values.grid, args...)
launch_over_wall_values!(walls::DomainWallValues{2}, kernel!::K, args::Vararg{Any, N}) where {K, N} = launch!(architecture(walls.values.grid), walls.values.grid, Val(:xz), kernel!, walls, walls.values.grid, args...)
launch_over_wall_values!(walls::DomainWallValues{3}, kernel!::K, args::Vararg{Any, N}) where {K, N} = launch!(architecture(walls.values.grid), walls.values.grid, Val(:xy), kernel!, walls, walls.values.grid, args...)

function launch_over_wall_values!(advection::StoredWallValueWENO, kernel!::K, args::Vararg{Any, N}) where {K, N}
    extrapolation = transported_wall_value(advection)
    walls = extrapolation.domain_walls
    launch_over_wall_values!(extrapolation.immersed_walls, kernel!, args...)
    launch_over_wall_values!(walls.x.left,  kernel!, args...)
    launch_over_wall_values!(walls.x.right, kernel!, args...)
    launch_over_wall_values!(walls.y.left,  kernel!, args...)
    launch_over_wall_values!(walls.y.right, kernel!, args...)
    launch_over_wall_values!(walls.z.left,  kernel!, args...)
    launch_over_wall_values!(walls.z.right, kernel!, args...)
    return nothing
end

#####
##### Interface for the models
#####

compute_wall_value_tendency!(advection, U, c) = nothing
compute_wall_value_tendency!(advection::StoredWallValueWENO, U, c) = launch_over_wall_values!(advection, _compute_wall_value_tendency!, advection.buffer_scheme.scheme, U, c)

initialize_wall_values!(advection, c) = nothing
initialize_wall_values!(advection::StoredWallValueWENO, c) = launch_over_wall_values!(advection, _initialize_wall_values!, advection.buffer_scheme.scheme, c)

cache_wall_values!(advection) = nothing
cache_wall_values!(advection::StoredWallValueWENO) = launch_over_wall_values!(advection, _cache_wall_values!)

split_runge_kutta_substep_wall_values!(advection, Δt) = nothing
split_runge_kutta_substep_wall_values!(advection::StoredWallValueWENO, Δt) = launch_over_wall_values!(advection, _split_runge_kutta_substep_wall_values!, Δt)

adams_bashforth_2_step_wall_values!(advection, Δt, χ) = nothing
adams_bashforth_2_step_wall_values!(advection::StoredWallValueWENO, Δt, χ) = launch_over_wall_values!(advection, _adams_bashforth_2_step_wall_values!, Δt, χ)

runge_kutta_3_substep_wall_values!(advection, Δt, γⁿ, ζⁿ) = nothing
runge_kutta_3_substep_wall_values!(advection::StoredWallValueWENO, Δt, γⁿ, ζⁿ) = launch_over_wall_values!(advection, _runge_kutta_3_substep_wall_values!, Δt, γⁿ, ζⁿ)

# The tracers of a model, and the wall values that the advection of each stores.
# map over the values: map over a NamedTuple does not specialize on the function, and boxes the advection schemes
compute_wall_value_tendencies!(advection::NamedTuple, U, tracers) = (map((a, c) -> compute_wall_value_tendency!(a, U, c), values(NamedTuple{keys(tracers)}(advection)), values(tracers)); nothing)
initialize_wall_values!(advection::NamedTuple, tracers) = (map(initialize_wall_values!, values(NamedTuple{keys(tracers)}(advection)), values(tracers)); nothing)
cache_wall_values!(advection::NamedTuple) = (map(cache_wall_values!, values(advection)); nothing)
split_runge_kutta_substep_wall_values!(advection::NamedTuple, Δt) = (map(a -> split_runge_kutta_substep_wall_values!(a, Δt), values(advection)); nothing)
adams_bashforth_2_step_wall_values!(advection::NamedTuple, Δt, χ) = (map(a -> adams_bashforth_2_step_wall_values!(a, Δt, χ), values(advection)); nothing)
runge_kutta_3_substep_wall_values!(advection::NamedTuple, Δt, γⁿ, ζⁿ) = (map(a -> runge_kutta_3_substep_wall_values!(a, Δt, γⁿ, ζⁿ), values(advection)); nothing)

"""
    fill_wall_value_halos!(advection)

Fill the halos of the wall values that `advection` stores, which the reconstruction and the tendencies read across the
boundaries of the grid.
"""
fill_wall_value_halos!(advection) = nothing

wall_values_field(::Nothing) = nothing
wall_values_field(walls) = walls.values

function fill_wall_value_halos!(advection::StoredWallValueWENO)
    extrapolation = transported_wall_value(advection)
    walls = extrapolation.domain_walls
    fill_halo_regions!((extrapolation.immersed_walls.values,
                        wall_values_field(walls.x.left), wall_values_field(walls.x.right),
                        wall_values_field(walls.y.left), wall_values_field(walls.y.right),
                        wall_values_field(walls.z.left), wall_values_field(walls.z.right)))
    return nothing
end

#####
##### Checkpointing
#####

Oceananigans.prognostic_state(::AbstractAdvectionScheme) = nothing
Oceananigans.prognostic_state(advection::StoredWallValueWENO) = Oceananigans.prognostic_state(transported_wall_value(advection))

Oceananigans.prognostic_state(extrapolation::TransportedWallValue) =
    (immersed_walls = Oceananigans.prognostic_state(extrapolation.immersed_walls),
     domain_walls   = Oceananigans.prognostic_state(extrapolation.domain_walls))

Oceananigans.prognostic_state(walls::ImmersedWallValues) =
    (values = Oceananigans.prognostic_state(walls.values), previous_tendency = on_architecture(CPU(), walls.previous_tendency))

Oceananigans.prognostic_state(walls::DomainWallValues) =
    (values = Oceananigans.prognostic_state(walls.values), previous_tendency = Oceananigans.prognostic_state(walls.previous_tendency))

Oceananigans.restore_prognostic_state!(advection::StoredWallValueWENO, ::Nothing) = advection
Oceananigans.restore_prognostic_state!(::ImmersedWallValues, ::Nothing) = nothing
Oceananigans.restore_prognostic_state!(::DomainWallValues, ::Nothing) = nothing

function Oceananigans.restore_prognostic_state!(advection::StoredWallValueWENO, from)
    extrapolation = transported_wall_value(advection)
    Oceananigans.restore_prognostic_state!(extrapolation.immersed_walls, from.immersed_walls)
    Oceananigans.restore_prognostic_state!(extrapolation.domain_walls, from.domain_walls)
    return advection
end

function Oceananigans.restore_prognostic_state!(walls::ImmersedWallValues, from)
    Oceananigans.restore_prognostic_state!(walls.values, from.values)
    copyto!(walls.previous_tendency, from.previous_tendency)
    return walls
end

function Oceananigans.restore_prognostic_state!(walls::DomainWallValues, from)
    Oceananigans.restore_prognostic_state!(walls.values, from.values)
    Oceananigans.restore_prognostic_state!(walls.previous_tendency, from.previous_tendency)
    return walls
end
