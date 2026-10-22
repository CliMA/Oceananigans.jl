using Oceananigans.Grids: inactive_node, flip_location
using Oceananigans.Utils: newton_div, BackendOptimizedDivision

#####
##### Interpolations to a three-dimensional location along one direction
#####
##### `biased_interpolate_xᶠᶜᶜ(i, j, k, grid, scheme, bias, ψ, args...)` reconstructs `ψ` at `(Face, Center, Center)`
##### from the `(Center, Center, Center)` nodes along `x`. Every scheme reconstructs from the stencil of `load_stencil`.
#####

for (d, ξ) in enumerate((:x, :y, :z)), ℓx in (:ᶜ, :ᶠ), ℓy in (:ᶜ, :ᶠ), ℓz in (:ᶜ, :ᶠ)
    ℓ = map(l -> l == :ᶜ ? Center() : Face(), (ℓx, ℓy, ℓz))
    biased_interpolation    = Symbol(:biased_interpolate_, ξ, ℓx, ℓy, ℓz)
    symmetric_interpolation = Symbol(:symmetric_interpolate_, ξ, ℓx, ℓy, ℓz)

    @eval begin
        @inline $biased_interpolation(i, j, k, grid, scheme, args...)    = biased_interpolate(i, j, k, grid, $(Val(d)), $ℓ, scheme, args...)
        @inline $symmetric_interpolation(i, j, k, grid, scheme, args...) = symmetric_interpolate(i, j, k, grid, $(Val(d)), $ℓ, scheme, args...)
    end
end

#####
##### Stencils continued past walls
#####

@inline stencil_node(i, j, k, ::Val{1}, s) = (i + s, j, k)
@inline stencil_node(i, j, k, ::Val{2}, s) = (i, j + s, k)
@inline stencil_node(i, j, k, ::Val{3}, s) = (i, j, k + s)

@inline node_location(::Val{1}, (ℓx, ℓy, ℓz)) = (flip_location(ℓx), ℓy, ℓz)
@inline node_location(::Val{2}, (ℓx, ℓy, ℓz)) = (ℓx, flip_location(ℓy), ℓz)
@inline node_location(::Val{3}, (ℓx, ℓy, ℓz)) = (ℓx, ℓy, flip_location(ℓz))

@inline stencil_value(i, j, k, grid, ψ, args...)           = @inbounds ψ[i, j, k]
@inline stencil_value(i, j, k, grid, ψ::Callable, args...) = @inbounds ψ(i, j, k, grid, args...)

# φ(a, b) φ(b, c) as the fraction (numerator, denominator), (0, 1) unless a, b and b, c share a sign
@inline function agreement(a, b, c)
    ab, bc = a * b, b * c
    d = (a^2 + b^2) * (b^2 + c^2)
    agree = (ab > 0) & (bc > 0) & (d > 0)
    return ifelse(agree, 4ab * bc, zero(a)), ifelse(agree, d, one(a))
end

# the squared agreement factors (n₁ / d₁)² and (n₂ / d₂)²
@inline agreement_squares((n₁, d₁), (n₂, d₂)) = (newton_div(BackendOptimizedDivision, n₁, d₁)^2, newton_div(BackendOptimizedDivision, n₂, d₂)^2)

# one division in Float64, where d₁ d₂, an eighth power of the differences, stays in range
@inline function agreement_squares((n₁, d₁)::NTuple{2, Float64}, (n₂, d₂)::NTuple{2, Float64})
    D = d₁ * d₂
    r = ifelse(D > 0, newton_div(BackendOptimizedDivision, one(D), D), zero(D))
    return (n₁ * d₂ * r)^2, (n₂ * d₁ * r)^2
end

# the agreement factors (G₁, G₂) of the nodes `c` inward from a wall, starting with the node next to it, of which `available` have data
@inline function wall_agreement(c, available)
    Δ₁ = ifelse(available ≥ 2, c[2] - c[1], zero(c[1]))
    Δ₂ = ifelse(available ≥ 3, c[3] - c[2], Δ₁)
    Δ₃ = ifelse(available ≥ 4, c[4] - c[3], Δ₂)
    Δ₄ = ifelse(available ≥ 5, c[5] - c[4], Δ₃)
    κ₁, κ₂, κ₃ = Δ₁ - Δ₂, Δ₂ - Δ₃, Δ₃ - Δ₄
    slope_agreement     = agreement(Δ₁, Δ₂, Δ₃)
    curvature_agreement = agreement(κ₁, κ₂, κ₃)
    return agreement_squares(slope_agreement, curvature_agreement)
end

# Second and First order schemes
@inline wall_agreement(c::NTuple{1}, available) = (zero(c[1]), zero(c[1]))

# Third and Fourth order schemes, all the others use the general `wall_agreement`
@inline function wall_agreement(c::NTuple{3}, available)
    Δ₁ = ifelse(available ≥ 2, c[2] - c[1], zero(c[1]))
    Δ₂ = ifelse(available ≥ 3, c[3] - c[2], Δ₁)
    n, d = agreement(Δ₁, Δ₂, Δ₂)
    return newton_div(BackendOptimizedDivision, n, d)^2, zero(Δ₁)
end

# the node next to the wall, the ghost slope and the ghost curvature, with the agreement factors (G₁, G₂)
@inline function wall_continuation(c, available, (G₁, G₂))
    Δ₁ = ifelse(available ≥ 2, c[2] - c[1], zero(c[1]))
    Δ₂ = ifelse(available ≥ 3, c[3] - c[2], Δ₁)
    return c[1], (G₂ + (1 - G₂) * G₁) * Δ₁, G₂ * (Δ₁ - Δ₂)
end

@inline wall_continuation(c::NTuple{1}, available, G) = (c[1], zero(c[1]), zero(c[1]))

@inline continued_value((c₁, σ, κ), g) = c₁ - g * σ - g * (g + 1) * κ / 2

# `ψ` continued past walls with the agreement factors `G = (Gᴸ, Gᴿ)`, its own when `nothing`; returns the stencil and the factors
@inline function load_stencil(i, j, k, grid, dir::Val{d}, ℓ, ::Val{N}, G::Union{Nothing, Tuple}, ψ, args...) where {d, N}
    ℓⁿ = node_location(dir, ℓ)
    σ  = ℓ[d] isa Center

    s(n) = n - N - 1 + σ
    # a face node lies past a wall when the cell between it and the point is inactive: a one-cell wall leaves its faces peripheral
    no_data(n) = ifelse(σ, inactive_node(stencil_node(i, j, k, dir, s(n) - (n > N))..., grid, ℓ...),
                           inactive_node(stencil_node(i, j, k, dir, s(n))...,           grid, ℓⁿ...))

    nᴸ = max(0,      ntuple(n -> ifelse(no_data(n),     n,      0),     Val(N))...)
    nᴿ = min(2N + 1, ntuple(n -> ifelse(no_data(N + n), N + n, 2N + 1), Val(N))...)

    S = ntuple(n -> stencil_value(stencil_node(i, j, k, dir, s(n))..., grid, ψ, args...), Val(2N))

    # the q-th node inward from each wall, picked along the scan that finds it; no wall shares the values of a wall at 1 or 2N
    R = Val(min(5, 2N - 1))

    left_wall_positions  = ntuple(n -> n + 1,  Val(N - 1))
    right_wall_positions = ntuple(n -> 2N - n, Val(N - 1))

    left_inward_node(q)  = foldl((c, n) -> ifelse(no_data(n), S[min(n + q, 2N)], c), left_wall_positions;  init = S[min(1 + q, 2N)])
    right_inward_node(q) = foldl((c, n) -> ifelse(no_data(n), S[max(n - q, 1)],  c), right_wall_positions; init = S[max(2N - q, 1)])

    cᴸ = ntuple(left_inward_node,  R)
    cᴿ = ntuple(right_inward_node, R)

    availableᴸ = min(nᴿ - 1, 2N) - nᴸ
    availableᴿ = nᴿ - 1 - nᴸ

    Gᴸ     = wall_agreement(cᴸ, availableᴸ)
    Gᴿ     = wall_agreement(cᴿ, availableᴿ)
    Gᴸ, Gᴿ = something(G, (Gᴸ, Gᴿ))
    left   = wall_continuation(cᴸ, availableᴸ, Gᴸ)
    right  = wall_continuation(cᴿ, availableᴿ, Gᴿ)

    left_ghost(n)  = continued_value(left,  nᴸ + 1 - n)
    right_ghost(n) = continued_value(right, n - nᴿ + 1)

    stencil = ntuple(n -> ifelse(n ≤ nᴸ, left_ghost(n), ifelse(n ≥ nᴿ, right_ghost(n), S[n])), Val(2N))

    return stencil, (Gᴸ, Gᴿ)
end

"""
$(TYPEDSIGNATURES)

Return the `2N` values of `ψ` that reconstruct `ψ` at `(i, j, k)`, located at `ℓ`, along the direction `dir`.

Scanning outward from `(i, j, k)` on each side, the first node without data marks a wall. Past the wall the g-th node takes
the ghost

```math
c₁ - g σ - g (g + 1) κ / 2, \\quad σ = [G₂ + (1 - G₂) G₁] Δ₁, \\quad κ = G₂ κ₁,
\\quad G₁ = [φ(Δ₁, Δ₂) φ(Δ₂, Δ₃)]², \\quad G₂ = [φ(κ₁, κ₂) φ(κ₂, κ₃)]²,
```

where `c₁` is the node with data next to the wall, `Δⱼ` the differences inward from it, `κⱼ = Δⱼ - Δⱼ₊₁`, and
`φ(a, b) = 2ab / (a² + b²)` when `a` and `b` share a sign, zero otherwise. Linear data, and quadratic data on five nodes, are
continued exactly, and next to a front (`G₁ = G₂ = 0`) the ghosts are constant.
"""
@inline load_stencil(i, j, k, grid, dir, ℓ, N::Val, ψ, args...) = first(load_stencil(i, j, k, grid, dir, ℓ, N, nothing, ψ, args...))
