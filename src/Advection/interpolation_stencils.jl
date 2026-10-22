using Oceananigans.Grids: inactive_node, flip_location

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

@inline stencil_value(i, j, k, grid, ψ, args...) = @inbounds ψ[i, j, k]
@inline stencil_value(i, j, k, grid, ψ::Callable, args...) = @inbounds ψ(i, j, k, grid, args...)

@inline agreement(a, b) = ifelse(a * b > 0, 2a * b / (a^2 + b^2), zero(a))

# S at a runtime position p, without indexing the tuple at runtime
@inline pick(S, p) = foldl((value, n) -> ifelse(n == p, S[n], value), ntuple(identity, Val(length(S))); init = S[1])

# first node with data next to the wall (`first`), inward direction `step`, `available` nodes with data inward
@inline function wall_continuation(S, first, step, available)
    c(q) = pick(S, first + (q - 1) * step)
    Δ₁ = ifelse(available ≥ 2, c(2) - c(1), zero(S[1]))
    Δ₂ = ifelse(available ≥ 3, c(3) - c(2), Δ₁)
    Δ₃ = ifelse(available ≥ 4, c(4) - c(3), Δ₂)
    Δ₄ = ifelse(available ≥ 5, c(5) - c(4), Δ₃)
    κ₁, κ₂, κ₃ = Δ₁ - Δ₂, Δ₂ - Δ₃, Δ₃ - Δ₄
    G₁ = (agreement(Δ₁, Δ₂) * agreement(Δ₂, Δ₃))^2
    G₂ = (agreement(κ₁, κ₂) * agreement(κ₂, κ₃))^2
    return c(1), (G₂ + (1 - G₂) * G₁) * Δ₁, G₂ * κ₁
end

@inline continued_value((c₁, σ, κ), g) = c₁ - g * σ - g * (g + 1) * κ / 2

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
@inline function load_stencil(i, j, k, grid, dir::Val{d}, ℓ, ::Val{N}, ψ, args...) where {d, N}
    ℓⁿ = node_location(dir, ℓ)
    σ  = ℓ[d] isa Center

    s(n) = n - N - 1 + σ
    # a face node lies past a wall when the cell between it and the point is inactive: a one-cell wall leaves its faces peripheral
    no_data(n) = ifelse(σ, inactive_node(stencil_node(i, j, k, dir, s(n) - (n > N))..., grid, ℓ...),
                           inactive_node(stencil_node(i, j, k, dir, s(n))..., grid, ℓⁿ...))

    nᴸ = max(0,      ntuple(n -> ifelse(no_data(n),     n,      0),     Val(N))...)
    nᴿ = min(2N + 1, ntuple(n -> ifelse(no_data(N + n), N + n, 2N + 1), Val(N))...)

    m(n) = clamp(n, nᴸ + 1, nᴿ - 1)
    S = ntuple(n -> stencil_value(stencil_node(i, j, k, dir, s(m(n)))..., grid, ψ, args...), Val(2N))

    left  = wall_continuation(S, nᴸ + 1, +1, min(nᴿ - 1, 2N) - nᴸ)
    right = wall_continuation(S, nᴿ - 1, -1, nᴿ - 1 - nᴸ)

    return ntuple(n -> ifelse(n ≤ nᴸ, continued_value(left, nᴸ + 1 - n), ifelse(n ≥ nᴿ, continued_value(right, n - nᴿ + 1), S[n])), Val(2N))
end
