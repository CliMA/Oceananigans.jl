# Experiment (not src): load_stencil with HierarchicalExtrapolation ghosts (extrapolations.jl of prs/hierarchical-ghost-cells).
# Past a wall the g-th ghost is c₁ - g σ - g (g + 1) κ / 2, from the nodes with data next to the wall inside the stencil:
# φ(a, b) = 2ab / (a² + b²) (0 unless a, b share a sign), G₁ = [φ(Δ₁, Δ₂) φ(Δ₂, Δ₃)]², G₂ = [φ(κ₁, κ₂) φ(κ₂, κ₃)]²,
# σ = [G₂ + (1 - G₂) G₁] Δ₁, κ = G₂ κ₁; missing differences repeat the last one.
@eval Oceananigans.Advection begin

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

@inline function load_stencil(i, j, k, grid, dir::Val{d}, ℓ, ::Val{N}, ψ, args...) where {d, N}
    ℓⁿ = node_location(dir, ℓ)
    σ  = ℓ[d] isa Center

    s(n) = n - N - 1 + σ
    no_data(n) = inactive_node(stencil_node(i, j, k, dir, s(n))..., grid, ℓⁿ...)

    nᴸ = max(0,      ntuple(n -> ifelse(no_data(n),     n,      0),     Val(N))...)
    nᴿ = min(2N + 1, ntuple(n -> ifelse(no_data(N + n), N + n, 2N + 1), Val(N))...)

    m(n) = clamp(n, nᴸ + 1, nᴿ - 1)
    S = ntuple(n -> stencil_value(stencil_node(i, j, k, dir, s(m(n)))..., grid, ψ, args...), Val(2N))

    left  = wall_continuation(S, nᴸ + 1, +1, min(nᴿ - 1, 2N) - nᴸ)
    right = wall_continuation(S, nᴿ - 1, -1, nᴿ - 1 - nᴸ)

    return ntuple(n -> ifelse(n ≤ nᴸ, continued_value(left, nᴸ + 1 - n), ifelse(n ≥ nᴿ, continued_value(right, n - nᴿ + 1), S[n])), Val(2N))
end

end
