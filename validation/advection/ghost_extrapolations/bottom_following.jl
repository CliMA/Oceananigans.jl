# Experiment (not src): hierarchical ghosts, except that along x and y a node without data takes the value of the first node
# with data above it, within N levels (the water on a staircase step); steps taller than N, coasts and domain walls keep the
# hierarchical ghost.
include(joinpath(@__DIR__, "hierarchical.jl"))

@eval Oceananigans.Advection begin

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

    function ghost(n, wall_value)
        iₙ, jₙ, kₙ = stencil_node(i, j, k, dir, s(n))
        q★ = min(N + 1, ntuple(q -> ifelse(inactive_node(iₙ, jₙ, kₙ + q, grid, ℓⁿ...), N + 1, q), Val(N))...)
        step_top = stencil_value(iₙ, jₙ, kₙ + min(q★, N), grid, ψ, args...)
        return ifelse((d != 3) & (q★ ≤ N), step_top, wall_value)
    end

    return ntuple(n -> ifelse(n ≤ nᴸ, ghost(n, continued_value(left, nᴸ + 1 - n)),
                       ifelse(n ≥ nᴿ, ghost(n, continued_value(right, n - nᴿ + 1)), S[n])), Val(2N))
end

end
