# Experiment (not src): hierarchical ghosts (main's Centered(order=order-1) advecting velocity) + the #6267 smoothness rule: when the
# stencil of ψ has a ghost, the WENO weights of FunctionStencil and VelocityStencil reconstructions use ψ's own ghosted
# differences instead of the smoothness function's (ghost_cell_reconstruction.jl: weight_differences(ghosted, δ, δˢ)).
include(joinpath(@__DIR__, "hierarchical.jl"))

@eval Oceananigans.Advection begin

@inline function stencil_has_ghost(i, j, k, grid, dir::Val{d}, ℓ, ::Val{N}) where {d, N}
    ℓⁿ = node_location(dir, ℓ)
    σ  = ℓ[d] isa Center
    no_data(n) = inactive_node(stencil_node(i, j, k, dir, n - N - 1 + σ)..., grid, ℓⁿ...)
    return |(ntuple(no_data, Val(2N))...)
end

@inline function biased_interpolate(i, j, k, grid, dir, ℓ, scheme::WENO{N, FT}, bias, ψ, VI::FunctionStencil, args...) where {N, FT}
    S  = weno_stencil(i, j, k, grid, dir, ℓ, scheme, bias, ψ, args...)
    ψ₀ = weno_anchor(scheme, S)
    δ  = weno_differences(scheme, S)
    δˢ = weno_differences(scheme, weno_stencil(i, j, k, grid, dir, ℓ, scheme, bias, VI.func, args...))
    ghosted = stencil_has_ghost(i, j, k, grid, dir, ℓ, Val(N))
    ω  = biased_weno_weights(ifelse(ghosted, δ, δˢ), grid, scheme, bias, VI, args...)
    return weno_reconstruction(scheme, ψ₀, δ, ω)
end

@inline function biased_interpolate(i, j, k, grid, dir, ℓ, scheme::WENO{N, FT}, bias, ψ, VI::VelocityStencil, u, v, args...) where {N, FT}
    S  = weno_stencil(i, j, k, grid, dir, ℓ, scheme, bias, ψ, u, v, args...)
    ψ₀ = weno_anchor(scheme, S)
    δ  = weno_differences(scheme, S)
    ghosted = stencil_has_ghost(i, j, k, grid, dir, ℓ, Val(N))
    uₛ = weno_stencil(i, j, k, grid, dir, ℓ, scheme, bias, ℑyᵃᶠᵃ, u)
    vₛ = weno_stencil(i, j, k, grid, dir, ℓ, scheme, bias, ℑxᶠᵃᵃ, v)
    βᵤ = beta_loop(scheme, ifelse(ghosted, δ, weno_differences(scheme, uₛ)))
    βᵥ = beta_loop(scheme, ifelse(ghosted, δ, weno_differences(scheme, vₛ)))
    β  = beta_sum(scheme, βᵤ, βᵥ)
    τ  = global_smoothness_indicator(Val(N), β)
    α  = zweno_alpha_loop(scheme, β, τ)
    ω  = α .* weight_normalization(scheme, α)
    return weno_reconstruction(scheme, ψ₀, δ, ω)
end

end
