@kernel function _assemble_diffusive_tracer_dissipation!(K, grid, χ, Vⁿ, Vⁿ⁻¹, Uⁿ⁺¹, cⁿ⁺¹, cⁿ)
    i, j, k = @index(Global, NTuple)
    compute_diffusive_dissipation!(K, i, j, k, grid, Vⁿ, Vⁿ⁻¹, χ, cⁿ⁺¹, cⁿ)
end

@inline function compute_diffusive_tracer_dissipation!(K::Tuple, i, j, k, grid, Vⁿ, Vⁿ⁻¹, χ, cⁿ⁺¹, cⁿ)
    for n in eachindex(K)
        compute_diffusive_dissipation!(K[n], i, j, k, grid, Vⁿ[n], Vⁿ⁻¹[n], χ, cⁿ⁺¹, cⁿ)
    end
end

@inline function compute_diffusive_tracer_dissipation!(K, i, j, k, grid, Vⁿ, Vⁿ⁻¹, χ, cⁿ⁺¹, cⁿ)
    C₁  = convert(eltype(grid), 1.5 + χ)
    C₂  = convert(eltype(grid), 0.5 + χ)

    δˣc★ = δxᶠᶜᶜ(i, j, k, grid, c★, cⁿ⁺¹, cⁿ)    
    δʸc★ = δyᶜᶠᶜ(i, j, k, grid, c★, cⁿ⁺¹, cⁿ)
    δᶻc★ = δzᶜᶜᶠ(i, j, k, grid, c★, cⁿ⁺¹, cⁿ)
    
    @inbounds begin
        K.x[i, j, k] = 2 * δˣc★ * (C₁ * Vⁿ.x[i, j, k] - C₂ * Vⁿ⁻¹.x[i, j, k])
        K.y[i, j, k] = 2 * δʸc★ * (C₁ * Vⁿ.y[i, j, k] - C₂ * Vⁿ⁻¹.y[i, j, k])
        K.z[i, j, k] = 2 * δᶻc★ * (C₁ * Vⁿ.z[i, j, k] - C₂ * Vⁿ⁻¹.z[i, j, k])
    end
end
