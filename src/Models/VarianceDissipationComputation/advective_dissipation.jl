# TODO: This is only for AB2, figure out how to generalize this for other timesteppers for example RK3
@kernel function _assemble_advective_tracer_dissipation!(P, grid, χ, Fⁿ, Fⁿ⁻¹, Uⁿ⁺¹, Uⁿ, Uⁿ⁻¹, cⁿ⁺¹, cⁿ)
    i, j, k = @index(Global, NTuple)

    δˣc★ = δxᶠᶜᶜ(i, j, k, grid, c★, cⁿ⁺¹, cⁿ)
    δˣc² = δxᶠᶜᶜ(i, j, k, grid, c², cⁿ⁺¹, cⁿ)
    
    δʸc★ = δyᶜᶠᶜ(i, j, k, grid, c★, cⁿ⁺¹, cⁿ)
    δʸc² = δyᶜᶠᶜ(i, j, k, grid, c², cⁿ⁺¹, cⁿ)
    
    δᶻc★ = δzᶜᶜᶠ(i, j, k, grid, c★, cⁿ⁺¹, cⁿ)
    δᶻc² = δzᶜᶜᶠ(i, j, k, grid, c², cⁿ⁺¹, cⁿ)
    
    C₁  = convert(eltype(grid), 1.5 + χ)
    C₂  = convert(eltype(grid), 0.5 + χ)

    @inbounds begin
        u₁ = C₁ *   Uⁿ.u[i, j, k] 
        u₂ = C₂ * Uⁿ⁻¹.u[i, j, k] 
        v₁ = C₁ *   Uⁿ.v[i, j, k] 
        v₂ = C₂ * Uⁿ⁻¹.v[i, j, k] 
        w₁ = C₁ *   Uⁿ.w[i, j, k] 
        w₂ = C₂ * Uⁿ⁻¹.w[i, j, k] 

        fx₁ = C₁ * Fⁿ.x[i, j, k]
        fx₂ = C₂ * Fⁿ⁻¹.x[i, j, k]
        fy₁ = C₁ * Fⁿ.y[i, j, k]
        fy₂ = C₂ * Fⁿ⁻¹.y[i, j, k]
        fz₁ = C₁ * Fⁿ.z[i, j, k]
        fz₂ = C₂ * Fⁿ⁻¹.z[i, j, k]

        P.x[i, j, k] = 2 * δˣc★ * (fx₁ - fx₂) - δˣc² * (u₁ - u₂)
        P.y[i, j, k] = 2 * δʸc★ * (fy₁ - fy₂) - δʸc² * (v₁ - v₂)
        P.z[i, j, k] = 2 * δᶻc★ * (fz₁ - fz₂) - δᶻc² * (w₁ - w₂)
    end
end

@kernel function _assemble_advective_vorticity_dissipation!(P, grid, χ, Fⁿ, Fⁿ⁻¹, Uⁿ⁺¹, Uⁿ, Uⁿ⁻¹, c, ζⁿ)
    i, j, k = @index(Global, NTuple)

    δˣζ★ = δxᶠᶜᶜ(i, j, k, grid, ζ★, Uⁿ⁺¹.u, Uⁿ⁺¹.v, ζⁿ)
    δˣζ² = δxᶠᶜᶜ(i, j, k, grid, ζ², Uⁿ⁺¹.u, Uⁿ⁺¹.v, ζⁿ)

    δʸζ★ = δyᶜᶠᶜ(i, j, k, grid, ζ★, Uⁿ⁺¹.u, Uⁿ⁺¹.v, ζⁿ)
    δʸζ² = δyᶜᶠᶜ(i, j, k, grid, ζ², Uⁿ⁺¹.u, Uⁿ⁺¹.v, ζⁿ)

    @inbounds begin
        u₁ = C₁ * ℑxyᶜᶠᵃ(i, j, k, grid, Δy_qᶠᶜᶜ, Uⁿ.u)   / Δyᶠᶜᶜ(i, j, k, grid)
        u₂ = C₂ * ℑxyᶜᶠᵃ(i, j, k, grid, Δy_qᶠᶜᶜ, Uⁿ⁻¹.u) / Δyᶠᶜᶜ(i, j, k, grid)
        v₁ = C₁ * ℑxyᶠᶜᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, Uⁿ.v)   / Δxᶜᶠᶜ(i, j, k, grid)
        v₂ = C₂ * ℑxyᶠᶜᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, Uⁿ⁻¹.v) / Δxᶜᶠᶜ(i, j, k, grid)

        fx₁ = C₁ * Fⁿ.x[i, j, k]
        fx₂ = C₂ * Fⁿ⁻¹.x[i, j, k]
        fy₁ = C₁ * Fⁿ.y[i, j, k]
        fy₂ = C₂ * Fⁿ⁻¹.y[i, j, k]

        P.x[i, j, k] = 2 * δˣζ★ * (fx₁ - fx₂) - δˣζ² * (u₁ - u₂)
        P.y[i, j, k] = 2 * δʸζ★ * (fy₁ - fy₂) - δʸζ² * (v₁ - v₂)
    end
end
