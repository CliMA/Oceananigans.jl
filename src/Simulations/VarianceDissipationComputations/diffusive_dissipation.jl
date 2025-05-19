@kernel function _assemble_ab2_diffusive_dissipation!(K, grid, χ, Vⁿ, Vⁿ⁻¹, cⁿ⁺¹, cⁿ)
    i, j, k = @index(Global, NTuple)

    C₁  = convert(eltype(grid), 1.5 + χ)
    C₂  = convert(eltype(grid), 0.5 + χ)

    δˣc★ = δxᶠᶜᶜ(i, j, k, grid, c★, cⁿ⁺¹, cⁿ)    
    δʸc★ = δyᶜᶠᶜ(i, j, k, grid, c★, cⁿ⁺¹, cⁿ)
    δᶻc★ = δzᶜᶜᶠ(i, j, k, grid, c★, cⁿ⁺¹, cⁿ)
    
    @inbounds begin
        vx₁ = C₁ * Vⁿ.x[i, j, k] / σⁿ(i, j, k, grid, f, c, c)
        vy₁ = C₁ * Vⁿ.y[i, j, k] / σⁿ(i, j, k, grid, c, f, c)
        vz₁ = C₁ * Vⁿ.z[i, j, k] / σⁿ(i, j, k, grid, c, c, f)

        vx₂ = C₂ * Vⁿ⁻¹.x[i, j, k] / σ⁻(i, j, k, grid, f, c, c)
        vy₂ = C₂ * Vⁿ⁻¹.y[i, j, k] / σ⁻(i, j, k, grid, c, f, c)
        vz₂ = C₂ * Vⁿ⁻¹.z[i, j, k] / σ⁻(i, j, k, grid, c, c, f)

        K.x[i, j, k] = 2 * δˣc★ * (vx₁ - vx₂)
        K.y[i, j, k] = 2 * δʸc★ * (vy₁ - vy₂)
        K.z[i, j, k] = 2 * δᶻc★ * (vz₁ - vz₂)
    end
end

@kernel function _assemble_rk3_diffusive_dissipation!(K, grid, Vⁿ, cⁿ⁺¹, cⁿ)
    i, j, k = @index(Global, NTuple)

    δˣc★ = δxᶠᶜᶜ(i, j, k, grid, c★, cⁿ⁺¹, cⁿ)    
    δʸc★ = δyᶜᶠᶜ(i, j, k, grid, c★, cⁿ⁺¹, cⁿ)
    δᶻc★ = δzᶜᶜᶠ(i, j, k, grid, c★, cⁿ⁺¹, cⁿ)
    
    @inbounds begin
        vx₁ = Vⁿ.x[i, j, k] / σⁿ(i, j, k, grid, f, c, c)
        vy₁ = Vⁿ.y[i, j, k] / σⁿ(i, j, k, grid, c, f, c)
        vz₁ = Vⁿ.z[i, j, k] / σⁿ(i, j, k, grid, c, c, f)

        K.x[i, j, k] = 2 * δˣc★ * vx₁ 
        K.y[i, j, k] = 2 * δʸc★ * vy₁ 
        K.z[i, j, k] = 2 * δᶻc★ * vz₁ 
    end
end

@kernel function _cache_diffusive_fluxes!(Vⁿ, Vⁿ⁻¹, grid, clo, K, b, c, c_id, clk, fields) 
    i, j, k = @index(Global, NTuple)

    Vⁿ⁻¹.x[i, j, k] = Vⁿ.x[i, j, k] 
    Vⁿ⁻¹.y[i, j, k] = Vⁿ.y[i, j, k] 
    Vⁿ⁻¹.z[i, j, k] = Vⁿ.z[i, j, k] 

    Vⁿ.x[i, j, k] = zero(grid)
    Vⁿ.y[i, j, k] = zero(grid)
    Vⁿ.z[i, j, k] = zero(grid)

    compute_diffusive_fluxes!(Vⁿ, 1, i, j, k, grid, clo, K, b, c, c_id, clk, fields) 
end

@kernel function _cache_diffusive_fluxes!(Vⁿ, grid, ::Val{3}, ℂ, clo, K, b, c, c_id, clk, fields) 
    i, j, k = @index(Global, NTuple)    

    Vⁿ.x[i, j, k] = zero(grid)
    Vⁿ.y[i, j, k] = zero(grid)
    Vⁿ.z[i, j, k] = zero(grid)

    compute_diffusive_fluxes!(Vⁿ, ℂ, i, j, k, grid, clo, K, b, c, c_id, clk, fields) 
end

@kernel function _cache_diffusive_fluxes!(Vⁿ, grid, substep, ℂ, clo, K, b, c, c_id, clk, fields) 
    i, j, k = @index(Global, NTuple)    
    compute_diffusive_fluxes!(Vⁿ, ℂ, i, j, k, grid, clo, K, b, c, c_id, clk, fields) 
end

@inline function compute_diffusive_fluxes!(Vⁿ, ℂ, i, j, k, grid, clo::Tuple, K::Tuple, args...)
    for n in eachindex(clo)
        compute_diffusive_fluxes!(Vⁿ, i, j, k, grid, clo[n], K[n], args...)
    end
end

@inline function compute_diffusive_fluxes!(Vⁿ, ℂ, i, j, k, grid, clo, K, b, c, c_id, clk, fields)
    @inbounds begin
        Vⁿ.x[i, j, k] += _diffusive_flux_x(i, j, k, grid, clo, K, c_id, c, clk, fields, b) * Axᶠᶜᶜ(i, j, k, grid) * σⁿ(i, j, k, grid, f, c, c) * ℂ
        Vⁿ.y[i, j, k] += _diffusive_flux_y(i, j, k, grid, clo, K, c_id, c, clk, fields, b) * Ayᶜᶠᶜ(i, j, k, grid) * σⁿ(i, j, k, grid, c, f, c) * ℂ
        Vⁿ.z[i, j, k] += _diffusive_flux_z(i, j, k, grid, clo, K, c_id, c, clk, fields, b) * Azᶜᶜᶠ(i, j, k, grid) * σⁿ(i, j, k, grid, c, c, f) * ℂ
    end
    return nothing
end