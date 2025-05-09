@kernel function _assemble_diffusive_dissipation!(K, grid, χ, Vⁿ, Vⁿ⁻¹, cⁿ⁺¹, cⁿ)
    i, j, k = @index(Global, NTuple)
    compute_diffusive_dissipation!(K, i, j, k, grid, Vⁿ, Vⁿ⁻¹, χ, cⁿ⁺¹, cⁿ)
end

@inline compute_diffusive_dissipation!(K, i, j, k, grid, Vⁿ::Nothing, Vⁿ⁻¹, χ, cⁿ⁺¹, cⁿ) = nothing
@inline compute_diffusive_dissipation!(::Tuple, i, j, k, grid, Vⁿ::Nothing, Vⁿ⁻¹, χ, cⁿ⁺¹, cⁿ) = nothing

@inline function compute_diffusive_dissipation!(K::Tuple, i, j, k, grid, Vⁿ, Vⁿ⁻¹, χ, cⁿ⁺¹, cⁿ)
    for n in eachindex(K)
        compute_diffusive_dissipation!(K[n], i, j, k, grid, Vⁿ[n], Vⁿ⁻¹[n], χ, cⁿ⁺¹, cⁿ)
    end
    return nothing
end

@inline function compute_diffusive_dissipation!(K, i, j, k, grid, Vⁿ, Vⁿ⁻¹, χ, cⁿ⁺¹, cⁿ)
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

    return nothing
end

@kernel function _cache_diffusive_fluxes!(Vⁿ, Vⁿ⁻¹, grid, closure, diffusivity, bouyancy, c, tracer_id, clk, model_fields) 
    i, j, k = @index(Global, NTuple)
    compute_diffusive_fluxes!(Vⁿ, Vⁿ⁻¹, i, j, k, grid, closure, diffusivity, bouyancy, c, tracer_id, clk, model_fields)
end

@inline compute_diffusive_fluxes!(::Nothing, Vⁿ⁻¹, i, j, k, grid, clo, K, args...) = nothing
@inline compute_diffusive_fluxes!(::Nothing, Vⁿ⁻¹, i, j, k, grid, ::Tuple, K, args...) = nothing

@inline function compute_diffusive_fluxes!(Vⁿ, Vⁿ⁻¹, i, j, k, grid, closure::Tuple, K, args...) 
    for n in eachindex(closure)
        compute_diffusive_fluxes!(Vⁿ[n], Vⁿ⁻¹[n], i, j, k, grid, closure[n], K[n], args...)
    end

    return nothing
end

@inline function compute_diffusive_fluxes!(Vⁿ, Vⁿ⁻¹, i, j, k, grid, clo, K, b, c, c_id, clk, fields)
    @inbounds begin
        Vⁿ⁻¹.x[i, j, k] = Vⁿ.x[i, j, k] 
        Vⁿ⁻¹.y[i, j, k] = Vⁿ.y[i, j, k] 
        Vⁿ⁻¹.z[i, j, k] = Vⁿ.z[i, j, k] 

        Vⁿ.x[i, j, k] = _diffusive_flux_x(i, j, k, grid, clo, K, Val(c_id), c, clk, fields, b) * Axᶠᶜᶜ(i, j, k, grid) * σⁿ(i, j, k, grid, f, c, c)
        Vⁿ.y[i, j, k] = _diffusive_flux_y(i, j, k, grid, clo, K, Val(c_id), c, clk, fields, b) * Ayᶜᶠᶜ(i, j, k, grid) * σⁿ(i, j, k, grid, c, f, c)
        Vⁿ.z[i, j, k] = _diffusive_flux_z(i, j, k, grid, clo, K, Val(c_id), c, clk, fields, b) * Azᶜᶜᶠ(i, j, k, grid) * σⁿ(i, j, k, grid, c, c, f)
    end
    
    return nothing
end
