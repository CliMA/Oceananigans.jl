@kernel function _assemble_ab2_advective_dissipation!(P, grid, χ, Fⁿ, Fⁿ⁻¹, Uⁿ, Uⁿ⁻¹, cⁿ⁺¹, cⁿ)
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
        u₁ = C₁ * Uⁿ.u[i, j, k] / σⁿ(i, j, k, grid, f, c, c)
        v₁ = C₁ * Uⁿ.v[i, j, k] / σⁿ(i, j, k, grid, c, f, c)
        w₁ = C₁ * Uⁿ.w[i, j, k] / σⁿ(i, j, k, grid, c, c, f)

        u₂ = C₂ * Uⁿ⁻¹.u[i, j, k] / σ⁻(i, j, k, grid, f, c, c)
        v₂ = C₂ * Uⁿ⁻¹.v[i, j, k] / σ⁻(i, j, k, grid, c, f, c)
        w₂ = C₂ * Uⁿ⁻¹.w[i, j, k] / σ⁻(i, j, k, grid, c, c, f)

        fx₁ = C₁ * Fⁿ.x[i, j, k] / σⁿ(i, j, k, grid, f, c, c)
        fy₁ = C₁ * Fⁿ.y[i, j, k] / σⁿ(i, j, k, grid, c, f, c)
        fz₁ = C₁ * Fⁿ.z[i, j, k] / σⁿ(i, j, k, grid, c, c, f)

        fx₂ = C₂ * Fⁿ⁻¹.x[i, j, k] / σ⁻(i, j, k, grid, f, c, c)
        fy₂ = C₂ * Fⁿ⁻¹.y[i, j, k] / σ⁻(i, j, k, grid, c, f, c)
        fz₂ = C₂ * Fⁿ⁻¹.z[i, j, k] / σ⁻(i, j, k, grid, c, c, f)

        P.x[i, j, k] = 2 * δˣc★ * (fx₁ - fx₂) - δˣc² * (u₁ - u₂)
        P.y[i, j, k] = 2 * δʸc★ * (fy₁ - fy₂) - δʸc² * (v₁ - v₂)
        P.z[i, j, k] = 2 * δᶻc★ * (fz₁ - fz₂) - δᶻc² * (w₁ - w₂)
    end
end

@kernel function _assemble_rk3_advective_dissipation!(P, grid, Fⁿ, Uⁿ, cⁿ⁺¹, cⁿ)
    i, j, k = @index(Global, NTuple)

    δˣc★ = δxᶠᶜᶜ(i, j, k, grid, c★, cⁿ⁺¹, cⁿ)
    δˣc² = δxᶠᶜᶜ(i, j, k, grid, c², cⁿ⁺¹, cⁿ)

    δʸc★ = δyᶜᶠᶜ(i, j, k, grid, c★, cⁿ⁺¹, cⁿ)
    δʸc² = δyᶜᶠᶜ(i, j, k, grid, c², cⁿ⁺¹, cⁿ)

    δᶻc★ = δzᶜᶜᶠ(i, j, k, grid, c★, cⁿ⁺¹, cⁿ)
    δᶻc² = δzᶜᶜᶠ(i, j, k, grid, c², cⁿ⁺¹, cⁿ)

    @inbounds begin
        u₁ = Uⁿ.u[i, j, k] / σⁿ(i, j, k, grid, f, c, c)
        v₁ = Uⁿ.v[i, j, k] / σⁿ(i, j, k, grid, c, f, c)
        w₁ = Uⁿ.w[i, j, k] / σⁿ(i, j, k, grid, c, c, f)

        fx₁ = Fⁿ.x[i, j, k] / σⁿ(i, j, k, grid, f, c, c)
        fy₁ = Fⁿ.y[i, j, k] / σⁿ(i, j, k, grid, c, f, c)
        fz₁ = Fⁿ.z[i, j, k] / σⁿ(i, j, k, grid, c, c, f)

        P.x[i, j, k] = 2 * δˣc★ * fx₁ - δˣc² * u₁
        P.y[i, j, k] = 2 * δʸc★ * fy₁ - δʸc² * v₁
        P.z[i, j, k] = 2 * δᶻc★ * fz₁ - δᶻc² * w₁
    end
end

@kernel function _cache_advective_fluxes!(Fⁿ, Fⁿ⁻¹, grid::AbstractGrid, advection, U, c)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Save previous advective fluxes
        Fⁿ⁻¹.x[i, j, k] = Fⁿ.x[i, j, k]
        Fⁿ⁻¹.y[i, j, k] = Fⁿ.y[i, j, k]
        Fⁿ⁻¹.z[i, j, k] = Fⁿ.z[i, j, k]

        # Calculate new advective fluxes
        Fⁿ.x[i, j, k] = _advective_tracer_flux_x(i, j, k, grid, advection, U.u, c) * σⁿ(i, j, k, grid, f, c, c)
        Fⁿ.y[i, j, k] = _advective_tracer_flux_y(i, j, k, grid, advection, U.v, c) * σⁿ(i, j, k, grid, c, f, c)
        Fⁿ.z[i, j, k] = _advective_tracer_flux_z(i, j, k, grid, advection, U.w, c) * σⁿ(i, j, k, grid, c, c, f)
    end
end

@kernel function _cache_advective_fluxes!(Fⁿ, grid::AbstractGrid, advection, U, c)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Calculate new advective fluxes
        Fⁿ.x[i, j, k] = _advective_tracer_flux_x(i, j, k, grid, advection, U.u, c) * σⁿ(i, j, k, grid, f, c, c)
        Fⁿ.y[i, j, k] = _advective_tracer_flux_y(i, j, k, grid, advection, U.v, c) * σⁿ(i, j, k, grid, c, f, c)
        Fⁿ.z[i, j, k] = _advective_tracer_flux_z(i, j, k, grid, advection, U.w, c) * σⁿ(i, j, k, grid, c, c, f)
    end
end