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
        u₁ = C₁ * Uⁿ.u[i, j, k]
        v₁ = C₁ * Uⁿ.v[i, j, k]
        w₁ = C₁ * Uⁿ.w[i, j, k]

        u₂ = C₂ * Uⁿ⁻¹.u[i, j, k]
        v₂ = C₂ * Uⁿ⁻¹.v[i, j, k]
        w₂ = C₂ * Uⁿ⁻¹.w[i, j, k]

        fx₁ = C₁ * Fⁿ.x[i, j, k]
        fy₁ = C₁ * Fⁿ.y[i, j, k]
        fz₁ = C₁ * Fⁿ.z[i, j, k]

        fx₂ = C₂ * Fⁿ⁻¹.x[i, j, k]
        fy₂ = C₂ * Fⁿ⁻¹.y[i, j, k]
        fz₂ = C₂ * Fⁿ⁻¹.z[i, j, k]

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
        u₁ = Uⁿ.u[i, j, k]
        v₁ = Uⁿ.v[i, j, k]
        w₁ = Uⁿ.w[i, j, k]

        fx₁ = Fⁿ.x[i, j, k]
        fy₁ = Fⁿ.y[i, j, k]
        fz₁ = Fⁿ.z[i, j, k]

        P.x[i, j, k] = 2 * δˣc★ * fx₁ - δˣc² * u₁
        P.y[i, j, k] = 2 * δʸc★ * fy₁ - δʸc² * v₁
        P.z[i, j, k] = 2 * δᶻc★ * fz₁ - δᶻc² * w₁
    end
end

@kernel function _cache_advective_fluxes!(Fⁿ, Fⁿ⁻¹, grid::AbstractGrid, advection, U, tracer)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Save previous advective fluxes
        Fⁿ⁻¹.x[i, j, k] = Fⁿ.x[i, j, k]
        Fⁿ⁻¹.y[i, j, k] = Fⁿ.y[i, j, k]
        Fⁿ⁻¹.z[i, j, k] = Fⁿ.z[i, j, k]

        # Calculate new advective fluxes
        Fⁿ.x[i, j, k] = _advective_tracer_flux_x(i, j, k, grid, advection, U.u, tracer)
        Fⁿ.y[i, j, k] = _advective_tracer_flux_y(i, j, k, grid, advection, U.v, tracer)
        Fⁿ.z[i, j, k] = _advective_tracer_flux_z(i, j, k, grid, advection, U.w, tracer)
    end
end

@kernel function _cache_advective_fluxes!(Fⁿ, grid::AbstractGrid, advection, U, tracer)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Calculate new advective fluxes
        Fⁿ.x[i, j, k] = _advective_tracer_flux_x(i, j, k, grid, advection, U.u, tracer)
        Fⁿ.y[i, j, k] = _advective_tracer_flux_y(i, j, k, grid, advection, U.v, tracer)
        Fⁿ.z[i, j, k] = _advective_tracer_flux_z(i, j, k, grid, advection, U.w, tracer)
    end
end
