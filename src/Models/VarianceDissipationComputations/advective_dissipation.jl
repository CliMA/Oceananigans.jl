@kernel function _assemble_advective_dissipation!(P, grid, C₁, C₂, Fⁿ, Fⁿ⁻¹, Uⁿ, Uⁿ⁻¹, cⁿ⁺¹, cⁿ)
    i, j, k = @index(Global, NTuple)

    δˣc★ = δxᶠᶜᶜ(i, j, k, grid, c★, cⁿ⁺¹, cⁿ)
    δˣc² = δxᶠᶜᶜ(i, j, k, grid, c², cⁿ⁺¹, cⁿ)

    δʸc★ = δyᶜᶠᶜ(i, j, k, grid, c★, cⁿ⁺¹, cⁿ)
    δʸc² = δyᶜᶠᶜ(i, j, k, grid, c², cⁿ⁺¹, cⁿ)

    δᶻc★ = δzᶜᶜᶠ(i, j, k, grid, c★, cⁿ⁺¹, cⁿ)
    δᶻc² = δzᶜᶜᶠ(i, j, k, grid, c², cⁿ⁺¹, cⁿ)

    @inbounds begin
        fx = C₁ * Fⁿ.x[i, j, k] - C₂ * Fⁿ⁻¹.x[i, j, k]
        fy = C₁ * Fⁿ.y[i, j, k] - C₂ * Fⁿ⁻¹.y[i, j, k]
        fz = C₁ * Fⁿ.z[i, j, k] - C₂ * Fⁿ⁻¹.z[i, j, k]

        u = C₁ * Uⁿ.u[i, j, k] - C₂ * Uⁿ⁻¹.u[i, j, k]
        v = C₁ * Uⁿ.v[i, j, k] - C₂ * Uⁿ⁻¹.v[i, j, k]
        w = C₁ * Uⁿ.w[i, j, k] - C₂ * Uⁿ⁻¹.w[i, j, k]

        P.x[i, j, k] = 2 * δˣc★ * fx - δˣc² * u
        P.y[i, j, k] = 2 * δʸc★ * fy - δʸc² * v
        P.z[i, j, k] = 2 * δᶻc★ * fz - δᶻc² * w
    end
end

#####
##### Fluxes and transports, area-weighted, on the grid the stage computed its tendencies on.
##### The vertical face area does not stretch; the horizontal ones are taken back to the stretching `σᵐ⁻¹`.
#####

@kernel function _accumulate_advective_fluxes!(F, U★, grid, advection, U, tracer, σᵐ⁻¹, weight, keep)
    i, j, k = @index(Global, NTuple)

    rˣ = @inbounds σᵐ⁻¹.x[i, j, 1] / σⁿ(i, j, k, grid, f, c, c)
    rʸ = @inbounds σᵐ⁻¹.y[i, j, 1] / σⁿ(i, j, k, grid, c, f, c)

    fx = _advective_tracer_flux_x(i, j, k, grid, advection, U.u, tracer) * rˣ
    fy = _advective_tracer_flux_y(i, j, k, grid, advection, U.v, tracer) * rʸ
    fz = _advective_tracer_flux_z(i, j, k, grid, advection, U.w, tracer)

    @inbounds begin
        F.x[i, j, k] = keep * F.x[i, j, k] + weight * fx
        F.y[i, j, k] = keep * F.y[i, j, k] + weight * fy
        F.z[i, j, k] = keep * F.z[i, j, k] + weight * fz

        U★.u[i, j, k] = keep * U★.u[i, j, k] + weight * U.u[i, j, k] * Axᶠᶜᶜ(i, j, k, grid) * rˣ
        U★.v[i, j, k] = keep * U★.v[i, j, k] + weight * U.v[i, j, k] * Ayᶜᶠᶜ(i, j, k, grid) * rʸ
        U★.w[i, j, k] = keep * U★.w[i, j, k] + weight * U.w[i, j, k] * Azᶜᶜᶠ(i, j, k, grid)
    end
end
