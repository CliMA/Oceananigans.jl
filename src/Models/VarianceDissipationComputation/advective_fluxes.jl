@kernel function _update_advective_tracer_fluxes!(Gⁿ, Fⁿ, Fⁿ⁻¹, cⁿ⁻¹, grid, advection, U, c)
    i, j, k = @index(Global, NTuple)
    u, v, w = U

    @inbounds begin
        # Save previous advective fluxes
        Fⁿ⁻¹.x[i, j, k] = Fⁿ.x[i, j, k]
        Fⁿ⁻¹.y[i, j, k] = Fⁿ.y[i, j, k]
        Fⁿ⁻¹.z[i, j, k] = Fⁿ.z[i, j, k]
        
        cⁿ⁻¹[i, j, k] = c[i, j, k]

        # Calculate new advective fluxes
        Fⁿ.x[i, j, k] = _advective_tracer_flux_x(i, j, k, grid, advection, u, c) 
        Fⁿ.y[i, j, k] = _advective_tracer_flux_y(i, j, k, grid, advection, v, c) 
        Fⁿ.z[i, j, k] = _advective_tracer_flux_z(i, j, k, grid, advection, w, c) 
        
        Gⁿ.x[i, j, k] = Axᶠᶜᶜ(i, j, k, grid) * δxᶠᶜᶜ(i, j, k, grid, c)^2 / Δxᶠᶜᶜ(i, j, k, grid)
        Gⁿ.y[i, j, k] = Ayᶜᶠᶜ(i, j, k, grid) * δyᶜᶠᶜ(i, j, k, grid, c)^2 / Δyᶜᶠᶜ(i, j, k, grid)
        Gⁿ.z[i, j, k] = Azᶜᶜᶠ(i, j, k, grid) * δzᶜᶜᶠ(i, j, k, grid, c)^2 / Δzᶜᶜᶠ(i, j, k, grid)
    end
end

@kernel function _update_advective_vorticity_fluxes!(Gⁿ, Fⁿ, Fⁿ⁻¹, ζⁿ⁻¹, grid, advection, U, c)
    i, j, k = @index(Global, NTuple)
    u, v, w = U

    @inbounds begin
        # Save previous advective fluxes
        Fⁿ⁻¹.x[i, j, k] = Fⁿ.x[i, j, k]
        Fⁿ⁻¹.y[i, j, k] = Fⁿ.y[i, j, k]
        
        ζⁿ⁻¹[i, j, k] = ζ₃ᶠᶠᶜ(i, j, k, grid, U.u, U.v)

        # Calculate new advective fluxes
        Fⁿ.x[i, j, k] =   horizontal_advection_V(i, j, k, grid, advection, u, ζ) * Axᶜᶠᶜ(i, j, k, grid) 
        Fⁿ.y[i, j, k] = - horizontal_advection_U(i, j, k, grid, advection, v, ζ) * Ayᶠᶜᶜ(i, j, k, grid)

        Gⁿ.x[i, j, k] = Axᶜᶠᶜ(i, j, k, grid) * δxᶜᶠᶜ(i, j, k, grid, ζ₃ᶠᶠᶜ, U.u)^2 / Δxᶜᶠᶜ(i, j, k, grid)
        Gⁿ.y[i, j, k] = Ayᶠᶜᶜ(i, j, k, grid) * δyᶠᶜᶜ(i, j, k, grid, ζ₃ᶠᶠᶜ, U.v)^2 / Δyᶠᶜᶜ(i, j, k, grid)
    end
end