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
