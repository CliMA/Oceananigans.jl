
@kenrel function _update_diffusive_tracer_fluxes!(Vⁿ, Vⁿ⁻¹, grid, closure, diffusivity, bouyancy, c, tracer_id, clk, model_fields) 
    i, j, k = @index(Global, NTuple)
    compute_diffusive_tracer_fluxes!(Vⁿ, Vⁿ⁻¹, i, j, k, grid, closure, diffusivity, bouyancy, c, tracer_id, clk, model_fields)
end

@inline function compute_diffusive_tracer_fluxes!(Vⁿ, Vⁿ⁻¹, i, j, k, grid, closure::Tuple, K, args...) 
    for n in eachindex(closure)
        compute_diffusive_tracer_fluxes!(Vⁿ[n], Vⁿ⁻¹[n], i, j, k, grid, closure[n], K[n], args...)
    end
end

@inline function compute_diffusive_tracer_fluxes!(Vⁿ, Vⁿ⁻¹, i, j, k, grid, clo, K, b, c, c_id, clk, fields)
    Vⁿ⁻¹.x[i, j, k] = Vⁿ.x[i, j, k]
    Vⁿ⁻¹.y[i, j, k] = Vⁿ.y[i, j, k]
    Vⁿ⁻¹.z[i, j, k] = Vⁿ.z[i, j, k]

    Vⁿ.x[i, j, k] = _diffusive_tracer_flux_x(i, j, k, grid, clo, K, Val(c_id), c, clk, fields, b) * Axᶠᶜᶜ(i, j, k, grid)
    Vⁿ.y[i, j, k] = _diffusive_tracer_flux_y(i, j, k, grid, clo, K, Val(c_id), c, clk, fields, b) * Ayᶜᶠᶜ(i, j, k, grid)
    Vⁿ.z[i, j, k] = _diffusive_tracer_flux_z(i, j, k, grid, clo, K, Val(c_id), c, clk, fields, b) * Azᶜᶜᶠ(i, j, k, grid)
end
