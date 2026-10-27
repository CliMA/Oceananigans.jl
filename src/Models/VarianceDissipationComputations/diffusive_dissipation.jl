using Oceananigans.TurbulenceClosures: is_vertically_implicit, ivd_diffusivity

# Vertically implicit diffusive flux, evaluated with cⁿ⁺¹ on the grid of the new time step
@inline function implicit_vertical_diffusive_flux(i, j, k, grid, κᶻ, cⁿ⁺¹)
    flux = @inbounds - κᶻ[i, j, k] * ∂zᶜᶜᶠ(i, j, k, grid, cⁿ⁺¹) * Azᶜᶜᶠ(i, j, k, grid)
    return flux * !peripheral_node(i, j, k, grid, c, c, f)
end

@kernel function _assemble_diffusive_dissipation!(K, grid, C₁, C₂, Vⁿ, Vⁿ⁻¹, κᶻ, cⁿ⁺¹, cⁿ)
    i, j, k = @index(Global, NTuple)

    δˣc★ = δxᶠᶜᶜ(i, j, k, grid, c★, cⁿ⁺¹, cⁿ)
    δʸc★ = δyᶜᶠᶜ(i, j, k, grid, c★, cⁿ⁺¹, cⁿ)
    δᶻc★ = δzᶜᶜᶠ(i, j, k, grid, c★, cⁿ⁺¹, cⁿ)

    @inbounds begin
        vx = C₁ * Vⁿ.x[i, j, k] - C₂ * Vⁿ⁻¹.x[i, j, k]
        vy = C₁ * Vⁿ.y[i, j, k] - C₂ * Vⁿ⁻¹.y[i, j, k]
        vz = C₁ * Vⁿ.z[i, j, k] - C₂ * Vⁿ⁻¹.z[i, j, k]

        K.x[i, j, k] = 2 * δˣc★ * vx
        K.y[i, j, k] = 2 * δʸc★ * vy
        K.z[i, j, k] = 2 * δᶻc★ * (vz + implicit_vertical_diffusive_flux(i, j, k, grid, κᶻ, cⁿ⁺¹))
    end
end

# `V` and `κᶻ` hold the running `Σₘ wₘ Vᵐ` and `Σₘ wₘ κᵐ` of the stages that enter the step
@kernel function _accumulate_diffusive_fluxes!(V, κᶻ, grid::AbstractGrid, clo, K, b, c, c_id, clk, fields, weight, keep)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        V.x[i, j, k] = keep * V.x[i, j, k]
        V.y[i, j, k] = keep * V.y[i, j, k]
        V.z[i, j, k] = keep * V.z[i, j, k]
        κᶻ[i, j, k]  = keep * κᶻ[i, j, k]
    end

    compute_diffusive_fluxes!(V, κᶻ, weight, i, j, k, grid, clo, K, b, c, c_id, clk, fields)
end

# Deal with tuples of closures and closure_fields
@inline compute_diffusive_fluxes!(V, κᶻ, w, i, j, k, grid, clo::Tuple{<:Any}, K, args...) =
    compute_diffusive_fluxes!(V, κᶻ, w, i, j, k, grid, clo[1], K[1], args...)

@inline function compute_diffusive_fluxes!(V, κᶻ, w, i, j, k, grid, clo::Tuple{<:Any, <:Any}, K, args...)
    compute_diffusive_fluxes!(V, κᶻ, w, i, j, k, grid, clo[1], K[1], args...)
    compute_diffusive_fluxes!(V, κᶻ, w, i, j, k, grid, clo[2], K[2], args...)
    return nothing
end

@inline function compute_diffusive_fluxes!(V, κᶻ, w, i, j, k, grid, clo::Tuple, K, args...)
    compute_diffusive_fluxes!(V, κᶻ, w, i, j, k, grid, clo[1], K[1], args...)
    compute_diffusive_fluxes!(V, κᶻ, w, i, j, k, grid, clo[2], K[2], args...)
    compute_diffusive_fluxes!(V, κᶻ, w, i, j, k, grid, clo[3:end], K[3:end], args...)
    return nothing
end

compute_diffusive_fluxes!(V, κᶻ, w, i, j, k, grid, ::Nothing, K, b, c, c_id, clk, fields) = nothing

# Explicit fluxes with the closure's own time discretization, which leaves out the vertically implicit flux
@inline function compute_diffusive_fluxes!(V, κᶻ, w, i, j, k, grid, clo, K, b, tracer, c_id, clk, fields)
    disc = time_discretization(clo)
    @inbounds begin
        V.x[i, j, k] += w * _diffusive_flux_x(i, j, k, grid, disc, clo, K, c_id, tracer, clk, fields, b) * Axᶠᶜᶜ(i, j, k, grid)
        V.y[i, j, k] += w * _diffusive_flux_y(i, j, k, grid, disc, clo, K, c_id, tracer, clk, fields, b) * Ayᶜᶠᶜ(i, j, k, grid)
        V.z[i, j, k] += w * _diffusive_flux_z(i, j, k, grid, disc, clo, K, c_id, tracer, clk, fields, b) * Azᶜᶜᶠ(i, j, k, grid)
        κᶻ[i, j, k]  += w * ifelse(is_vertically_implicit(clo), ivd_diffusivity(i, j, k, grid, c, c, f, clo, K, c_id, clk, fields), zero(grid))
    end
    return nothing
end
