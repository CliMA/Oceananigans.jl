using Oceananigans.TurbulenceClosures: is_vertically_implicit, ivd_diffusivity

# Vertically implicit diffusive flux, evaluated with cⁿ⁺¹ on the grid of the new time step
@inline function implicit_vertical_diffusive_flux(i, j, k, grid, κᶻ, cⁿ⁺¹)
    flux = @inbounds - κᶻ[i, j, k] * ∂zᶜᶜᶠ(i, j, k, grid, cⁿ⁺¹) * Azᶜᶜᶠ(i, j, k, grid)
    return flux * !peripheral_node(i, j, k, grid, c, c, f)
end

@kernel function _assemble_ab2_diffusive_dissipation!(K, grid, χ, Vⁿ, Vⁿ⁻¹, κᶻ, cⁿ⁺¹, cⁿ)
    i, j, k = @index(Global, NTuple)

    C₁  = convert(eltype(grid), 1.5 + χ)
    C₂  = convert(eltype(grid), 0.5 + χ)

    δˣc★ = δxᶠᶜᶜ(i, j, k, grid, c★, cⁿ⁺¹, cⁿ)
    δʸc★ = δyᶜᶠᶜ(i, j, k, grid, c★, cⁿ⁺¹, cⁿ)
    δᶻc★ = δzᶜᶜᶠ(i, j, k, grid, c★, cⁿ⁺¹, cⁿ)

    @inbounds begin
        vx₁ = C₁ * Vⁿ.x[i, j, k]
        vy₁ = C₁ * Vⁿ.y[i, j, k]
        vz₁ = C₁ * Vⁿ.z[i, j, k]

        vx₂ = C₂ * Vⁿ⁻¹.x[i, j, k]
        vy₂ = C₂ * Vⁿ⁻¹.y[i, j, k]
        vz₂ = C₂ * Vⁿ⁻¹.z[i, j, k]

        K.x[i, j, k] = 2 * δˣc★ * (vx₁ - vx₂)
        K.y[i, j, k] = 2 * δʸc★ * (vy₁ - vy₂)
        K.z[i, j, k] = 2 * δᶻc★ * (vz₁ - vz₂ + implicit_vertical_diffusive_flux(i, j, k, grid, κᶻ, cⁿ⁺¹))
    end
end

@kernel function _assemble_rk3_diffusive_dissipation!(K, grid, Vⁿ, κᶻ, cⁿ⁺¹, cⁿ)
    i, j, k = @index(Global, NTuple)

    δˣc★ = δxᶠᶜᶜ(i, j, k, grid, c★, cⁿ⁺¹, cⁿ)
    δʸc★ = δyᶜᶠᶜ(i, j, k, grid, c★, cⁿ⁺¹, cⁿ)
    δᶻc★ = δzᶜᶜᶠ(i, j, k, grid, c★, cⁿ⁺¹, cⁿ)

    @inbounds begin
        vx₁ = Vⁿ.x[i, j, k]
        vy₁ = Vⁿ.y[i, j, k]
        vz₁ = Vⁿ.z[i, j, k]

        K.x[i, j, k] = 2 * δˣc★ * vx₁
        K.y[i, j, k] = 2 * δʸc★ * vy₁
        K.z[i, j, k] = 2 * δᶻc★ * (vz₁ + implicit_vertical_diffusive_flux(i, j, k, grid, κᶻ, cⁿ⁺¹))
    end
end

# QAB2 Implementation
@kernel function _cache_diffusive_fluxes!(Vⁿ, Vⁿ⁻¹, κᶻ, grid::AbstractGrid, clo, K, b, c, c_id, clk, fields)
    i, j, k = @index(Global, NTuple)

    Vⁿ⁻¹.x[i, j, k] = Vⁿ.x[i, j, k]
    Vⁿ⁻¹.y[i, j, k] = Vⁿ.y[i, j, k]
    Vⁿ⁻¹.z[i, j, k] = Vⁿ.z[i, j, k]

    Vⁿ.x[i, j, k] = zero(grid)
    Vⁿ.y[i, j, k] = zero(grid)
    Vⁿ.z[i, j, k] = zero(grid)
    κᶻ[i, j, k]   = zero(grid)

    compute_diffusive_fluxes!(Vⁿ, κᶻ, i, j, k, grid, clo, K, b, c, c_id, clk, fields)
end

# RK3 Implementation for the last substep
@kernel function _cache_diffusive_fluxes!(Vⁿ, κᶻ, grid::AbstractGrid, clo, K, b, c, c_id, clk, fields)
    i, j, k = @index(Global, NTuple)

    Vⁿ.x[i, j, k] = zero(grid)
    Vⁿ.y[i, j, k] = zero(grid)
    Vⁿ.z[i, j, k] = zero(grid)
    κᶻ[i, j, k]   = zero(grid)

    compute_diffusive_fluxes!(Vⁿ, κᶻ, i, j, k, grid, clo, K, b, c, c_id, clk, fields)
end

# Deal with tuples of closures and closure_fields
@inline compute_diffusive_fluxes!(Vⁿ, κᶻ, i, j, k, grid, clo::Tuple{<:Any}, K, args...) =
    compute_diffusive_fluxes!(Vⁿ, κᶻ, i, j, k, grid, clo[1], K[1], args...)

@inline function compute_diffusive_fluxes!(Vⁿ, κᶻ, i, j, k, grid, clo::Tuple{<:Any, <:Any}, K, args...)
    compute_diffusive_fluxes!(Vⁿ, κᶻ, i, j, k, grid, clo[1], K[1], args...)
    compute_diffusive_fluxes!(Vⁿ, κᶻ, i, j, k, grid, clo[2], K[2], args...)
    return nothing
end

@inline function compute_diffusive_fluxes!(Vⁿ, κᶻ, i, j, k, grid, clo::Tuple, K, args...)
    compute_diffusive_fluxes!(Vⁿ, κᶻ, i, j, k, grid, clo[1], K[1], args...)
    compute_diffusive_fluxes!(Vⁿ, κᶻ, i, j, k, grid, clo[2], K[2], args...)
    compute_diffusive_fluxes!(Vⁿ, κᶻ, i, j, k, grid, clo[3:end], K[3:end], args...)
    return nothing
end

compute_diffusive_fluxes!(Vⁿ, κᶻ, i, j, k, grid, ::Nothing, K, b, c, c_id, clk, fields) = nothing

# Explicit fluxes with the closure's own time discretization, which leaves out the vertically implicit flux
@inline function compute_diffusive_fluxes!(Vⁿ, κᶻ, i, j, k, grid, clo, K, b, tracer, c_id, clk, fields)
    disc = time_discretization(clo)
    @inbounds begin
        Vⁿ.x[i, j, k] += _diffusive_flux_x(i, j, k, grid, disc, clo, K, c_id, tracer, clk, fields, b) * Axᶠᶜᶜ(i, j, k, grid)
        Vⁿ.y[i, j, k] += _diffusive_flux_y(i, j, k, grid, disc, clo, K, c_id, tracer, clk, fields, b) * Ayᶜᶠᶜ(i, j, k, grid)
        Vⁿ.z[i, j, k] += _diffusive_flux_z(i, j, k, grid, disc, clo, K, c_id, tracer, clk, fields, b) * Azᶜᶜᶠ(i, j, k, grid)
        κᶻ[i, j, k]   += ifelse(is_vertically_implicit(clo), ivd_diffusivity(i, j, k, grid, c, c, f, clo, K, c_id, clk, fields), zero(grid))
    end
    return nothing
end
