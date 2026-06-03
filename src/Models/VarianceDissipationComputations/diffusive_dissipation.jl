using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization

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

# QAB2 Implementation
@kernel function _cache_diffusive_fluxes!(Vⁿ, Vⁿ⁻¹, grid::AbstractGrid, clo, K, b, c, c_id, clk, fields)
    i, j, k = @index(Global, NTuple)

    Vⁿ⁻¹.x[i, j, k] = Vⁿ.x[i, j, k]
    Vⁿ⁻¹.y[i, j, k] = Vⁿ.y[i, j, k]
    Vⁿ⁻¹.z[i, j, k] = Vⁿ.z[i, j, k]

    Vⁿ.x[i, j, k] = zero(grid)
    Vⁿ.y[i, j, k] = zero(grid)
    Vⁿ.z[i, j, k] = zero(grid)

    compute_diffusive_fluxes!(Vⁿ, i, j, k, grid, clo, K, b, c, c_id, clk, fields)
end

# RK3 Implementation for the last substep
@kernel function _cache_diffusive_fluxes!(Vⁿ, grid::AbstractGrid, clo, K, b, c, c_id, clk, fields)
    i, j, k = @index(Global, NTuple)

    Vⁿ.x[i, j, k] = zero(grid)
    Vⁿ.y[i, j, k] = zero(grid)
    Vⁿ.z[i, j, k] = zero(grid)

    compute_diffusive_fluxes!(Vⁿ, i, j, k, grid, clo, K, b, c, c_id, clk, fields)
end

# Deal with tuples of closures and closure_fields
@inline compute_diffusive_fluxes!(Vⁿ, i, j, k, grid, clo::Tuple{<:Any}, K, args...) =
    compute_diffusive_fluxes!(Vⁿ, i, j, k, grid, clo[1], K[1], args...)

@inline function compute_diffusive_fluxes!(Vⁿ, i, j, k, grid, clo::Tuple{<:Any, <:Any}, K, args...)
    compute_diffusive_fluxes!(Vⁿ, i, j, k, grid, clo[1], K[1], args...)
    compute_diffusive_fluxes!(Vⁿ, i, j, k, grid, clo[2], K[2], args...)
    return nothing
end

@inline function compute_diffusive_fluxes!(Vⁿ, i, j, k, grid, clo::Tuple, K, args...)
    compute_diffusive_fluxes!(Vⁿ, i, j, k, grid, clo[1], K[1], args...)
    compute_diffusive_fluxes!(Vⁿ, i, j, k, grid, clo[2], K[2], args...)
    compute_diffusive_fluxes!(Vⁿ, i, j, k, grid, clo[3:end], K[3:end], args...)
    return nothing
end

compute_diffusive_fluxes!(Vⁿ, i, j, k, grid, ::Nothing, K, b, c, c_id, clk, fields) = nothing

compute_diffusive_fluxes!(Vⁿ, i, j, k, grid::SphericalShellGrid, ::Nothing, K, b, c, c_id, clk, fields) = nothing

const etd = ExplicitTimeDiscretization()

@inline function compute_diffusive_fluxes!(Vⁿ, i, j, k, grid, clo, K, b, tracer, c_id, clk, fields)
    @inbounds begin
        Vⁿ.x[i, j, k] += _diffusive_flux_x(i, j, k, grid, etd, clo, K, c_id, tracer, clk, fields, b) * Axᶠᶜᶜ(i, j, k, grid) * σⁿ(i, j, k, grid, f, c, c)
        Vⁿ.y[i, j, k] += _diffusive_flux_y(i, j, k, grid, etd, clo, K, c_id, tracer, clk, fields, b) * Ayᶜᶠᶜ(i, j, k, grid) * σⁿ(i, j, k, grid, c, f, c)
        Vⁿ.z[i, j, k] += _diffusive_flux_z(i, j, k, grid, etd, clo, K, c_id, tracer, clk, fields, b) * Azᶜᶜᶠ(i, j, k, grid) * σⁿ(i, j, k, grid, c, c, f)
    end
    return nothing
end

@inline function compute_diffusive_fluxes!(Vⁿ, i, j, k, grid::SphericalShellGrid, clo, K, b, tracer, c_id, clk, fields)
    transport_flux_x =
        transverse_computational_width_uᶠᶜᶜ(i, j, k, grid) *
        covariant_to_contravariant_flux_uᶠᶜᶜ(i, j, k, grid,
                                             _diffusive_flux_x(i, j, k, grid, etd, clo, K, c_id, tracer, clk, fields, b),
                                             _diffusive_flux_y(i, j, k, grid, etd, clo, K, c_id, tracer, clk, fields, b))

    transport_flux_y =
        transverse_computational_width_vᶜᶠᶜ(i, j, k, grid) *
        covariant_to_contravariant_flux_vᶜᶠᶜ(i, j, k, grid,
                                             _diffusive_flux_x(i, j, k, grid, etd, clo, K, c_id, tracer, clk, fields, b),
                                             _diffusive_flux_y(i, j, k, grid, etd, clo, K, c_id, tracer, clk, fields, b))

    @inbounds begin
        Vⁿ.x[i, j, k] += transport_flux_x * σⁿ(i, j, k, grid, f, c, c)
        Vⁿ.y[i, j, k] += transport_flux_y * σⁿ(i, j, k, grid, c, f, c)
        Vⁿ.z[i, j, k] += _diffusive_flux_z(i, j, k, grid, etd, clo, K, c_id, tracer, clk, fields, b) * Azᶜᶜᶠ(i, j, k, grid) * σⁿ(i, j, k, grid, c, c, f)
    end

    return nothing
end
