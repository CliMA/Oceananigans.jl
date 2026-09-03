using Oceananigans.Grids: AbstractGrid
using Oceananigans.Fields: CenterField
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Utils: launch!
using KernelAbstractions: @index, @kernel

"""
    BoundsPreservation(minimum_value, maximum_value, maximum_courant_number, limiter)

The interval a tracer is restricted to, the Courant number ``ω̂₁`` up to which the bound holds, and the field
`limiter` holding the cell-wise rescaling factor ``θ``. `limiter` is `nothing` until the scheme is materialized
on a grid.
"""
struct BoundsPreservation{FT, L}
    minimum_value :: FT
    maximum_value :: FT
    maximum_courant_number :: FT
    limiter :: L
end

Base.show(io::IO, bounds::BoundsPreservation) = print(io, "(", bounds.minimum_value, ", ", bounds.maximum_value, ")")

Adapt.adapt_structure(to, bounds::BoundsPreservation) =
    BoundsPreservation(Adapt.adapt(to, bounds.minimum_value),
                       Adapt.adapt(to, bounds.maximum_value),
                       Adapt.adapt(to, bounds.maximum_courant_number),
                       Adapt.adapt(to, bounds.limiter))

Architectures.on_architecture(to, bounds::BoundsPreservation) =
    BoundsPreservation(on_architecture(to, bounds.minimum_value),
                       on_architecture(to, bounds.maximum_value),
                       on_architecture(to, bounds.maximum_courant_number),
                       on_architecture(to, bounds.limiter))

const _ε₂ = 1e-20

# Note: this can probably be generalized to include UpwindBiased
const BoundsPreservingWENO = WENO{<:Any, <:Any, <:Any, <:Any, <:BoundsPreservation}

#####
##### Cell-wise rescaling factor, from cᵢ = ω̂₁ c⁻ + ω̂₁ c⁺ + (1 - 2ω̂₁) p̃ in each direction
#####

@inline function reconstruction_extrema_x(i, j, k, grid, scheme, c, m, M, ω̂₁)
    c⁻ = _biased_interpolate_xᶠᵃᵃ(i,   j, k, grid, scheme, RightBias, c)
    c⁺ = _biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, LeftBias,  c)
    cᵢ = @inbounds c[i, j, k]
    p̃  = (cᵢ - ω̂₁ * (c⁻ + c⁺)) / (1 - 2ω̂₁)

    return min(m, p̃, c⁻, c⁺), max(M, p̃, c⁻, c⁺)
end

@inline function reconstruction_extrema_y(i, j, k, grid, scheme, c, m, M, ω̂₁)
    c⁻ = _biased_interpolate_yᵃᶠᵃ(i, j,   k, grid, scheme, RightBias, c)
    c⁺ = _biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, LeftBias,  c)
    cᵢ = @inbounds c[i, j, k]
    p̃  = (cᵢ - ω̂₁ * (c⁻ + c⁺)) / (1 - 2ω̂₁)

    return min(m, p̃, c⁻, c⁺), max(M, p̃, c⁻, c⁺)
end

@inline function reconstruction_extrema_z(i, j, k, grid, scheme, c, m, M, ω̂₁)
    c⁻ = _biased_interpolate_zᵃᵃᶠ(i, j, k,   grid, scheme, RightBias, c)
    c⁺ = _biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, LeftBias,  c)
    cᵢ = @inbounds c[i, j, k]
    p̃  = (cᵢ - ω̂₁ * (c⁻ + c⁺)) / (1 - 2ω̂₁)

    return min(m, p̃, c⁻, c⁺), max(M, p̃, c⁻, c⁺)
end

# Support for Flat directions
@inline reconstruction_extrema_x(i, j, k, ::AbstractGrid{FT, Flat, TY, TZ}, scheme, c, m, M, ω̂₁) where {FT, TY, TZ} = (m, M)
@inline reconstruction_extrema_y(i, j, k, ::AbstractGrid{FT, TX, Flat, TZ}, scheme, c, m, M, ω̂₁) where {FT, TX, TZ} = (m, M)
@inline reconstruction_extrema_z(i, j, k, ::AbstractGrid{FT, TX, TY, Flat}, scheme, c, m, M, ω̂₁) where {FT, TX, TY} = (m, M)

@inline function bounds_preserving_limiter(i, j, k, grid, scheme::BoundsPreservingWENO, c)
    FT = eltype(c)
    ε₂ = convert(FT, _ε₂)

    cᵐⁱⁿ = scheme.bounds.minimum_value
    cᵐᵃˣ = scheme.bounds.maximum_value
    ω̂₁ = scheme.bounds.maximum_courant_number

    cᵢ = @inbounds c[i, j, k]

    m, M = cᵢ, cᵢ
    m, M = reconstruction_extrema_x(i, j, k, grid, scheme, c, m, M, ω̂₁)
    m, M = reconstruction_extrema_y(i, j, k, grid, scheme, c, m, M, ω̂₁)
    m, M = reconstruction_extrema_z(i, j, k, grid, scheme, c, m, M, ω̂₁)

    θᵐᵃˣ = abs((cᵐᵃˣ - cᵢ) / (M - cᵢ + ε₂))
    θᵐⁱⁿ = abs((cᵐⁱⁿ - cᵢ) / (m - cᵢ + ε₂))

    return min(θᵐᵃˣ, θᵐⁱⁿ, one(FT))
end

@kernel function _compute_bounds_preserving_limiter!(θ, grid, scheme, c)
    i, j, k = @index(Global, NTuple)
    @inbounds θ[i, j, k] = bounds_preserving_limiter(i, j, k, grid, scheme, c)
end

function update_advection!(scheme::BoundsPreservingWENO, model, tracer)
    isnothing(tracer) && return nothing # the `momentum` entry has no tracer to limit
    compute_bounds_preserving_limiter!(scheme, model.grid, tracer)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Fill the rescaling factor ``θ`` carried by a bounds-preserving `scheme` from the tracer `c`.
"""
function compute_bounds_preserving_limiter!(scheme, grid, c)
    θ = scheme.bounds.limiter
    launch!(architecture(grid), grid, :xyz, _compute_bounds_preserving_limiter!, θ, grid, scheme, c)
    fill_halo_regions!(θ)
    return nothing
end

@inline function rescaled_reconstruction(ĉ, i, j, k, grid, θ, c)
    cᵢ = @inbounds c[i, j, k]
    θᵢ = @inbounds θ[i, j, k]
    return θᵢ * (ĉ - cᵢ) + cᵢ
end

@inline without_bounds_preservation(scheme) = scheme

@inline function without_bounds_preservation(scheme::WENO{N, FT, WCT}) where {N, FT, WCT}
    return WENO{N, FT, WCT}(nothing, scheme.buffer_scheme, scheme.advecting_velocity_scheme, scheme.time_discretization)
end

#####
##### Flux divergence
#####

@inline div_Uc(i, j, k, grid, advection::BoundsPreservingWENO, U, ::ZeroField) = zero(grid)
@inline div_Uc(i, j, k, grid, advection::BoundsPreservingWENO, ::ZeroU, c) = zero(grid)
@inline div_Uc(i, j, k, grid, advection::BoundsPreservingWENO, ::ZeroU, ::ZeroField) = zero(grid)

# Is this immersed-boundary safe without having to extend it in ImmersedBoundaries.jl? I think so... (velocity on immmersed boundaries is masked to 0)
# For bounds preserving advection, we need fluxes at both cell-faces to compute the flux on one face.
# So we extend div_Uc in order to compute the fluxes at i and i+1 in one go and avoid recomputation.
@inline function div_Uc(i, j, k, grid, advection::BoundsPreservingWENO, U, c)
    div_x = bounded_tracer_flux_divergence_x(i, j, k, grid, advection, 1, U.u, c)
    div_y = bounded_tracer_flux_divergence_y(i, j, k, grid, advection, 1, U.v, c)
    div_z = bounded_tracer_flux_divergence_z(i, j, k, grid, advection, 1, U.w, c)

    return 1/Vᶜᶜᶜ(i, j, k, grid) * (div_x + div_y + div_z)
end

# Support for Flat directions
@inline bounded_tracer_flux_divergence_x(i, j, k, ::AbstractGrid{FT, Flat, TY, TZ}, advection::BoundsPreservingWENO, args...) where {FT, TY, TZ} = zero(FT)
@inline bounded_tracer_flux_divergence_y(i, j, k, ::AbstractGrid{FT, TX, Flat, TZ}, advection::BoundsPreservingWENO, args...) where {FT, TX, TZ} = zero(FT)
@inline bounded_tracer_flux_divergence_z(i, j, k, ::AbstractGrid{FT, TX, TY, Flat}, advection::BoundsPreservingWENO, args...) where {FT, TX, TY} = zero(FT)

@inline function bounded_tracer_flux_divergence_x(i, j, k, grid, advection::BoundsPreservingWENO, ρ, u, c)
    θ = advection.bounds.limiter

    c₊ᴸ = _biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, advection, LeftBias,  c)
    c₊ᴿ = _biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, advection, RightBias, c)
    c₋ᴸ = _biased_interpolate_xᶠᵃᵃ(i,   j, k, grid, advection, LeftBias,  c)
    c₋ᴿ = _biased_interpolate_xᶠᵃᵃ(i,   j, k, grid, advection, RightBias, c)

    c₊ᴸ = rescaled_reconstruction(c₊ᴸ, i,   j, k, grid, θ, c)
    c₊ᴿ = rescaled_reconstruction(c₊ᴿ, i+1, j, k, grid, θ, c)
    c₋ᴸ = rescaled_reconstruction(c₋ᴸ, i-1, j, k, grid, θ, c)
    c₋ᴿ = rescaled_reconstruction(c₋ᴿ, i,   j, k, grid, θ, c)

    u⁺ = @inbounds u[i+1, j, k]
    u⁻ = @inbounds u[i,   j, k]
    Ax_ρuc⁺ = ℑxᶠᵃᵃ(i+1, j, k, grid, ρ) * Axᶠᶜᶜ(i+1, j, k, grid) * upwind_biased_product(u⁺, c₊ᴸ, c₊ᴿ)
    Ax_ρuc⁻ = ℑxᶠᵃᵃ(i,   j, k, grid, ρ) * Axᶠᶜᶜ(i,   j, k, grid) * upwind_biased_product(u⁻, c₋ᴸ, c₋ᴿ)

    return Ax_ρuc⁺ - Ax_ρuc⁻
end

@inline function bounded_tracer_flux_divergence_y(i, j, k, grid, advection::BoundsPreservingWENO, ρ, v, c)
    θ = advection.bounds.limiter

    c₊ᴸ = _biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, advection, LeftBias,  c)
    c₊ᴿ = _biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, advection, RightBias, c)
    c₋ᴸ = _biased_interpolate_yᵃᶠᵃ(i, j,   k, grid, advection, LeftBias,  c)
    c₋ᴿ = _biased_interpolate_yᵃᶠᵃ(i, j,   k, grid, advection, RightBias, c)

    c₊ᴸ = rescaled_reconstruction(c₊ᴸ, i, j,   k, grid, θ, c)
    c₊ᴿ = rescaled_reconstruction(c₊ᴿ, i, j+1, k, grid, θ, c)
    c₋ᴸ = rescaled_reconstruction(c₋ᴸ, i, j-1, k, grid, θ, c)
    c₋ᴿ = rescaled_reconstruction(c₋ᴿ, i, j,   k, grid, θ, c)

    v⁺ = @inbounds v[i, j+1, k]
    v⁻ = @inbounds v[i, j,   k]
    Ay_ρvc⁺ = ℑyᵃᶠᵃ(i, j+1, k, grid, ρ) * Ayᶜᶠᶜ(i, j+1, k, grid) * upwind_biased_product(v⁺, c₊ᴸ, c₊ᴿ)
    Ay_ρvc⁻ = ℑyᵃᶠᵃ(i, j,   k, grid, ρ) * Ayᶜᶠᶜ(i, j,   k, grid) * upwind_biased_product(v⁻, c₋ᴸ, c₋ᴿ)

    return Ay_ρvc⁺ - Ay_ρvc⁻
end

@inline function bounded_tracer_flux_divergence_z(i, j, k, grid, advection::BoundsPreservingWENO, ρ, w, c)
    θ = advection.bounds.limiter

    c₊ᴸ = _biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, advection, LeftBias,  c)
    c₊ᴿ = _biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, advection, RightBias, c)
    c₋ᴸ = _biased_interpolate_zᵃᵃᶠ(i, j, k,   grid, advection, LeftBias,  c)
    c₋ᴿ = _biased_interpolate_zᵃᵃᶠ(i, j, k,   grid, advection, RightBias, c)

    c₊ᴸ = rescaled_reconstruction(c₊ᴸ, i, j, k,   grid, θ, c)
    c₊ᴿ = rescaled_reconstruction(c₊ᴿ, i, j, k+1, grid, θ, c)
    c₋ᴸ = rescaled_reconstruction(c₋ᴸ, i, j, k-1, grid, θ, c)
    c₋ᴿ = rescaled_reconstruction(c₋ᴿ, i, j, k,   grid, θ, c)

    w⁺ = @inbounds w[i, j, k+1]
    w⁻ = @inbounds w[i, j, k]
    Az_ρwc⁺ = ℑzᵃᵃᶠ(i, j, k+1, grid, ρ) * Azᶜᶜᶠ(i, j, k+1, grid) * upwind_biased_product(w⁺, c₊ᴸ, c₊ᴿ)
    Az_ρwc⁻ = ℑzᵃᵃᶠ(i, j, k,   grid, ρ) * Azᶜᶜᶠ(i, j, k,   grid) * upwind_biased_product(w⁻, c₋ᴸ, c₋ᴿ)

    return Az_ρwc⁺ - Az_ρwc⁻
end
