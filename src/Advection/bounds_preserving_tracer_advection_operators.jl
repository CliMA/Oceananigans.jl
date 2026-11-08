using Oceananigans.Grids: AbstractGrid
using Oceananigans.Fields: CenterField
using Oceananigans.BoundaryConditions: fill_bottom_and_top_halo!
using Oceananigans.Utils: launch!
using KernelAbstractions: @index, @kernel

"""
    BoundsPreservation(minimum_value, maximum_value; maximum_cfl=5//18, flux_limiter=nothing)

The interval a tracer is restricted to and the options of the limiters that keep it there, for the `bounds` of
[`WENO`](@ref): the maximum-principle-satisfying rescaling of [Zhang and Shu (2010)](@cite ZhangShu2010), which
guarantees the interval for a forward Euler or strong-stability-preserving update whose Courant numbers satisfy
`λˣ + λʸ + λᶻ ≤ maximum_cfl`, and optionally a [`FluxLimiter`](@ref) that limits the fluxes between cells so
that the tracer stays within the interval for any time step.

Keyword arguments
=================
- `maximum_cfl`: The Courant number `ω̂₁ ∈ (0, 1/2)` up to which the rescaling alone guarantees the interval.
                 Raising it admits a larger time step and rescales more aggressively. With a `flux_limiter` the
                 interval is guaranteed for any time step, and `maximum_cfl` only sets how hard the rescaling
                 clips the reconstructions. Default: `5//18`.
- `flux_limiter`: `nothing` or `false` for none, `true` for the default `FluxLimiter()`, or a `FluxLimiter` such
                  as `FluxLimiter(credit=ColumnSweep())`. Default: `nothing`.

The materialized struct also holds the field `reconstruction_limiter` with the cell-wise rescaling factor ``θ``,
which is `nothing` until the scheme is materialized on a grid, like the fields of `flux_limiter`.
"""
struct BoundsPreservation{FT, L, F}
    minimum_value :: FT
    maximum_value :: FT
    maximum_cfl :: FT
    reconstruction_limiter :: L
    flux_limiter :: F
end

function BoundsPreservation(minimum_value, maximum_value; maximum_cfl=5//18, flux_limiter=nothing)
    minimum_value < maximum_value || throw(ArgumentError("minimum_value must be below maximum_value, found ($minimum_value, $maximum_value)"))
    0 < maximum_cfl < 1//2 || throw(ArgumentError("maximum_cfl must lie in (0, 1/2), found $maximum_cfl"))

    flux_limiter isa Union{Bool, Nothing, FluxLimiter} || throw(ArgumentError("flux_limiter must be a Bool, nothing or a FluxLimiter, found $flux_limiter"))
    flux_limiter = flux_limiter === true ? FluxLimiter() : flux_limiter === false ? nothing : flux_limiter

    minimum_value, maximum_value, maximum_cfl = promote(minimum_value, maximum_value, maximum_cfl)
    return BoundsPreservation(minimum_value, maximum_value, maximum_cfl, nothing, flux_limiter)
end

# The bounds with their numbers converted to the float type of a scheme
with_float_type(FT, ::Nothing) = nothing
with_float_type(FT, bounds::BoundsPreservation) =
    BoundsPreservation(convert(FT, bounds.minimum_value), convert(FT, bounds.maximum_value),
                       convert(FT, bounds.maximum_cfl), bounds.reconstruction_limiter, bounds.flux_limiter)

Base.show(io::IO, bounds::BoundsPreservation) = print(io, "(", bounds.minimum_value, ", ", bounds.maximum_value, ")")

Adapt.adapt_structure(to, bounds::BoundsPreservation) =
    BoundsPreservation(Adapt.adapt(to, bounds.minimum_value),
                       Adapt.adapt(to, bounds.maximum_value),
                       Adapt.adapt(to, bounds.maximum_cfl),
                       Adapt.adapt(to, bounds.reconstruction_limiter),
                       Adapt.adapt(to, bounds.flux_limiter))

Architectures.on_architecture(to, bounds::BoundsPreservation) =
    BoundsPreservation(on_architecture(to, bounds.minimum_value),
                       on_architecture(to, bounds.maximum_value),
                       on_architecture(to, bounds.maximum_cfl),
                       on_architecture(to, bounds.reconstruction_limiter),
                       on_architecture(to, bounds.flux_limiter))

const _ε₂ = 1e-20

# Note: this can probably be generalized to include UpwindBiased
const BoundsPreservingWENO = WENO{<:Any, <:Any, <:Any, <:Any, <:BoundsPreservation}

# The limiter is also computed in the first horizontal halo layer, whose reconstructions reach one cell further
@inline Grids.required_halo_size_x(::WENO{N, <:Any, <:Any, <:Any, <:BoundsPreservation}) where N = N + 1
@inline Grids.required_halo_size_y(::WENO{N, <:Any, <:Any, <:Any, <:BoundsPreservation}) where N = N + 1

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
    ω̂₁ = scheme.bounds.maximum_cfl

    cᵢ = @inbounds c[i, j, k]

    m, M = cᵢ, cᵢ
    m, M = reconstruction_extrema_x(i, j, k, grid, scheme, c, m, M, ω̂₁)
    m, M = reconstruction_extrema_y(i, j, k, grid, scheme, c, m, M, ω̂₁)
    m, M = reconstruction_extrema_z(i, j, k, grid, scheme, c, m, M, ω̂₁)

    θᵐᵃˣ = abs((cᵐᵃˣ - cᵢ) / (M - cᵢ + ε₂))
    θᵐⁱⁿ = abs((cᵐⁱⁿ - cᵢ) / (m - cᵢ - ε₂))

    return min(θᵐᵃˣ, θᵐⁱⁿ, one(FT))
end

@kernel function _compute_bounds_preserving_limiter!(θ, grid, scheme, c)
    i, j, k = @index(Global, NTuple)
    @inbounds θ[i, j, k] = bounds_preserving_limiter(i, j, k, grid, scheme, c)
end

update_bounds_preserving_limiter!(scheme, grid, c) = nothing

update_bounds_preserving_limiter!(scheme::FluxFormAdvection, grid, c) = update_bounds_preserving_limiter!(scheme.z, grid, c)

"""
$(TYPEDSIGNATURES)

Fill the rescaling factor ``θ`` carried by a bounds-preserving `scheme` from the tracer `c`.
``θ`` is computed in the first horizontal halo layer too, from the halos of `c`, so that
only its bottom and top halos are filled afterwards and no halo is communicated.
"""
function update_bounds_preserving_limiter!(scheme::BoundsPreservingWENO, grid, c)
    isnothing(c) && return nothing # the `momentum` entry has no tracer to limit
    θ = scheme.bounds.reconstruction_limiter
    parameters = horizontally_extended_interior_kernel_parameters(grid)
    launch!(architecture(grid), grid, parameters, _compute_bounds_preserving_limiter!, θ, grid, scheme, c)
    fill_bottom_and_top_halo!(θ.data, θ.boundary_conditions, θ.indices, (Center(), Center(), Center()), grid)
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
##### Bounded face fluxes and their divergence
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

    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (div_x + div_y + div_z)
end

# Support for Flat directions
@inline bounded_face_flux_x(i, j, k, ::AbstractGrid{FT, Flat, TY, TZ}, advection::BoundsPreservingWENO, args...) where {FT, TY, TZ} = zero(FT)
@inline bounded_face_flux_y(i, j, k, ::AbstractGrid{FT, TX, Flat, TZ}, advection::BoundsPreservingWENO, args...) where {FT, TX, TZ} = zero(FT)
@inline bounded_face_flux_z(i, j, k, ::AbstractGrid{FT, TX, TY, Flat}, advection::BoundsPreservingWENO, args...) where {FT, TX, TY} = zero(FT)

@inline bounded_tracer_flux_divergence_x(i, j, k, ::AbstractGrid{FT, Flat, TY, TZ}, advection::BoundsPreservingWENO, args...) where {FT, TY, TZ} = zero(FT)
@inline bounded_tracer_flux_divergence_y(i, j, k, ::AbstractGrid{FT, TX, Flat, TZ}, advection::BoundsPreservingWENO, args...) where {FT, TX, TZ} = zero(FT)
@inline bounded_tracer_flux_divergence_z(i, j, k, ::AbstractGrid{FT, TX, TY, Flat}, advection::BoundsPreservingWENO, args...) where {FT, TX, TY} = zero(FT)

# The advective flux of `c` through the face (i, j, k), weighted by the density `ρ` (`1` for `div_Uc`),
# built from the reconstructions rescaled by θ on either side of the face.
@inline function bounded_face_flux_x(i, j, k, grid, advection::BoundsPreservingWENO, ρ, u, c)
    θ = advection.bounds.reconstruction_limiter
    cᴸ = _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, advection, LeftBias,  c)
    cᴿ = _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, advection, RightBias, c)
    cᴸ = rescaled_reconstruction(cᴸ, i-1, j, k, grid, θ, c)
    cᴿ = rescaled_reconstruction(cᴿ, i,   j, k, grid, θ, c)
    ũ = @inbounds u[i, j, k]
    return ℑxᶠᵃᵃ(i, j, k, grid, ρ) * Axᶠᶜᶜ(i, j, k, grid) * upwind_biased_product(ũ, cᴸ, cᴿ)
end

@inline function bounded_face_flux_y(i, j, k, grid, advection::BoundsPreservingWENO, ρ, v, c)
    θ = advection.bounds.reconstruction_limiter
    cᴸ = _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, advection, LeftBias,  c)
    cᴿ = _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, advection, RightBias, c)
    cᴸ = rescaled_reconstruction(cᴸ, i, j-1, k, grid, θ, c)
    cᴿ = rescaled_reconstruction(cᴿ, i, j,   k, grid, θ, c)
    ṽ = @inbounds v[i, j, k]
    return ℑyᵃᶠᵃ(i, j, k, grid, ρ) * Ayᶜᶠᶜ(i, j, k, grid) * upwind_biased_product(ṽ, cᴸ, cᴿ)
end

@inline function bounded_face_flux_z(i, j, k, grid, advection::BoundsPreservingWENO, ρ, w, c)
    θ = advection.bounds.reconstruction_limiter
    cᴸ = _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, advection, LeftBias,  c)
    cᴿ = _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, advection, RightBias, c)
    cᴸ = rescaled_reconstruction(cᴸ, i, j, k-1, grid, θ, c)
    cᴿ = rescaled_reconstruction(cᴿ, i, j, k,   grid, θ, c)
    w̃ = @inbounds w[i, j, k]
    return ℑzᵃᵃᶠ(i, j, k, grid, ρ) * Azᶜᶜᶠ(i, j, k, grid) * upwind_biased_product(w̃, cᴸ, cᴿ)
end

@inline bounded_tracer_flux_divergence_x(i, j, k, grid, advection::BoundsPreservingWENO, ρ, u, c) =
    bounded_face_flux_x(i+1, j, k, grid, advection, ρ, u, c) - bounded_face_flux_x(i, j, k, grid, advection, ρ, u, c)

@inline bounded_tracer_flux_divergence_y(i, j, k, grid, advection::BoundsPreservingWENO, ρ, v, c) =
    bounded_face_flux_y(i, j+1, k, grid, advection, ρ, v, c) - bounded_face_flux_y(i, j, k, grid, advection, ρ, v, c)

@inline bounded_tracer_flux_divergence_z(i, j, k, grid, advection::BoundsPreservingWENO, ρ, w, c) =
    bounded_face_flux_z(i, j, k+1, grid, advection, ρ, w, c) - bounded_face_flux_z(i, j, k, grid, advection, ρ, w, c)
