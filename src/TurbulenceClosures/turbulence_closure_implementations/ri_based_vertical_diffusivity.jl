using Oceananigans.Architectures: architecture
using Oceananigans.BuoyancyModels: ∂z_b
using Oceananigans.Operators
using Oceananigans.Grids: inactive_node
using Oceananigans.Operators: ℑzᵃᵃᶜ

import Oceananigans.Architectures: on_architecture

struct RiBasedVerticalDiffusivity{TD, FT, R, HR} <: AbstractScalarDiffusivity{TD, VerticalFormulation, 1}
    ν₀  :: FT
    κ₀  :: FT
    κᶜᵃ :: FT
    Cᵉⁿ :: FT
    Cᵃᵛ :: FT
    Ri₀ :: FT
    Riᵟ :: FT
    Ri_dependent_tapering :: R
    horizontal_Ri_filter :: HR
    minimum_entrainment_buoyancy_gradient :: FT
    maximum_diffusivity :: FT
    maximum_viscosity :: FT
end

function RiBasedVerticalDiffusivity{TD}(ν₀::FT,
                                        κ₀::FT,
                                        κᶜᵃ::FT,
                                        Cᵉⁿ::FT,
                                        Cᵃᵛ::FT,
                                        Ri₀::FT,
                                        Riᵟ::FT,
                                        Ri_dependent_tapering::R,
                                        horizontal_Ri_filter::HR,
                                        minimum_entrainment_buoyancy_gradient::FT,
                                        maximum_diffusivity::FT,
                                        maximum_viscosity::FT) where {TD, FT, R, HR}
                                       

    return RiBasedVerticalDiffusivity{TD, FT, R, HR}(ν₀, κ₀, κᶜᵃ, Cᵉⁿ, Cᵃᵛ, Ri₀, Riᵟ,
                                                     Ri_dependent_tapering,
                                                     horizontal_Ri_filter,
                                                     minimum_entrainment_buoyancy_gradient,
                                                     maximum_diffusivity,
                                                     maximum_viscosity)
end

# Ri-dependent tapering flavor
struct PiecewiseLinearRiDependentTapering end
struct ExponentialRiDependentTapering end
struct HyperbolicTangentRiDependentTapering end

Base.summary(::HyperbolicTangentRiDependentTapering) = "HyperbolicTangentRiDependentTapering" 
Base.summary(::ExponentialRiDependentTapering) = "ExponentialRiDependentTapering" 
Base.summary(::PiecewiseLinearRiDependentTapering) = "PiecewiseLinearRiDependentTapering" 

# Horizontal filtering for the Richardson number
struct FivePointHorizontalFilter end
@inline filter_horizontally(i, j, k, grid, ::Nothing, ϕ) = @inbounds ϕ[i, j, k]
@inline filter_horizontally(i, j, k, grid, ::FivePointHorizontalFilter, ϕ) = ℑxyᶜᶜᵃ(i, j, k, grid, ℑxyᶠᶠᵃ, ϕ)

"""
    RiBasedVerticalDiffusivity([time_discretization = VerticallyImplicitTimeDiscretization(),
                               FT = Float64;]
                               Ri_dependent_tapering = HyperbolicTangentRiDependentTapering(),
                               horizontal_Ri_filter = nothing,
                               minimum_entrainment_buoyancy_gradient = 1e-10,
                               maximum_diffusivity = Inf,
                               maximum_viscosity = Inf,
                               ν₀  = 0.7,
                               κ₀  = 0.5,
                               κᶜᵃ = 1.7,
                               Cᵉⁿ = 0.1,
                               Cᵃᵛ = 0.6,
                               Ri₀ = 0.1,
                               Riᵟ = 0.4,
                               warning = true)

Return a closure that estimates the vertical viscosity and diffusivity
from "convective adjustment" coefficients `ν₀` and `κ₀` multiplied by
a decreasing function of the Richardson number, ``Ri``. 

Arguments
=========

* `time_discretization`: Either `ExplicitTimeDiscretization()` or `VerticallyImplicitTimeDiscretization()`, 
                         which integrates the terms involving only ``z``-derivatives in the
                         viscous and diffusive fluxes with an implicit time discretization.
                         Default `VerticallyImplicitTimeDiscretization()`.

* `FT`: Float type; default `Float64`.

Keyword arguments
=================

* `Ri_dependent_tapering`: The ``Ri``-dependent tapering.
  Options are: `PiecewiseLinearRiDependentTapering()`,
  `HyperbolicTangentRiDependentTapering()` (default), and
  `ExponentialRiDependentTapering()`.

* `ν₀`: Non-convective viscosity (units of kinematic viscosity, typically m² s⁻¹).

* `κ₀`: Non-convective diffusivity for tracers (units of diffusivity, typically m² s⁻¹).

* `κᶜᵃ`: Convective adjustment diffusivity for tracers (units of diffusivity, typically m² s⁻¹).

* `Cᵉⁿ`: Entrainment coefficient for tracers (non-dimensional).
         Set `Cᵉⁿ = 0` to turn off the penetrative entrainment diffusivity.

* `Cᵃᵛ`: Time-averaging coefficient for viscosity and diffusivity (non-dimensional).

* `Ri₀`: ``Ri`` threshold for decreasing viscosity and diffusivity (non-dimensional).

* `Riᵟ`: ``Ri``-width over which viscosity and diffusivity decreases to 0 (non-dimensional).

* `minimum_entrainment_buoyancy_gradient`: Minimum buoyancy gradient for application of the entrainment
                                           diffusvity. If the entrainment buoyancy gradient is less than the
                                           minimum value, the entrainment diffusivity is 0. Units of 
                                           buoyancy gradient (typically s⁻²).

* `maximum_diffusivity`: A limiting maximum tracer diffusivity (units of diffusivity, typically m² s⁻¹).

* `maximum_viscosity`: A limiting maximum viscosity (units of kinematic viscosity, typically m² s⁻¹).

* `horizontal_Ri_filter`: Horizontal filter to apply to Ri, which can help alleviate noise for
                          some simulations. The default is `nothing`, or no filtering. The other
                          option is `horizontal_Ri_filter = FivePointHorizontalFilter()`.
"""
function RiBasedVerticalDiffusivity(time_discretization = VerticallyImplicitTimeDiscretization(),
                                    FT = Float64;
                                    Ri_dependent_tapering = HyperbolicTangentRiDependentTapering(),
                                    horizontal_Ri_filter = nothing,
                                    minimum_entrainment_buoyancy_gradient = 1e-10,
                                    maximum_diffusivity = Inf,
                                    maximum_viscosity = Inf,
                                    ν₀  = 0.7,
                                    κ₀  = 0.5,
                                    κᶜᵃ = 1.7,
                                    Cᵉⁿ = 0.1,
                                    Cᵃᵛ = 0.6,
                                    Ri₀ = 0.1,
                                    Riᵟ = 0.4,
                                    warning = true)
    if warning
        @warn "RiBasedVerticalDiffusivity is an experimental turbulence closure that \n" *
              "is unvalidated and whose default parameters are not calibrated for \n" * 
              "realistic ocean conditions or for use in a three-dimensional \n" *
              "simulation. Use with caution and report bugs and problems with physics \n" *
              "to https://github.com/CliMA/Oceananigans.jl/issues."
    end

    TD = typeof(time_discretization)

    return RiBasedVerticalDiffusivity{TD}(convert(FT, ν₀),
                                          convert(FT, κ₀),
                                          convert(FT, κᶜᵃ),
                                          convert(FT, Cᵉⁿ),
                                          convert(FT, Cᵃᵛ),
                                          convert(FT, Ri₀),
                                          convert(FT, Riᵟ),
                                          Ri_dependent_tapering,
                                          horizontal_Ri_filter,
                                          convert(FT, minimum_entrainment_buoyancy_gradient),
                                          convert(FT, maximum_diffusivity),
                                          convert(FT, maximum_viscosity))
end

RiBasedVerticalDiffusivity(FT::DataType; kw...) =
    RiBasedVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kw...)

#####
##### Diffusivity field utilities
#####

const RBVD = RiBasedVerticalDiffusivity
const RBVDArray = AbstractArray{<:RBVD}
const FlavorOfRBVD = Union{RBVD, RBVDArray}
const c = Center()
const f = Face()

@inline viscosity_location(::FlavorOfRBVD)   = (c, c, f)
@inline diffusivity_location(::FlavorOfRBVD) = (c, c, f)

@inline viscosity(::FlavorOfRBVD, diffusivities) = diffusivities.κᵘ
@inline diffusivity(::FlavorOfRBVD, diffusivities, id) = diffusivities.κᶜ

with_tracers(tracers, closure::FlavorOfRBVD) = closure

# Note: computing diffusivities at cell centers for now.
function DiffusivityFields(grid, tracer_names, bcs, closure::FlavorOfRBVD)
    κᶜ = Field((Center, Center, Face), grid)
    κᵘ = Field((Center, Center, Face), grid)
    Ri = Field((Center, Center, Face), grid)
    return (; κᶜ, κᵘ, Ri)
end

function compute_diffusivities!(diffusivities, closure::FlavorOfRBVD, model; parameters = :xyz)
    arch = model.architecture
    grid = model.grid
    clock = model.clock
    tracers = model.tracers
    buoyancy = model.buoyancy
    velocities = model.velocities
    top_tracer_bcs = NamedTuple(c => tracers[c].boundary_conditions.top for c in propertynames(tracers))

    launch!(arch, grid, parameters,
            compute_ri_number!,
            diffusivities,
            grid,
            closure,
            velocities,
            tracers,
            buoyancy,
            top_tracer_bcs,
            clock)

    # Use `only_local_halos` to ensure that no communication occurs during
    # this call to fill_halo_regions!
    fill_halo_regions!(diffusivities.Ri; only_local_halos=true)

    launch!(arch, grid, parameters,
            compute_ri_based_diffusivities!,
            diffusivities,
            grid,
            closure,
            velocities,
            tracers,
            buoyancy,
            top_tracer_bcs,
            clock)

    return nothing
end

# 1. x < x₀     => taper = 1
# 2. x > x₀ + δ => taper = 0
# 3. Otherwise, vary linearly between 1 and 0

const Linear = PiecewiseLinearRiDependentTapering
const Exp    = ExponentialRiDependentTapering
const Tanh   = HyperbolicTangentRiDependentTapering

@inline taper(::Linear, x::T, x₀, δ) where T = one(T) - min(one(T), max(zero(T), (x - x₀) / δ))
@inline taper(::Exp,    x::T, x₀, δ) where T = exp(- max(zero(T), (x - x₀) / δ))
@inline taper(::Tanh,   x::T, x₀, δ) where T = (one(T) - tanh((x - x₀) / δ)) / 2

@inline ϕ²(i, j, k, grid, ϕ, args...) = ϕ(i, j, k, grid, args...)^2

@inline function shear_squaredᶜᶜᶠ(i, j, k, grid, velocities)
    ∂z_u² = ℑxᶜᵃᵃ(i, j, k, grid, ϕ², ∂zᶠᶜᶠ, velocities.u)
    ∂z_v² = ℑyᵃᶜᵃ(i, j, k, grid, ϕ², ∂zᶜᶠᶠ, velocities.v)
    return ∂z_u² + ∂z_v²
end

@inline function Riᶜᶜᶠ(i, j, k, grid, velocities, buoyancy, tracers)
    S² = shear_squaredᶜᶜᶠ(i, j, k, grid, velocities)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    Ri = N² / S²

    # Clip N² and avoid NaN
    return ifelse(N² <= 0, zero(grid), Ri)
end

const c = Center()
const f = Face()

@kernel function compute_ri_number!(diffusivities, grid, closure::FlavorOfRBVD,
                                    velocities, tracers, buoyancy, tracer_bcs, clock)
    i, j, k = @index(Global, NTuple)
    @inbounds diffusivities.Ri[i, j, k] = Riᶜᶜᶠ(i, j, k, grid, velocities, buoyancy, tracers)
end

@kernel function compute_ri_based_diffusivities!(diffusivities, grid, closure::FlavorOfRBVD,
                                                velocities, tracers, buoyancy, tracer_bcs, clock)
    i, j, k = @index(Global, NTuple)
    _compute_ri_based_diffusivities!(i, j, k, diffusivities, grid, closure,
                                     velocities, tracers, buoyancy, tracer_bcs, clock)
end


@inline function _compute_ri_based_diffusivities!(i, j, k, diffusivities, grid, closure,
                                                  velocities, tracers, buoyancy, tracer_bcs, clock)

    # Ensure this works with "ensembles" of closures, in addition to ordinary single closures
    closure_ij = getclosure(i, j, closure)

    ν₀  = closure_ij.ν₀
    κ₀  = closure_ij.κ₀
    κᶜᵃ = closure_ij.κᶜᵃ
    Cᵉⁿ = closure_ij.Cᵉⁿ
    Cᵃᵛ = closure_ij.Cᵃᵛ
    Ri₀ = closure_ij.Ri₀
    Riᵟ = closure_ij.Riᵟ
    tapering = closure_ij.Ri_dependent_tapering
    Ri_filter = closure_ij.horizontal_Ri_filter
    N²ᵉⁿ = closure_ij.minimum_entrainment_buoyancy_gradient
    Qᵇ = top_buoyancy_flux(i, j, grid, buoyancy, tracer_bcs, clock, merge(velocities, tracers))

    # Convection and entrainment
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    N²_above = ∂z_b(i, j, k+1, grid, buoyancy, tracers)

    # Conditions
    # TODO: apply a minimum entrainment buoyancy gradient?
    convecting = N² < 0 # applies regardless of Qᵇ
    entraining = (N² > N²ᵉⁿ) & (N²_above < 0) & (Qᵇ > 0)

    # Convective adjustment diffusivity
    κᶜᵃ = ifelse(convecting, κᶜᵃ, zero(grid))

    # Entrainment diffusivity
    κᵉⁿ = ifelse(entraining, Cᵉⁿ * Qᵇ / N², zero(grid))

    # (Potentially) apply a horizontal filter to the Richardson number
    Ri = filter_horizontally(i, j, k, grid, Ri_filter, diffusivities.Ri)

    # Shear mixing diffusivity and viscosity
    τ = taper(tapering, Ri, Ri₀, Riᵟ)
    κᶜ★ = κ₀ * τ
    κᵘ★ = ν₀ * τ

    # Previous diffusivities
    κᶜ = diffusivities.κᶜ
    κᵘ = diffusivities.κᵘ

    # New diffusivities
    κᶜ⁺ = κᶜᵃ + κᵉⁿ + κᶜ★
    κᵘ⁺ = κᵘ★

    # Limit by specified maximum
    κᶜ⁺ = min(κᶜ⁺, closure_ij.maximum_diffusivity) 
    κᵘ⁺ = min(κᵘ⁺, closure_ij.maximum_viscosity) 

    # Set to zero on periphery and NaN within inactive region
    on_periphery = peripheral_node(i, j, k, grid, c, c, f)
    within_inactive = inactive_node(i, j, k, grid, c, c, f)
    κᶜ⁺ = ifelse(on_periphery, zero(grid), ifelse(within_inactive, NaN, κᶜ⁺))
    κᵘ⁺ = ifelse(on_periphery, zero(grid), ifelse(within_inactive, NaN, κᵘ⁺))

    # Update by averaging in time
    @inbounds κᶜ[i, j, k] = (Cᵃᵛ * κᶜ[i, j, k] + κᶜ⁺) / (1 + Cᵃᵛ)
    @inbounds κᵘ[i, j, k] = (Cᵃᵛ * κᵘ[i, j, k] + κᵘ⁺) / (1 + Cᵃᵛ)

    return nothing
end

#####
##### Show
#####

Base.summary(closure::RiBasedVerticalDiffusivity{TD}) where TD = string("RiBasedVerticalDiffusivity{$TD}")

function Base.show(io::IO, closure::RiBasedVerticalDiffusivity)
    print(io, summary(closure), '\n')
    print(io, "├── Ri_dependent_tapering: ", prettysummary(closure.Ri_dependent_tapering), '\n')
    print(io, "├── κ₀: ", prettysummary(closure.κ₀), '\n')
    print(io, "├── ν₀: ", prettysummary(closure.ν₀), '\n')
    print(io, "├── κᶜᵃ: ", prettysummary(closure.κᶜᵃ), '\n')
    print(io, "├── Cᵉⁿ: ", prettysummary(closure.Cᵉⁿ), '\n')
    print(io, "├── Cᵃᵛ: ", prettysummary(closure.Cᵃᵛ), '\n')
    print(io, "├── Ri₀: ", prettysummary(closure.Ri₀), '\n')
    print(io, "├── Riᵟ: ", prettysummary(closure.Riᵟ), '\n')
    print(io, "├── maximum_diffusivity: ", prettysummary(closure.maximum_diffusivity), '\n')
    print(io, "└── maximum_viscosity: ", prettysummary(closure.maximum_viscosity))
end
