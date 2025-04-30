using Oceananigans.Architectures: architecture
using Oceananigans.BuoyancyFormulations: ∂z_b
using Oceananigans.Operators
using Oceananigans.Grids: inactive_node
using Oceananigans.Operators: ℑzᵃᵃᶜ

struct RiBasedVerticalDiffusivity{TD, FT, HR} <: AbstractScalarDiffusivity{TD, VerticalFormulation, 1}
    ν₀ :: FT
    κ₀ :: FT
    νs :: FT
    νc :: FT
    Prs :: FT
    Prc :: FT
    Ric :: FT
    ΔRi :: FT
    horizontal_Ri_filter :: HR
end

function RiBasedVerticalDiffusivity{TD}(ν₀::FT,
                                        κ₀::FT,
                                        νs :: FT,
                                        νc :: FT,
                                        Prs :: FT,
                                        Prc :: FT,
                                        Ric :: FT,
                                        ΔRi :: FT,
                                        horizontal_Ri_filter::HR) where {TD, FT, HR}


    return RiBasedVerticalDiffusivity{TD, FT, HR}(ν₀, κ₀, νs, νc, Prs, Prc, Ric, ΔRi,
                                                  horizontal_Ri_filter)
end

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
                                    FT = Oceananigans.defaults.FloatType;
                                    ν₀ = 1e-5,
                                    κ₀ = 1e-5,
                                    νs = 0.0616,
                                    νc = 0.761,
                                    Prs = 1.08,
                                    Prc = 0.175,
                                    Ric = 0.437,
                                    ΔRi = 9.7e-3,
                                    horizontal_Ri_filter = FivePointHorizontalFilter(),
                                    warning = true)
    if warning
        @warn "RiBasedVerticalDiffusivity is an experimental turbulence closure that \n" *
              "is unvalidated and whose default parameters are not calibrated for \n" *
              "realistic ocean conditions or for use in a three-dimensional \n" *
              "simulation. Use with caution and report bugs and problems with physics \n" *
              "to https://github.com/CliMA/Oceananigans.jl/issues."
    end

    TD = typeof(time_discretization)

    return RiBasedVerticalDiffusivity{TD}(convert(FT, ν₀ ),
                                          convert(FT, κ₀ ),
                                          convert(FT, νs ),
                                          convert(FT, νc ),
                                          convert(FT, Prs),
                                          convert(FT, Prc),
                                          convert(FT, Ric),
                                          convert(FT, ΔRi),
                                          horizontal_Ri_filter)
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

@inline viscosity(::FlavorOfRBVD, diffusivities) = diffusivities.κu
@inline diffusivity(::FlavorOfRBVD, diffusivities, id) = diffusivities.κc

with_tracers(tracers, closure::FlavorOfRBVD) = closure

# Note: computing diffusivities at cell centers for now.
function build_diffusivity_fields(grid, clock, tracer_names, bcs, closure::FlavorOfRBVD)
    κc = Field((Center, Center, Face), grid)
    κu = Field((Center, Center, Face), grid)
    Ri = Field((Center, Center, Face), grid)
    return (; κc, κu, Ri)
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
    νs  = closure_ij.νs 
    νc  = closure_ij.νc 
    Prs = closure_ij.Prs
    Prc = closure_ij.Prc
    Ric = closure_ij.Ric
    ΔRi = closure_ij.ΔRi
    Ri_filter = closure_ij.horizontal_Ri_filter

    κs = νs / Prs
    κc = νc / Prc

    # (Potentially) apply a horizontal filter to the Richardson number
    Ri = filter_horizontally(i, j, k, grid, Ri_filter, diffusivities.Ri)

    # Shear mixing diffusivity and viscosity
    νconv  = (νs - νc) * tanh(Ri / ΔRi) + νs
    νshear = (ν₀ - νs) * Ri / Ric + νs

    κconv  = (κs - κc) * tanh(Ri / ∆Ri) + κs
    κshear = (κ₀ - κs) * Ri / Ric + κs

    # Previous diffusivities
    κc = diffusivities.κc
    κu = diffusivities.κu

    # New diffusivities
    κc⁺ = ifelse(Ri < 0, κconv, ifelse(Ri < Ric, κshear, κ₀))
    κu⁺ = ifelse(Ri < 0, νconv, ifelse(Ri < Ric, νshear, ν₀))

    # Set to zero on periphery and NaN within inactive region
    on_periphery = peripheral_node(i, j, k, grid, c, c, f)

    # Update by averaging in time
    @inbounds κc[i, j, k] = ifelse(on_periphery, zero(grid), κc⁺)
    @inbounds κu[i, j, k] = ifelse(on_periphery, zero(grid), κu⁺)

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
