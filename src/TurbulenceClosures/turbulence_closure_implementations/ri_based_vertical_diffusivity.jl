using Oceananigans.Architectures: architecture, device_event, arch_array
using Oceananigans.BuoyancyModels: ∂z_b
using Oceananigans.Operators
using Oceananigans.Operators: ℑzᵃᵃᶜ

struct RiBasedVerticalDiffusivity{TD, FT, R} <: AbstractScalarDiffusivity{TD, VerticalFormulation}
    ν₀  :: FT
    κ₀  :: FT
    κᶜ  :: FT
    Cᵉ  :: FT
    Ri₀_κ :: FT
    Riᵟ_κ :: FT
    Ri₀_ν :: FT
    Riᵟ_ν :: FT
    Ri_dependent_tapering :: R
end

function RiBasedVerticalDiffusivity{TD}(ν₀::FT,
                                        κ₀::FT,
                                        κᶜ::FT,
                                        Cᵉ::FT,
                                        Ri₀_κ::FT,
                                        Riᵟ_κ::FT,
                                        Ri₀_ν::FT,
                                        Riᵟ_ν::FT,
                                        Ri_dependent_tapering::R) where {TD, FT, R}

    return RiBasedVerticalDiffusivity{TD, FT, R}(ν₀, κ₀, κᶜ, Cᵉ,
                                                 Ri₀_κ, Riᵟ_κ,
                                                 Ri₀_ν, Riᵟ_ν,
                                                 Ri_dependent_tapering)
end

# Ri-dependent tapering flavor
struct PiecewiseLinearRiDependentTapering end
struct ExponentialRiDependentTapering end
struct HyperbolicTangentRiDependentTapering end

"""
    RiBasedVerticalDiffusivity(time_discretization = VerticallyImplicitTimeDiscretization(),
                               FT = Float64;
                               Ri_dependent_tapering = ExponentialRiDependentTapering(),
                               ν₀  = 0.30,
                               κ₀  = 0.42,
                               κᶜ  = 4.0,
                               Cᵉ  = 0.57,
                               Ri₀ = 0.27,
                               Riᵟ = 0.20,
                               warning = true)

Return a closure that estimates the vertical viscosity and diffusivity
from "convective adjustment" coefficients `ν₀` and `κ₀` multiplied by
a decreasing function of the Richardson number, ``Ri``.

Keyword Arguments
=================
* `ν₀` (Float64): Non-convective viscosity.
* `κ₀` (Float64): Non-convective diffusivity for tracers.
* `κᶜ` (Float64): Convective adjustment diffusivity for tracers.
* `Cᵉ` (Float64): Entrainment coefficient for tracers.
* `Ri₀` (Float64): ``Ri`` threshold for decreasing viscosity and diffusivity.
* `Riᵟ` (Float64): ``Ri``-width over which viscosity and diffusivity decreases to 0.
"""
function RiBasedVerticalDiffusivity(time_discretization = VerticallyImplicitTimeDiscretization(),
                                    FT = Float64;
                                    #Ri_dependent_tapering = ExponentialRiDependentTapering(),
                                    Ri_dependent_tapering = HyperbolicTangentRiDependentTapering(),
                                    ν₀  = 0.30,
                                    κ₀  = 0.42,
                                    κᶜ  = 4.0,
                                    Cᵉ  = 0.57,
                                    Ri₀_κ = 0.27,
                                    Riᵟ_κ = 0.20,
                                    Ri₀_ν = 0.27,
                                    Riᵟ_ν = 0.20,
                                    warning = true)
    if warning
        @warn "RiBasedVerticalDiffusivity is an experimental turbulence closure that \n" *
              "is unvalidated and whose default parameters are not calibrated for \n" * 
              "realistic ocean conditions or for use in a three-dimensional \n" *
              "simulation. Use with caution and report bugs and problems with physics \n" *
              "to https://github.com/CliMA/Oceananigans.jl/issues."
    end

    TD = typeof(time_discretization)
    R = typeof(Ri_dependent_tapering)

    return RiBasedVerticalDiffusivity{TD}(FT(ν₀), FT(κ₀), FT(κᶜ), FT(Cᵉ),
                                          FT(Ri₀_κ), FT(Riᵟ_κ),
                                          FT(Ri₀_ν), FT(Riᵟ_ν),
                                          Ri_dependent_tapering)
end

RiBasedVerticalDiffusivity(FT::DataType; kw...) =
    RiBasedVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kw...)

#####
##### Diffusivity field utilities
#####

const RBVD = RiBasedVerticalDiffusivity
const RBVDArray = AbstractArray{<:RBVD}
const FlavorOfRBVD = Union{RBVD, RBVDArray}

@inline viscosity_location(::FlavorOfRBVD)   = (Center(), Center(), Face())
@inline diffusivity_location(::FlavorOfRBVD) = (Center(), Center(), Face())

@inline viscosity(::FlavorOfRBVD, diffusivities) = diffusivities.ν
@inline diffusivity(::FlavorOfRBVD, diffusivities, id) = diffusivities.κ

with_tracers(tracers, closure::FlavorOfRBVD) = closure

# Note: computing diffusivities at cell centers for now.
function DiffusivityFields(grid, tracer_names, bcs, closure::FlavorOfRBVD)
    κ = Field{Center, Center, Face}(grid)
    ν = Field{Center, Center, Face}(grid)
    return (; κ, ν)
end

function calculate_diffusivities!(diffusivities, closure::FlavorOfRBVD, model)
    arch = model.architecture
    grid = model.grid
    clock = model.clock
    tracers = model.tracers
    buoyancy = model.buoyancy
    velocities = model.velocities
    top_tracer_bcs = NamedTuple(c => tracers[c].boundary_conditions.top for c in propertynames(tracers))

    event = launch!(arch, grid, :xyz,
                    compute_ri_based_diffusivities!,
                    diffusivities,
                    grid,
                    closure,
                    velocities,
                    tracers,
                    buoyancy,
                    top_tracer_bcs,
                    clock,
                    dependencies = device_event(arch))

    wait(device(arch), event)

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

@inline function Riᶜᶜᶠ(i, j, k, grid, velocities, tracers, buoyancy)
    ∂z_u² = ℑxᶜᵃᵃ(i, j, k, grid, ϕ², ∂zᶠᶜᶠ, velocities.u)
    ∂z_v² = ℑyᵃᶜᵃ(i, j, k, grid, ϕ², ∂zᶜᶠᶠ, velocities.v)
    S² = ∂z_u² + ∂z_v²
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    Ri = N² / S²

    # Clip N² and avoid NaN
    return ifelse(N² <= 0, zero(grid), Ri)
end

@inline Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy) =
    ℑzᵃᵃᶜ(i, j, k, grid, Riᶜᶜᶠ, velocities, tracers, buoyancy)

@kernel function compute_ri_based_diffusivities!(diffusivities, grid, closure::FlavorOfRBVD,
                                                 velocities, tracers, buoyancy, tracer_bcs, clock)

    i, j, k, = @index(Global, NTuple)

    # Ensure this works with "ensembles" of closures, in addition to ordinary single closures
    closure_ij = getclosure(i, j, closure)

    ν₀  = closure_ij.ν₀
    κ₀  = closure_ij.κ₀
    κᶜ  = closure_ij.κᶜ
    Cᵉ  = closure_ij.Cᵉ

    #Ri₀ = closure_ij.Ri₀
    #Riᵟ = closure_ij.Riᵟ

    Ri₀_κ = closure_ij.Ri₀_κ
    Riᵟ_κ = closure_ij.Riᵟ_κ
    Ri₀_ν = closure_ij.Ri₀_ν
    Riᵟ_ν = closure_ij.Riᵟ_ν

    tapering = closure_ij.Ri_dependent_tapering
    Qᵇ = top_buoyancy_flux(i, j, grid, buoyancy, tracer_bcs, clock, merge(velocities, tracers))

    # Convection and entrainment
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    N²_above = ∂z_b(i, j, k+1, grid, buoyancy, tracers)
    convecting = N² < 0
    entraining = (!convecting) & (N²_above < 0)

    # Convective adjustment diffusivity
    κᶜ = ifelse(convecting, κᶜ, zero(grid))

    # Entrainment diffusivity
    κᵉ = ifelse(Qᵇ > 0, Cᵉ * Qᵇ / N², zero(grid))
    κᵉ = ifelse(entraining, Cᵉ, zero(grid))

    # Shear mixing diffusivity and viscosity
    Ri = Riᶜᶜᶠ(i, j, k, grid, velocities, tracers, buoyancy)

    # τ = taper(tapering, Ri, Ri₀, Riᵟ)
    # κ★ = κ₀ * τ
    # ν★ = ν₀ * τ

    τ_κ = taper(ExponentialRiDependentTapering(), Ri, Ri₀_κ, Riᵟ_κ)
    κ★ = κ₀ * τ_κ

    τ_ν = taper(PiecewiseLinearRiDependentTapering(), Ri, Ri₀_ν, Riᵟ_ν)
    ν★ = ν₀ * τ_ν

    @inbounds diffusivities.κ[i, j, k] = κᶜ + κᵉ + κ★
    @inbounds diffusivities.ν[i, j, k] = ν★
end

#####
##### Show
#####

Base.summary(closure::RiBasedVerticalDiffusivity{TD}) where TD = string("RiBasedVerticalDiffusivity{$TD}")
Base.show(io::IO, closure::RiBasedVerticalDiffusivity) = print(io, summary(closure))
