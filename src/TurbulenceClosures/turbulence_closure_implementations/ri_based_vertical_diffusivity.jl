using Oceananigans.Architectures: architecture, device_event, arch_array
using Oceananigans.BuoyancyModels: ∂z_b
using Oceananigans.Operators
using Oceananigans.Operators: ℑzᵃᵃᶜ

struct RiBasedVerticalDiffusivity{TD, FT, R} <: AbstractScalarDiffusivity{TD, VerticalFormulation}
    ν₀  :: FT
    κ₀  :: FT
    κᶜ  :: FT
    Ri₀ :: FT
    Riᵟ :: FT
    Ri_dependent_tapering :: R
end

function RiBasedVerticalDiffusivity{TD}(ν₀::FT, κ₀::FT, κᶜ::FT, Ri₀::FT, Riᵟ::FT,
                                        Ri_dependent_tapering::R) where {TD, FT, R}

    return RiBasedVerticalDiffusivity{TD, FT, R}(ν₀, κ₀, κᶜ, Ri₀, Riᵟ, Ri_dependent_tapering)
end

# Ri-dependent tapering flavor
struct PiecewiseLinearRiDependentTapering end
struct ExponentialRiDependentTapering end
struct HyperbolicTangentRiDependentTapering end

"""
    RiBasedVerticalDiffusivity([time_discretization = VerticallyImplicitTimeDiscretization(),
                               FT = Float64;]
                               coefficient_z_location = Face(),
                               Ri_dependent_tapering = ExponentialRiDependentTapering(),
                               ν₀   =  0.92,
                               Ri₀ν = -1.34,
                               Riᵟν =  0.61,
                               κ₀   =  0.18,
                               Ri₀κ = -0.13,
                               Riᵟκ =  0.6)

Return a closure that estimates the vertical viscosity and diffusivity
from "convective adjustment" coefficients `ν₀` and `κ₀` multiplied by
a decreasing function of the Richardson number, ``Ri``.

Keyword Arguments
=================
* `ν₀` (Float64): Non-convective viscosity.
* `κ₀` (Float64): Non-convective diffusivity for tracers.
* `κᶜ` (Float64): Convective adjustment diffusivity for tracers.
* `Ri₀` (Float64): ``Ri`` threshold for decreasing viscosity and diffusivity.
* `Riᵟ` (Float64): ``Ri``-width over which viscosity and diffusivity decreases to 0.
"""
function RiBasedVerticalDiffusivity(time_discretization = VerticallyImplicitTimeDiscretization(),
                                    FT = Float64;
                                    Ri_dependent_tapering = ExponentialRiDependentTapering(),
                                    ν₀  = 2.7e-2,
                                    κ₀  = 1.9e-2,
                                    κᶜ  = 0.8,
                                    Ri₀ = 0.4,
                                    Riᵟ = 0.2)

    TD = typeof(time_discretization)
    R = typeof(Ri_dependent_tapering)

    return RiBasedVerticalDiffusivity{TD}(FT(ν₀), FT(κ₀), FT(κᶜ), FT(Ri₀), FT(Riᵟ), Ri_dependent_tapering)
end

RiBasedVerticalDiffusivity(FT::DataType; kw...) =
    RiBasedVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kw...)

#####
##### Diffusivity field utilities
#####

const RBVD = RiBasedVerticalDiffusivity
const RBVDArray = AbstractArray{<:RBVD}
const FlavorOfRBVD = Union{RBVD, RBVDArray}

#@inline viscosity_location(::FlavorOfRBVD) = (Center(), Center(), Face())
#@inline diffusivity_location(::FlavorOfRBVD) = (Center(), Center(), Face())

@inline viscosity_location(::FlavorOfRBVD) = (Center(), Center(), Center())
@inline diffusivity_location(::FlavorOfRBVD) = (Center(), Center(), Center())

@inline viscosity(::FlavorOfRBVD, diffusivities) = diffusivities.ν
@inline diffusivity(::FlavorOfRBVD, diffusivities, id) = diffusivities.κ

with_tracers(tracers, closure::FlavorOfRBVD) = closure

# Note: computing diffusivities at cell centers for now.
function DiffusivityFields(grid, tracer_names, bcs, closure::FlavorOfRBVD)
    κ = Field{Center, Center, Center}(grid)
    ν = Field{Center, Center, Center}(grid)
    #κ = Field{Center, Center, Face}(grid)
    #ν = Field{Center, Center, Face}(grid)
    return (; κ, ν)
end

function calculate_diffusivities!(diffusivities, closure::FlavorOfRBVD, model)

    arch = model.architecture
    grid = model.grid
    tracers = model.tracers
    buoyancy = model.buoyancy
    velocities = model.velocities

    event = launch!(arch, grid, :xyz,
                    compute_ri_based_diffusivities!, diffusivities, grid, closure, velocities, tracers, buoyancy,
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
    return ifelse(S² == 0, Inf, ifelse(N² <= 0, zero(grid), N² / S²))
end

@inline Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy) =
    ℑzᵃᵃᶜ(i, j, k, grid, Riᶜᶜᶠ, velocities, tracers, buoyancy)

@kernel function compute_ri_based_diffusivities!(diffusivities, grid, closure::FlavorOfRBVD,
                                                 velocities, tracers, buoyancy)

    i, j, k, = @index(Global, NTuple)

    # Ensure this works with "ensembles" of closures, in addition to ordinary single closures
    closure_ij = getclosure(i, j, closure)

    ν₀  = closure_ij.ν₀
    κ₀  = closure_ij.κ₀
    κᶜ  = closure_ij.κᶜ
    Ri₀ = closure_ij.Ri₀
    Riᵟ = closure_ij.Riᵟ
    tapering = closure_ij.Ri_dependent_tapering

    # For a ccf-based scheme
    # Ri = Riᶜᶜᶠ(i, j, k, grid, velocities, tracers, buoyancy)
    # N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    # convecting = N² < 0

    # For a ccc-based scheme
    Ri = Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy)
    N²⁺ = ∂z_b(i, j, k+1, grid, buoyancy, tracers)
    convecting = (N² < 0) | (N²⁺ < 0)

    κa = ifelse(convecting, κᶜ, zero(grid))

    @inbounds diffusivities.κ[i, j, k] = (κa + κ₀ * taper(tapering, Ri, Ri₀, Riᵟ)) # * Δzᶜᶜᶠ(i, j, k, grid)^2
    @inbounds diffusivities.ν[i, j, k] =       ν₀ * taper(tapering, Ri, Ri₀, Riᵟ)  # * Δzᶜᶜᶠ(i, j, k, grid)^2
end

#####
##### Show
#####

Base.summary(closure::RiBasedVerticalDiffusivity{TD}) where TD = string("RiBasedVerticalDiffusivity{$TD}")
Base.show(io::IO, closure::RiBasedVerticalDiffusivity) = print(io, summary(closure))
