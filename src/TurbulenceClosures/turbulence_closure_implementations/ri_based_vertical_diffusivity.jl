using Oceananigans.Architectures: architecture, device_event, arch_array
using Oceananigans.BuoyancyModels: ∂z_b
using Oceananigans.Operators: ℑzᵃᵃᶜ
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: Riᶜᶜᶜ, Riᶜᶜᶠ

struct RiBasedVerticalDiffusivity{TD, FT, LZ} <: AbstractScalarDiffusivity{TD, VerticalFormulation}
    ν₀   :: FT
    Ri₀ν :: FT
    Riᵟν :: FT
    κ₀   :: FT
    Ri₀κ :: FT
    Riᵟκ :: FT
    coefficient_z_location :: LZ
end

RiBasedVerticalDiffusivity{TD}(ν₀::FT, Ri₀ν::FT, Riᵟν::FT, κ₀::FT, Ri₀κ::FT, Riᵟκ::FT, coefficient_z_location::LZ) where {TD, FT, LZ} =
    RiBasedVerticalDiffusivity{TD, FT, LZ}(ν₀, Ri₀ν, Riᵟν, κ₀, Ri₀κ, Riᵟκ, coefficient_z_location)

"""
    RiBasedVerticalDiffusivity([td=VerticallyImplicitTimeDiscretization(), FT=Float64] kwargs...)

Returns a closure that estimates the vertical viscosity and diffusivity
from "convective adjustment" coefficients `ν₀` and `κ₀` multiplied by
a decreasing function of the Richardson number.

Keyword Arguments
=========

* ν₀ (Float64 parameter): Convective adjustment viscosity. Default: 0.01
* Ri₀ν (Float64 parameter): Ri threshold for decreasing viscosity. Default: -0.5
* Riᵟν (Float64 parameter): Width over which Ri decreases to 0. Default: 1.0
* κ₀ (Float64 parameter): Convective adjustment diffusivity for tracers. Default: 0.1
* Ri₀κ (Float64 parameter): Ri threshold for decreasing viscosity. Default: -0.5
* Riᵟκ (Float64 parameter): Width over which Ri decreases to 0. Default: 1.0
* coefficient_z_location (Face() or Center()): The vertical location of the diffusivity and viscosity.
                                               Default: Face().
"""
function RiBasedVerticalDiffusivity(time_discretization = VerticallyImplicitTimeDiscretization(),
                                    FT = Float64;
                                    coefficient_z_location = Face(),
                                    ν₀   = 0.01,
                                    Ri₀ν = -0.5,
                                    Riᵟν = 1.0,
                                    κ₀   = 0.1,
                                    Ri₀κ = -0.5,
                                    Riᵟκ = 1.0)

    TD = typeof(time_discretization)

    coefficient_z_location isa Face || coefficient_z_location isa Center ||
        error("coefficient_z_location is $LZ but must be `Face()` or `Center()`!")

    return RiBasedVerticalDiffusivity{TD}(ν₀, Ri₀ν, Riᵟν, κ₀, Ri₀κ, Riᵟκ, coefficient_z_location)
end

RiBasedVerticalDiffusivity(FT::DataType; kw...) =
    RiBasedVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kw...)

#####
##### Diffusivity field utilities
#####

const RBVD{LZ} = RiBasedVerticalDiffusivity{<:Any, <:Any, LZ} where LZ
const RBVDArray{LZ} = AbstractArray{<:RBVD{LZ}} where LZ
const FlavorOfRBVD{LZ} = Union{RBVD{LZ}, RBVDArray{LZ}} where LZ

@inline viscosity_location(::FlavorOfRBVD{LZ}) where LZ = (Center(), Center(), LZ())
@inline diffusivity_location(::FlavorOfRBVD{LZ}) where LZ = (Center(), Center(), LZ())
@inline viscosity(::FlavorOfRBVD, diffusivities) = diffusivities.ν
@inline diffusivity(::FlavorOfRBVD, diffusivities, id) = diffusivities.κ

with_tracers(tracers, closure::FlavorOfRBVD) = closure

# Note: computing diffusivities at cell centers for now.
function DiffusivityFields(grid, tracer_names, bcs, closure::FlavorOfRBVD{LZ}) where LZ
    κ = Field{Center, Center, LZ}(grid)
    ν = Field{Center, Center, LZ}(grid)
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

# 1. x < x₀     => step_down = 1
# 2. x > x₀ + δ => step_down = 0
# 3. Otherwise, vary linearly between 1 and 0
@inline step_down(x::T, x₀, δ) where T = one(T) - min(one(T), max(zero(T), (x - x₀) / δ))

@kernel function compute_ri_based_diffusivities!(diffusivities, grid, closure::FlavorOfRBVD{LZ}, velocities, tracers, buoyancy) where LZ
    i, j, k, = @index(Global, NTuple)

    # Ensure this works with "ensembles" of closures, in addition to ordinary single closures
    closure_ij = getclosure(i, j, closure)

    ν₀   = closure_ij.ν₀    
    Ri₀ν = closure_ij.Ri₀ν 
    Riᵟν = closure_ij.Riᵟν 
    κ₀   = closure_ij.κ₀    
    Ri₀κ = closure_ij.Ri₀κ 
    Riᵟκ = closure_ij.Riᵟκ 

    Ri = ifelse(LZ === Type{Face}, Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy),
                                   Riᶜᶜᶠ(i, j, k, grid, velocities, tracers, buoyancy))

    @inbounds diffusivities.κ[i, j, k] = κ₀ * step_down(Ri, Ri₀κ, Riᵟκ)
    @inbounds diffusivities.ν[i, j, k] = ν₀ * step_down(Ri, Ri₀ν, Riᵟν)
end

#####
##### Show
#####

Base.summary(closure::RiBasedVerticalDiffusivity{TD}) where TD = string("RiBasedVerticalDiffusivity{$TD}")
Base.show(io::IO, closure::RiBasedVerticalDiffusivity) = print(io, summary(closure))

