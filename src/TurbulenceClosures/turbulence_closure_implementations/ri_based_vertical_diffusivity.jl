using Oceananigans.Architectures: architecture, device_event, arch_array
using Oceananigans.BuoyancyModels: ∂z_b
using Oceananigans.Operators: ℑzᵃᵃᶜ
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: Riᶜᶜᶜ

struct RiBasedVerticalDiffusivity{TD, FT} <: AbstractScalarDiffusivity{TD, VerticalFormulation}
    ν₀   :: FT
    Ri₀ν :: FT
    Riᵟν :: FT
    κ₀   :: FT
    Ri₀κ :: FT
    Riᵟκ :: FT
end

"""
    RiBasedVerticalDiffusivity
"""
function RiBasedVerticalDiffusivity(time_discretization = VerticallyImplicitTimeDiscretization(),
                                    FT = Float64;
                                    ν₀   = 0.01,
                                    Ri₀ν = 0.0,
                                    Riᵟν = 0.5,
                                    κ₀   = 0.1,
                                    Ri₀κ = 0.0,
                                    Riᵟκ = 0.5)

    TD = typeof(time_discretization)
    return RiBasedVerticalDiffusivity{TD, FT}(ν₀, Ri₀ν, Riᵟν, κ₀, Ri₀κ, Riᵟκ)
end

RiBasedVerticalDiffusivity(FT::DataType; kw...) =
    RiBasedVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kw...)


#####
##### Diffusivity field utilities
#####

const RBVD = RiBasedVerticalDiffusivity
const RBVDArray = AbstractArray{<:RBVD}
const FlavorOfRBVD = Union{RBVD, RBVDArray}

with_tracers(tracers, closure::FlavorOfRBVD) = closure

# Note: computing diffusivities at cell centers for now.
DiffusivityFields(grid, tracer_names, bcs, closure::FlavorOfRBVD) =
    (; κ = CenterField(grid), ν = CenterField(grid))

@inline viscosity(::FlavorOfRBVD, diffusivities) = diffusivities.ν
@inline diffusivity(::FlavorOfRBVD, diffusivities, id) = diffusivities.κ

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

@kernel function compute_ri_based_diffusivities!(diffusivities, grid, closure, velocities, tracers, buoyancy)
    i, j, k, = @index(Global, NTuple)

    # Ensure this works with "ensembles" of closures, in addition to ordinary single closures
    closure_ij = getclosure(i, j, closure)

    ν₀   = closure_ij.ν₀    
    Ri₀ν = closure_ij.Ri₀ν 
    Riᵟν = closure_ij.Riᵟν 
    κ₀   = closure_ij.κ₀    
    Ri₀κ = closure_ij.Ri₀κ 
    Riᵟκ = closure_ij.Riᵟκ 

    Ri = Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy)

    @inbounds diffusivities.κ[i, j, k] = κ₀ * step_down(Ri, Ri₀κ, Riᵟκ)
    @inbounds diffusivities.ν[i, j, k] = ν₀ * step_down(Ri, Ri₀ν, Riᵟν)
end

#####
##### Show
#####

Base.summary(closure::RiBasedVerticalDiffusivity{TD}) where TD = string("RiBasedVerticalDiffusivity{$TD}")
Base.show(io::IO, closure::RiBasedVerticalDiffusivity) = print(io, summary(closure))

