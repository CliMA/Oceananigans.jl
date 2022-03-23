using Oceananigans.Architectures: architecture, device_event, arch_array
using Oceananigans.BuoyancyModels: ∂z_b
using Oceananigans.Operators: ℑzᵃᵃᶜ
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: Riᶜᶜᶜ, scale

struct RiBasedVerticalDiffusivity{TD, FT} <: AbstractScalarDiffusivity{TD, VerticalFormulation}
    ν₁    :: FT
    r_ν   :: FT
    Riᶜ_ν :: FT
    Riᵟ_ν :: FT
    κ₁    :: FT
    r_κ  :: FT
    Riᶜ_κ :: FT
    Riᵟ_κ :: FT
end

"""
    RiBasedVerticalDiffusivity
"""
function RiBasedVerticalDiffusivity(time_discretization = VerticallyImplicitTimeDiscretization(),
                                    FT = Float64;
                                    ν₁ = 0.1,
                                    r_ν = 1e-2,
                                    Riᶜ_ν = -2.0,
                                    Riᵟ_ν = 0.1,
                                    κ₁ = 0.1,
                                    r_κ = 1e-2,
                                    Riᶜ_κ = -2.0,
                                    Riᵟ_κ = 0.1)

    TD = typeof(time_discretization)
    return RiBasedVerticalDiffusivity{TD, FT}(ν₁, r_ν, Riᶜ_ν, Riᵟ_ν, κ₁, r_κ, Riᶜ_κ, Riᵟ_κ)
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

@kernel function compute_ri_based_diffusivities!(diffusivities, grid, closure, velocities, tracers, buoyancy)
    i, j, k, = @index(Global, NTuple)

    # Ensure this works with "ensembles" of closures, in addition to ordinary single closures
    closure_ij = getclosure(i, j, closure)

    ν₁    = closure_ij.ν₁    
    r_ν   = closure_ij.r_ν     
    Riᶜ_ν = closure_ij.Riᶜ_ν 
    Riᵟ_ν = closure_ij.Riᵟ_ν 
    κ₁    = closure_ij.κ₁    
    r_κ   = closure_ij.r_κ   
    Riᶜ_κ = closure_ij.Riᶜ_κ 
    Riᵟ_κ = closure_ij.Riᵟ_κ 

    Ri = Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy)

    @inbounds diffusivities.κ[i, j, k] = scale(Ri, κ₁, r_κ, Riᶜ_κ, Riᵟ_κ)
    @inbounds diffusivities.ν[i, j, k] = scale(Ri, ν₁, r_ν, Riᶜ_ν, Riᵟ_ν)
end

#####
##### Show
#####

Base.summary(closure::RiBasedVerticalDiffusivity{TD}) where TD = string("RiBasedVerticalDiffusivity{$TD}")
Base.show(io::IO, closure::RiBasedVerticalDiffusivity) = print(io, summary(closure))

