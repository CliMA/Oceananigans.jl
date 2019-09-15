module TurbulenceClosures

export
  IsotropicDiffusivity,
  ConstantIsotropicDiffusivity,

  ConstantAnisotropicDiffusivity,

  ConstantSmagorinsky,
  DeardorffSmagorinsky,
  BlasiusSmagorinsky,

  AnisotropicMinimumDissipation,
  RozemaAnisotropicMinimumDissipation,
  VerstappenAnisotropicMinimumDissipation,

  TurbulentDiffusivities,
  calc_diffusivities!,

  ∇_κ_∇c,
  ∇_κ_∇T,
  ∇_κ_∇S,
  ∂ⱼ_2ν_Σ₁ⱼ,
  ∂ⱼ_2ν_Σ₂ⱼ,
  ∂ⱼ_2ν_Σ₃ⱼ,

  cell_diffusion_timescale

using
  Oceananigans,
  Oceananigans.Operators,
  GPUifyLoops

using Oceananigans: AbstractArchitecture, AbstractGrid, buoyancy_perturbation

@hascuda using CUDAdrv, CUDAnative

# Approximate viscosities and thermal diffusivities for seawater
# at 20ᵒC and 35 psu, according to Sharqawy et al., "Thermophysical
# properties of seawater: A review of existing correlations and data" (2010).
const ν₀ = 1.05e-6
const κ₀ = 1.46e-7

abstract type TurbulenceClosure{T} end
abstract type IsotropicDiffusivity{T} <: TurbulenceClosure{T} end
abstract type TensorDiffusivity{T} <: TurbulenceClosure{T} end
abstract type AbstractSmagorinsky{T} <: IsotropicDiffusivity{T} end
abstract type AbstractAnisotropicMinimumDissipation{T} <: IsotropicDiffusivity{T} end

@inline ∇_κ_∇T(args...) = ∇_κ_∇c(args...)
@inline ∇_κ_∇S(args...) = ∇_κ_∇c(args...)

# Fallback constructor for diffusivity types without precomputed diffusivities:
TurbulentDiffusivities(arch::AbstractArchitecture, grid::AbstractGrid, args...) = nothing

include("turbulence_closure_utils.jl")
include("closure_operators.jl")
include("velocity_tracer_gradients.jl")

include("constant_diffusivity_closures.jl")
include("smagorinsky.jl")
include("rozema_anisotropic_minimum_dissipation.jl")
include("verstappen_anisotropic_minimum_dissipation.jl")

include("turbulence_closure_diagnostics.jl")

# Some value judgements here:
const AnisotropicMinimumDissipation = VerstappenAnisotropicMinimumDissipation
const ConstantSmagorinsky = DeardorffSmagorinsky

end # module
