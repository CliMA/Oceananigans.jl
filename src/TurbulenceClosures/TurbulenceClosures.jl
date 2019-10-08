module TurbulenceClosures

export
  IsotropicDiffusivity,
  ConstantIsotropicDiffusivity,
  ConstantAnisotropicDiffusivity,
  ConstantSmagorinsky,
  SmagorinskyLilly,
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

import Oceananigans: with_tracers

using Oceananigans: AbstractArchitecture, AbstractGrid, buoyancy_perturbation, buoyancy_frequency_squared, 
                    TracerFields, device, launch_config

@hascuda using CUDAdrv, CUDAnative

####
#### Molecular viscosity and thermal diffusivity definitions
#### Approximate viscosities and thermal diffusivities for seawater at 20ᵒC and 35 psu,
#### according to Sharqawy et al., "Thermophysical properties of seawater: A review of
#### existing correlations and data" (2010).
####

const ν₀ = 1.05e-6
const κ₀ = 1.46e-7

####
#### Abstract types
####

"""
    TurbulenceClosure{FT}

Abstract supertype for turbulence closures with model parameters stored as properties of
type `FT`.
"""
abstract type TurbulenceClosure{FT} end

"""
    IsotropicDiffusivity{FT} <: TurbulenceClosure{FT}

Abstract supertype for turbulence closures that are defined by an isotropic viscosity
and isotropic diffusivities with model parameters stored as properties of type `FT`.
"""
abstract type IsotropicDiffusivity{FT} <: TurbulenceClosure{FT} end

"""
    TensorDiffusivity{FT} <: TurbulenceClosure{FT}

Abstract supertype for turbulence closures that are defined by a tensor viscosity and
tensor diffusivities with model parameters stored as properties of type `FT`.
"""
abstract type TensorDiffusivity{FT} <: TurbulenceClosure{FT} end

"""
    AbstractSmagorinsky{FT}

Abstract supertype for large eddy simulation models based off the model described
by Smagorinsky with model parameters stored as properties of type `FT`.
"""
abstract type AbstractSmagorinsky{FT} <: IsotropicDiffusivity{FT} end

"""
    AbstractAnisotropicMinimumDissipation{FT}

Abstract supertype for large eddy simulation models based on the anisotropic minimum
dissipation principle with model parameters stored as properties of type `FT`.
"""
abstract type AbstractAnisotropicMinimumDissipation{FT} <: IsotropicDiffusivity{FT} end

####
#### Include module code
####

# Fallback constructor for diffusivity types without precomputed diffusivities:
TurbulentDiffusivities(arch::AbstractArchitecture, grid::AbstractGrid, args...) = nothing

include("turbulence_closure_utils.jl")
include("closure_operators.jl")
include("velocity_tracer_gradients.jl")

include("constant_isotropic_diffusivity.jl")
include("constant_anisotropic_diffusivity.jl")
include("smagorinsky.jl")
include("verstappen_anisotropic_minimum_dissipation.jl")
include("rozema_anisotropic_minimum_dissipation.jl")

include("turbulence_closure_diagnostics.jl")

####
#### Some value judgements here
####

"""
    AnisotropicMinimumDissipation

An alias for `VerstappenAnisotropicMinimumDissipation`.
"""
const AnisotropicMinimumDissipation = VerstappenAnisotropicMinimumDissipation

"""
    ConstantSmagorinsky

An alias for `SmagorinskyLilly`.
"""
const ConstantSmagorinsky = SmagorinskyLilly

end # module
