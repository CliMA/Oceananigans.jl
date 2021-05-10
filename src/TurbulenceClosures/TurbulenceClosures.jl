module TurbulenceClosures

export
    AbstractEddyViscosityClosure,
    IsotropicDiffusivity,
    AnisotropicDiffusivity,
    AnisotropicBiharmonicDiffusivity,
    TwoDimensionalLeith,
    ConstantSmagorinsky,
    SmagorinskyLilly,
    AnisotropicMinimumDissipation,
    HorizontallyCurvilinearAnisotropicDiffusivity,

    DiffusivityFields,
    calculate_diffusivities!,

    ∇_κ_∇c,
    ∇_κ_∇T,
    ∇_κ_∇S,
    ∂ⱼ_2ν_Σ₁ⱼ,
    ∂ⱼ_2ν_Σ₂ⱼ,
    ∂ⱼ_2ν_Σ₃ⱼ,

    cell_diffusion_timescale

using CUDA
using KernelAbstractions

import Oceananigans.Utils: with_tracers

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.BoundaryConditions
using Oceananigans.Fields
using Oceananigans.BuoyancyModels
using Oceananigans.Utils

using Oceananigans.Architectures: AbstractArchitecture, device

#####
##### Abstract types
#####

"""
    AbstractTurbulenceClosure

Abstract supertype for turbulence closures.
"""
abstract type AbstractTurbulenceClosure{TimeDiscretization} end

#####
##### Include module code
#####

include("implicit_explicit_time_discretization.jl")
include("turbulence_closure_utils.jl")
include("diffusion_operators.jl")
include("viscous_dissipation_operators.jl")
include("velocity_tracer_gradients.jl")
include("abstract_isotropic_diffusivity_closure.jl")
include("abstract_eddy_viscosity_closure.jl")
include("closure_tuples.jl")

# Implementations:
include("turbulence_closure_implementations/nothing_closure.jl")
include("turbulence_closure_implementations/isotropic_diffusivity.jl")
include("turbulence_closure_implementations/anisotropic_diffusivity.jl")
include("turbulence_closure_implementations/horizontally_curvilinear_anisotropic_diffusivity.jl")
include("turbulence_closure_implementations/horizontally_curvilinear_anisotropic_biharmonic_diffusivity.jl")
include("turbulence_closure_implementations/anisotropic_biharmonic_diffusivity.jl")
include("turbulence_closure_implementations/leith_enstrophy_diffusivity.jl")
include("turbulence_closure_implementations/smagorinsky_lilly.jl")
include("turbulence_closure_implementations/anisotropic_minimum_dissipation.jl")

# Miscellaneous utilities
include("diffusivity_fields.jl")
include("turbulence_closure_diagnostics.jl")
include("vertically_implicit_diffusion_solver.jl")

#####
##### Some value judgements here
#####

"""
    ConstantSmagorinsky

An alias for `SmagorinskyLilly`.
"""
const ConstantSmagorinsky = SmagorinskyLilly

end # module
