module TurbulenceClosures

export
    AbstractEddyViscosityClosure,
    IsotropicDiffusivity,
    AnisotropicDiffusivity,
    AnisotropicBiharmonicDiffusivity,
    TwoDimensionalLeith,
    SmagorinskyLilly,
    AnisotropicMinimumDissipation,
    HorizontallyCurvilinearAnisotropicDiffusivity,
    ConvectiveAdjustmentVerticalDiffusivity,

    DiffusivityFields,
    calculate_diffusivities!,

    ∇_dot_qᶜ,
    ∂ⱼ_τ₁ⱼ,
    ∂ⱼ_τ₂ⱼ,
    ∂ⱼ_τ₃ⱼ,

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

# Fallbacks
add_closure_specific_boundary_conditions(closure, boundary_conditions, args...) = boundary_conditions

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
include("turbulence_closure_implementations/tke_based_vertical_diffusivity.jl")
include("turbulence_closure_implementations/convective_adjustment_vertical_diffusivity.jl")

# Miscellaneous utilities
include("diffusivity_fields.jl")
include("turbulence_closure_diagnostics.jl")
include("vertically_implicit_diffusion_solver.jl")

#####
##### Some value judgements here
#####

end # module
