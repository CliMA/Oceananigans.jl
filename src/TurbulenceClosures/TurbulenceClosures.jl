module TurbulenceClosures

export
    AbstractEddyViscosityClosure,
    ScalarDiffusivity,
    ScalarBiharmonicDiffusivity,
    TwoDimensionalLeith,
    SmagorinskyLilly,
    AnisotropicMinimumDissipation,
    ConvectiveAdjustmentVerticalDiffusivity,
    IsopycnalSkewSymmetricDiffusivity,

    ExplicitTimeDiscretization,
    VerticallyImplicitTimeDiscretization,
    ThreeDimensionalFormulation,
    HorizontalFormulation,
    VerticalFormulation, 

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
using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.BoundaryConditions
using Oceananigans.Fields
using Oceananigans.BuoyancyModels
using Oceananigans.Utils

using Oceananigans.Architectures: AbstractArchitecture, device

const VerticallyBoundedGrid{FT} = AbstractGrid{FT, <:Any, <:Any, <:Bounded}

#####
##### Abstract types
#####

"""
    AbstractTurbulenceClosure

Abstract supertype for turbulence closures.
"""
abstract type AbstractTurbulenceClosure{TimeDiscretization} end

@inline get_closure_i(i, closure::AbstractVector{<:AbstractTurbulenceClosure}) = @inbounds closure[i]
@inline get_closure_i(i, closure::AbstractTurbulenceClosure) = closure

@inline get_closure_ij(i, j, closure::AbstractMatrix{<:AbstractTurbulenceClosure}) = @inbounds closure[i, j]
@inline get_closure_ij(i, j, closure::AbstractTurbulenceClosure) = closure

# Fallbacks
validate_closure(closure) = closure

const ClosureKinda = Union{Nothing, AbstractTurbulenceClosure, AbstractArray{<:AbstractTurbulenceClosure}}
add_closure_specific_boundary_conditions(closure::ClosureKinda, bcs, args...) = bcs

# To allow indexing a diffusivity with (Lx, Ly, Lz, i, j, k, grid)
struct DiscreteDiffusionFunction{F} <: Function
    func :: F
end

#####
##### Include module code
#####

include("implicit_explicit_time_discretization.jl")
include("turbulence_closure_utils.jl")
include("diffusion_operators.jl")
include("viscous_dissipation_operators.jl")
include("velocity_tracer_gradients.jl")
include("abstract_scalar_diffusivity_closure.jl")
include("abstract_eddy_viscosity_closure.jl")
include("abstract_scalar_biharmonic_diffusivity_closure.jl")
include("closure_tuples.jl")
include("isopycnal_rotation_tensor_components.jl")

# Implementations:
include("turbulence_closure_implementations/nothing_closure.jl")
include("turbulence_closure_implementations/scalar_diffusivity.jl")
include("turbulence_closure_implementations/scalar_biharmonic_diffusivity.jl")
include("turbulence_closure_implementations/leith_enstrophy_diffusivity.jl")
include("turbulence_closure_implementations/isopycnal_skew_symmetric_diffusivity.jl")
include("turbulence_closure_implementations/smagorinsky_lilly.jl")
include("turbulence_closure_implementations/anisotropic_minimum_dissipation.jl")
include("turbulence_closure_implementations/CATKEVerticalDiffusivities/CATKEVerticalDiffusivities.jl")
include("turbulence_closure_implementations/convective_adjustment_vertical_diffusivity.jl")

using .CATKEVerticalDiffusivities: CATKEVerticalDiffusivity

# Miscellaneous utilities
include("diffusivity_fields.jl")
include("turbulence_closure_diagnostics.jl")
include("vertically_implicit_diffusion_solver.jl")

#####
##### Some value judgements here
#####

end # module
