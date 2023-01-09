module TurbulenceClosures

export
    AbstractEddyViscosityClosure,
    VerticalScalarDiffusivity,
    HorizontalScalarDiffusivity,
    HorizontalDivergenceScalarDiffusivity,
    ScalarDiffusivity,
    VerticalScalarBiharmonicDiffusivity,
    HorizontalScalarBiharmonicDiffusivity,
    HorizontalDivergenceScalarBiharmonicDiffusivity,
    ScalarBiharmonicDiffusivity,
    TwoDimensionalLeith,
    SmagorinskyLilly,
    AnisotropicMinimumDissipation,
    ConvectiveAdjustmentVerticalDiffusivity,
    RiBasedVerticalDiffusivity,
    IsopycnalSkewSymmetricDiffusivity,
    FluxTapering,

    ExplicitTimeDiscretization,
    VerticallyImplicitTimeDiscretization,

    DiffusivityFields,
    calculate_diffusivities!,

    ∇_dot_qᶜ,
    ∂ⱼ_τ₁ⱼ,
    ∂ⱼ_τ₂ⱼ,
    ∂ⱼ_τ₃ⱼ,

    cell_diffusion_timescale

using CUDA
using KernelAbstractions
using Adapt 

import Oceananigans.Utils: with_tracers, prettysummary

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

# Fallbacks
validate_closure(closure) = closure
closure_summary(closure) = summary(closure)
with_tracers(tracers, closure::AbstractTurbulenceClosure) = closure
calculate_diffusivities!(K, closure::AbstractTurbulenceClosure, args...) = nothing

const ClosureKinda = Union{Nothing, AbstractTurbulenceClosure, AbstractArray{<:AbstractTurbulenceClosure}}
add_closure_specific_boundary_conditions(closure::ClosureKinda, bcs, args...) = bcs

#####
##### The magic
#####

# Closure ensemble util
@inline getclosure(i, j, closure::AbstractMatrix{<:AbstractTurbulenceClosure}) = @inbounds closure[i, j]
@inline getclosure(i, j, closure::AbstractVector{<:AbstractTurbulenceClosure}) = @inbounds closure[i]
@inline getclosure(i, j, closure::AbstractTurbulenceClosure) = closure

@inline surface(i, j, k, grid)                = znode(i, j, grid.Nz+1, grid, Center(), Center(), Face())
@inline bottom(i, j, k, grid)                 = znode(i, j, 1,         grid, Center(), Center(), Face())
@inline depthᶜᶜᶠ(i, j, k, grid)               = surface(i, j, k, grid) - znode(i, j, k, grid, Center(), Center(), Face())
@inline height_above_bottomᶜᶜᶠ(i, j, k, grid) = znode(i, j, k, grid, Center(), Center(), Face()) - bottom(i, j, k, grid)
@inline wall_vertical_distanceᶜᶜᶠ(i, j, k, grid) = min(depthᶜᶜᶠ(i, j, k, grid), height_above_bottomᶜᶜᶠ(i, j, k, grid))

include("discrete_diffusion_function.jl")
include("implicit_explicit_time_discretization.jl")
include("turbulence_closure_utils.jl")
include("closure_kernel_operators.jl")
include("velocity_tracer_gradients.jl")
include("abstract_scalar_diffusivity_closure.jl")
include("abstract_scalar_biharmonic_diffusivity_closure.jl")
include("closure_tuples.jl")
include("isopycnal_rotation_tensor_components.jl")

# Implicit closure terms (diffusion + linear terms)
include("vertically_implicit_diffusion_solver.jl")

# Implementations:
include("turbulence_closure_implementations/nothing_closure.jl")

# AbstractScalarDiffusivity closures:
include("turbulence_closure_implementations/scalar_diffusivity.jl")
include("turbulence_closure_implementations/scalar_biharmonic_diffusivity.jl")
include("turbulence_closure_implementations/smagorinsky_lilly.jl")
include("turbulence_closure_implementations/anisotropic_minimum_dissipation.jl")
include("turbulence_closure_implementations/convective_adjustment_vertical_diffusivity.jl")
include("turbulence_closure_implementations/CATKEVerticalDiffusivities/CATKEVerticalDiffusivities.jl")
include("turbulence_closure_implementations/ri_based_vertical_diffusivity.jl")

# Special non-abstracted diffusivities:
# TODO: introduce abstract typing for these
include("turbulence_closure_implementations/isopycnal_skew_symmetric_diffusivity.jl")
include("turbulence_closure_implementations/leith_enstrophy_diffusivity.jl")

using .CATKEVerticalDiffusivities: CATKEVerticalDiffusivity

# Miscellaneous utilities
include("diffusivity_fields.jl")
include("turbulence_closure_diagnostics.jl")

#####
##### Some value judgements here
#####

end # module
