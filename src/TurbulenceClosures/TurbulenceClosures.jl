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
##### Molecular viscosity and thermal diffusivity definitions
##### Approximate viscosities and thermal diffusivities for seawater at 20ᵒC and 35 psu,
##### according to Sharqawy et al., "Thermophysical properties of seawater: A review of
##### existing correlations and data" (2010).
#####

const ν₀ = 1.05e-6
const κ₀ = 1.46e-7

#####
##### Abstract types
#####

"""
    AbstractTurbulenceClosure

Abstract supertype for turbulence closures.
"""
abstract type AbstractTurbulenceClosure end


"""
    AbstractTensorDiffusivity <: AbstractTurbulenceClosure

Abstract supertype for turbulence closures that are defined by a tensor viscosity and
tensor diffusivities.
"""
abstract type AbstractTensorDiffusivity <: AbstractTurbulenceClosure end

#####
##### 'Tupled closure' implementation
#####

for stress_div in (:∂ⱼ_2ν_Σ₁ⱼ, :∂ⱼ_2ν_Σ₂ⱼ, :∂ⱼ_2ν_Σ₃ⱼ)
    @eval begin
        @inline function $stress_div(i, j, k, grid::AbstractGrid{FT}, clock, closure_tuple::Tuple, U,
                                     K_tuple, args...) where FT

            stress_div_ijk = zero(FT)

            ntuple(Val(length(closure_tuple))) do α
                @inbounds closure = closure_tuple[α]
                @inbounds K = K_tuple[α]
                stress_div_ijk += $stress_div(i, j, k, grid, clock, closure, U, K, args...)
            end

            return stress_div_ijk
        end
    end
end

@inline function ∇_κ_∇c(i, j, k, grid::AbstractGrid{FT}, clock, closure_tuple::Tuple,
                        c, tracer_index, K_tuple, args...) where FT

    flux_div_ijk = zero(FT)

    ntuple(Val(length(closure_tuple))) do α
        @inbounds closure = closure_tuple[α]
        @inbounds K = K_tuple[α]
        flux_div_ijk +=  ∇_κ_∇c(i, j, k, grid, clock, closure, c, tracer_index, K, args...)
    end

    return flux_div_ijk
end

function calculate_diffusivities!(K_tuple::Tuple, arch, grid, closure_tuple::Tuple, args...)
    ntuple(Val(length(closure_tuple))) do α
        @inbounds closure = closure_tuple[α]
        @inbounds K = K_tuple[α]
        calculate_diffusivities!(K, arch, grid, closure, args...)
    end

    return nothing
end

with_tracers(tracers, closure_tuple::Tuple) =
    Tuple(with_tracers(tracers, closure) for closure in closure_tuple)

#####
##### Include module code
#####

include("turbulence_closure_utils.jl")
include("diffusion_operators.jl")
include("viscous_dissipation_operators.jl")
include("velocity_tracer_gradients.jl")
include("abstract_eddy_viscosity_closures.jl")

include("closure_tuples.jl")

include("turbulence_closure_implementations/nothing_closure.jl")
include("turbulence_closure_implementations/isotropic_diffusivity.jl")
include("turbulence_closure_implementations/anisotropic_diffusivity.jl")
include("turbulence_closure_implementations/horizontally_curvilinear_anisotropic_diffusivity.jl")
include("turbulence_closure_implementations/horizontally_curvilinear_anisotropic_biharmonic_diffusivity.jl")
include("turbulence_closure_implementations/anisotropic_biharmonic_diffusivity.jl")

# Eddy viscosity closures
include("turbulence_closure_implementations/leith_enstrophy_diffusivity.jl")
include("turbulence_closure_implementations/smagorinsky_lilly.jl")
include("turbulence_closure_implementations/anisotropic_minimum_dissipation.jl")

include("diffusivity_fields.jl")
include("turbulence_closure_diagnostics.jl")

#####
##### Some value judgements here
#####

"""
    ConstantSmagorinsky

An alias for `SmagorinskyLilly`.
"""
const ConstantSmagorinsky = SmagorinskyLilly

end # module
