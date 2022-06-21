module Advection

export 
    div_𝐯u, div_𝐯v, div_𝐯w, div_Uc,

    momentum_flux_uu,
    momentum_flux_uv,
    momentum_flux_uw,
    momentum_flux_vu,
    momentum_flux_vv,
    momentum_flux_vw,
    momentum_flux_wu,
    momentum_flux_wv,
    momentum_flux_ww,
    advective_tracer_flux_x,
    advective_tracer_flux_y,
    advective_tracer_flux_z,

    AdvectionScheme,
    Centered, CenteredSecondOrder, CenteredFourthOrder,
    UpwindBiased, UpwindBiasedFirstOrder, UpwindBiasedThirdOrder, UpwindBiasedFifthOrder,
    WENO, WENO3, WENO5, WENO7, WENO9, WENO11,
    VectorInvariant,
    EnergyConservingScheme,
    EnstrophyConservingScheme

using DocStringExtensions

using Base: @propagate_inbounds
using Adapt 
using OffsetArrays
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Grids
using Oceananigans.Grids: with_halo, return_metrics
using Oceananigans.Architectures: arch_array, architecture, CPU

using Oceananigans.Operators

import Base: show, summary
import Oceananigans.Grids: required_halo_size

abstract type AbstractAdvectionScheme{Buffer} end
abstract type AbstractCenteredAdvectionScheme{Buffer} <: AbstractAdvectionScheme{Buffer} end
abstract type AbstractUpwindBiasedAdvectionScheme{Buffer} <: AbstractAdvectionScheme{Buffer} end

@inline boundary_buffer(::AbstractAdvectionScheme{N}) where N = N
@inline required_halo_size(scheme::AbstractAdvectionScheme{Buffer}) where Buffer = Buffer 

include("centered_advective_fluxes.jl")
include("upwind_biased_advective_fluxes.jl")
include("flat_advective_fluxes.jl")

include("reconstruction_coefficients.jl")
include("centered_reconstruction.jl")
include("upwind_biased_reconstruction.jl")
include("weno_reconstruction.jl")
include("weno_interpolants.jl")
include("stretched_weno_smoothness.jl")
include("vector_invariant_advection.jl")

include("topologically_conditional_interpolation.jl")

include("momentum_advection_operators.jl")
include("tracer_advection_operators.jl")
include("positivity_preserving_tracer_advection_operators.jl")

end # module
