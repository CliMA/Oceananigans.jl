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

    CenteredSecondOrder,
    UpwindBiasedFirstOrder,
    UpwindBiasedThirdOrder,
    UpwindBiasedFifthOrder,
    CenteredFourthOrder,
    WENO5,
    VectorInvariant,
    EnergyConservingScheme,
    EnstrophyConservingScheme

using DocStringExtensions

using Oceananigans.Grids
using Oceananigans.Operators

import Oceananigans.Grids: required_halo_size

abstract type AbstractAdvectionScheme{Buffer} end
abstract type AbstractCenteredAdvectionScheme{Buffer} <: AbstractAdvectionScheme{Buffer} end
abstract type AbstractUpwindBiasedAdvectionScheme{Buffer} <: AbstractAdvectionScheme{Buffer} end

required_halo_size(scheme::AbstractAdvectionScheme{Buffer}) where Buffer = Buffer + 1

include("topologically_conditional_interpolation.jl")

include("centered_advective_fluxes.jl")
include("upwind_biased_advective_fluxes.jl")
include("flat_advective_fluxes.jl")

include("upwind_biased_first_order.jl")
include("centered_second_order.jl")
include("upwind_biased_third_order.jl")
include("centered_fourth_order.jl")
include("upwind_biased_fifth_order.jl")
include("weno_fifth_order.jl")

include("momentum_advection_operators.jl")
include("tracer_advection_operators.jl")
include("vector_invariant_advection.jl")

end # module
