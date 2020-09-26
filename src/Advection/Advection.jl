module Advection

export 
    div_ũu, div_ũv, div_ũw, div_uc,

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
    UpwindBiasedThirdOrder,
    CenteredFourthOrder,
    WENO, WENO5

using Oceananigans.Operators

abstract type AbstractAdvectionScheme end
abstract type AbstractCenteredAdvectionScheme <: AbstractAdvectionScheme end
abstract type AbstractUpwindBiasedAdvectionScheme <: AbstractAdvectionScheme end

include("topologically_conditional_interpolation.jl")

include("centered_advective_fluxes.jl")
include("upwind_biased_advective_fluxes.jl")

include("centered_second_order.jl")
include("upwind_biased_third_order.jl")
include("centered_fourth_order.jl")

include("weno_reconstruction.jl")
include("weno.jl")
include("weno5.jl")

include("momentum_advection_operators.jl")
include("tracer_advection_operators.jl")

end # module
