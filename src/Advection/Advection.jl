module Advection

export 
    div_uc, div_ũu, div_ũv, div_ũw,

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
    CenteredFourthOrder,
    WENO, WENO5

using Oceananigans.Operators

abstract type AbstractAdvectionScheme end

include("centered_second_order.jl")
include("centered_fourth_order.jl")
include("weno_reconstruction.jl")
include("weno.jl")
include("weno5.jl")
include("momentum_advection_operators.jl")
include("tracer_advection_operators.jl")

end # module
