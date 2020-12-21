module RatesOfConvergence

using Oceananigans.Advection

using Printf

export ForwardEuler, AdamsBashforth2
export one_time_step!

export UpwindBiasedFirstOrder, CenteredSixthOrder
export advective_flux
export rate_of_convergence
export labels
export shapes
export colors

export plot_solutions!

struct ForwardEuler end
struct AdamsBashforth2 end

# From Advection.jl
#abstract type AbstractAdvectionScheme end
#abstract type AbstractCenteredAdvectionScheme <: AbstractAdvectionScheme end
#abstract type AbstractUpwindBiasedAdvectionScheme <: AbstractAdvectionScheme end

#struct UpwindBiasedFirstOrder <: AbstractUpwindBiasedAdvectionScheme end
#struct CenteredSixthOrder     <: AbstractCenteredAdvectionScheme end

struct UpwindBiasedFirstOrder  end
struct CenteredSixthOrder      end

include("time_stepping.jl")
include("advection_schemes.jl")
include("plot_convergence.jl")

end # module
