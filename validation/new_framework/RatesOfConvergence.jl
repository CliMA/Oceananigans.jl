module RatesOfConvergence

using Oceananigans.Advection

using Printf

export one_time_step!
export Forwardeuler, AdamsBashforth2

export advective_flux
export rate_of_convergence
export labels
export shapes
export colors
export UpwindBiasedFirstOrder, CenteredSixthOrder

export plot_solutions!

struct ForwardEuler end
struct AdamsBashforth2 end

#struct UpwindBiasedFirstOrder <: AbstractUpwindBiasedAdvectionScheme end
#struct CenteredSixthOrder     <: AbstractCenteredAdvectionScheme end

struct UpwindBiasedFirstOrder  end
struct CenteredSixthOrder      end

include("time_stepping.jl")
include("advection_schemes.jl")
include("plot_convergence.jl")

end # module
