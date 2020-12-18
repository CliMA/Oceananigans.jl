module RatesOfConvergence

export Time_Stepper
export advective_flux
export rate_of_convergence
export labels
export shapes
export colors

export UpwindBiasedFirstOrder, CenteredSixthOrder

struct UpwindBiasedFirstOrder  end
struct CenteredSixthOrder      end

include("time_stepping.jl")

include("advection_schemes.jl")

include("plotting_convergence.jl")

end # module
