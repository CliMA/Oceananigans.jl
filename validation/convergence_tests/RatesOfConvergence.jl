module RatesOfConvergence

using Oceananigans.Advection

using Printf

export rate_of_convergence, labels, shapes, colors, halos, plot_solutions!

include("advection_schemes.jl")
include("plot_convergence.jl")

end # module
