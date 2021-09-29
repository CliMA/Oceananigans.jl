# Exponential decay at a point
#
# We set up a problem in which a tracer obeys
#
# ∂c/∂t = - c
#
# at a single point. This leads to exponential decay,
#
# c(t) = exp(-t)
#
# in the case that c(0) = 1. We use this problem to test
# the order of convergence of the time-stepper.

module PointExponentialDecay

using Printf
using Statistics
using Oceananigans

include("analysis.jl")

# Simple Exponential decay
c(t) = exp(-t)
Fᶜ = Relaxation(rate=1)

function run_test(; Δt, stop_iteration, timestepper, architecture = CPU())

    grid = RegularRectilinearGrid(size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))

    model = NonhydrostaticModel(architecture = architecture,
                                 timestepper = timestepper,
                                        grid = grid,
                                    coriolis = nothing,
                                    buoyancy = nothing,
                                     tracers = :c,
                                     forcing = (c=Fᶜ,))

    set!(model, c=1)
    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, iteration_interval=stop_iteration)

    @info "Running point exponential decay test for Δt = $Δt, stop_iteration = $stop_iteration..."
    run!(simulation)

    # Calculate errors
    c_simulation = model.tracers.c[1, 1, 1]
    c_analytical = c(model.clock.time)
    L₁_error = abs(c_simulation - c_analytical)

    return (simulation = c_simulation,
            analytical = c_analytical,
                    L₁ = L₁_error)
end

end # module
