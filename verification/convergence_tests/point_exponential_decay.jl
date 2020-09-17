# # "Point exponential decay" time-stepper convergence test

using PyPlot
using Oceananigans

# Define a few utilities for running tests and unpacking and plotting results
include("ConvergenceTests/ConvergenceTests.jl")

using .ConvergenceTests
using .ConvergenceTests.PointExponentialDecay: run_test

defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
removespine(side) = gca().spines[side].set_visible(false)
removespines(sides...) = [removespine(side) for side in sides]

""" Run advection-diffusion test for all Nx in resolutions. """
function run_convergence_test(stop_time, proposal_Δt, timestepper)

    # Adjust time-step
    stop_iterations = [round(Int, stop_time / dt) for dt in proposal_Δt]
                 Δt = [stop_time / stop_iter for stop_iter in stop_iterations]

    # Run the tests
    results = [run_test(timestepper=timestepper, Δt=dt, stop_iteration=stop_iter)
               for (dt, stop_iter) in zip(Δt, stop_iterations)]

    return results, Δt
end

unpack_errors(results) = map(r -> r.L₁, results)

stop_time = 3
Δt = 10 .^ range(-3, 0, length=30)  # Equally spaced in log space.
ab2_results, Δt = run_convergence_test(stop_time, Δt, :QuasiAdamsBashforth2)
rk3_results, Δt = run_convergence_test(stop_time, Δt, :RungeKutta3)

ab2_L₁ = unpack_errors(ab2_results)
rk3_L₁ = unpack_errors(rk3_results)

fig, axs = subplots()

loglog(Δt, ab2_L₁, "o--", alpha=0.8, linewidth=1, label="Measured error, Quasi-second-order Adams-Bashforth")
loglog(Δt, rk3_L₁, "^-.", alpha=0.8, linewidth=1, label="Measured error, Third-order Runge-Kutta")

# Guide line to confirm second-order scaling
loglog(Δt, ab2_L₁[1] .* Δt / Δt[1], "k-", alpha=0.4, linewidth=2, label=L"\sim \Delta t")
loglog(Δt, rk3_L₁[1] .* (Δt / Δt[1]).^3, "r-", alpha=0.4, linewidth=2, label=L"\sim \Delta t^3")

xlabel(L"\Delta t")
ylabel("\$L_1\$-norm of \$ | c_\\mathrm{simulated}(t) - \\mathrm{e}^{-t} | \$")
title("Oceananigans time-stepper convergence for \$ c(t) = \\mathrm{e}^{-t} \$")
removespines("top", "right")
legend()

filepath = joinpath(@__DIR__, "figs", "point_exponential_decay_time_stepper_convergence.png")
savefig(filepath, dpi=480)

test_rate_of_convergence(ab2_L₁, Δt, data_points=9, expected=1.0, atol=0.01)
test_rate_of_convergence(rk3_L₁, Δt, data_points=9, expected=3.0, atol=0.01)
