# # "Point exponential decay" time-stepper convergence test

using PyPlot

# Define a few utilities for running tests and unpacking and plotting results

include("ConvergenceTests/ConvergenceTests.jl") # we use the GaussianAdvectionDiffusion module here.

defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
removespine(side) = gca().spines[side].set_visible(false)
removespines(sides...) = [removespine(side) for side in sides]

run_single_test = ConvergenceTests.PointExponentialDecay.run_test

""" Run advection-diffusion test for all Nx in resolutions. """
function run_convergence_test(stop_time, proposal_Δt...)

    # Adjust time-step
    stop_iterations = [round(Int, stop_time / dt) for dt in proposal_Δt]
                 Δt = [stop_time / stop_iter for stop_iter in stop_iterations]

    # Run the tests
    results = [run_single_test(Δt=dt, stop_iteration=stop_iter)
               for (dt, stop_iter) in zip(Δt, stop_iterations)]

    return results, Δt
end

unpack_errors(results) = map(r -> r.L₁, results)

stop_time = 3
Δt = vcat(1 ./ collect(1000:-100:100), 1 ./ collect(90:-10:10), 1 ./ collect(9:-1:1))
results, Δt = run_convergence_test(stop_time, Δt...)
L₁ = unpack_errors(results)

fig, axs = subplots()

loglog(Δt, L₁, "o--", alpha=0.8, linewidth=1, label="Measured error")

# Guide line to confirm second-order scaling
loglog(Δt, L₁[1] .* Δt / Δt[1], "k-", alpha=0.4, linewidth=2, label=L"\sim \Delta t")

xlabel(L"\Delta t")
ylabel("\$L_1\$-norm of \$ | c_\\mathrm{simulated}(t) - \\mathrm{e}^{-t} | \$")
title("Oceananigans time-stepper convergence for \$ c(t) = \\mathrm{e}^{-t} \$")
removespines("top", "right")
legend()

savefig("figs/point_exponential_decay_time_stepper_convergence.png", dpi=480)
