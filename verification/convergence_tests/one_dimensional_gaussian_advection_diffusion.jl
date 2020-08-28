# "Gaussian advection-diffusion" Spatial resolution convergence test

using PyPlot

# Define a few utilities for running tests and unpacking and plotting results
include("ConvergenceTests/ConvergenceTests.jl")

using .ConvergenceTests.OneDimensionalGaussianAdvectionDiffusion: run_test
using .ConvergenceTests.OneDimensionalUtils: plot_solutions!, plot_error_convergence!

""" Run advection-diffusion test for all Nx in resolutions. """
function run_convergence_test(κ, U, resolutions...)

    # Determine save time-step
           Lx = 2.5
    stop_time = 0.25
            h = Lx / maximum(resolutions)
           Δt = min(0.1 * h / U, 0.01 * h^2 / κ)

    # Adjust time-step
    stop_iteration = round(Int, stop_time / Δt)
                Δt = stop_time / stop_iteration

    # Run the tests
    results = [run_test(Nx=Nx, Δt=Δt, stop_iteration=stop_iteration, U=U, κ=κ, width=0.1) for Nx in resolutions]

    return results
end

#####
##### Run test
#####

Nx = 2 .^ (6:8) # N = 64 through N = 256
diffusion_results = run_convergence_test(1e-1, 0, Nx...)
advection_results = run_convergence_test(1e-6, 3, Nx...)
advection_diffusion_results = run_convergence_test(1e-2, 1, Nx...)

#####
##### Plot solution and error profile
#####

all_results = (diffusion_results, advection_results, advection_diffusion_results)
names = ("diffusion only",  "advection only",  "advection-diffusion")
linestyles = ("-", "--", ":")
specialcolors = ("xkcd:black", "xkcd:indigo", "xkcd:wine red")

# Solutions
close("all")

fig, axs = subplots(nrows=2, figsize=(12, 6), sharex=true)

legends = ConvergenceTests.OneDimensionalUtils.plot_solutions!(axs, all_results, names, linestyles, specialcolors)

savefig("figs/gaussian_advection_diffusion_solutions.png", dpi=480,
        bbox_extra_artists=legends, bbox_inches="tight")

# Error profile
fig, axs = subplots()

legend = ConvergenceTests.OneDimensionalUtils.plot_error_convergence!(axs, Nx, all_results, names)

savefig("figs/gaussian_advection_diffusion_error_convergence.png", dpi=480,
        bbox_extra_artists=(legend,), bbox_inches="tight")
