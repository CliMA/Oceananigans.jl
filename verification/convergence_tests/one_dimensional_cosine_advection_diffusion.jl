# "Cosine advection-diffusion" Spatial resolution convergence test

using PyPlot

using Oceananigans.Grids

# Define a few utilities for running tests and unpacking and plotting results

include("ConvergenceTests/ConvergenceTests.jl")

run_test = ConvergenceTests.OneDimensionalCosineAdvectionDiffusion.run_test

""" Run advection-diffusion test for all Nx in resolutions. """
function run_convergence_test(κ, U, resolutions...)

    # Determine save time-step
             Lx = 2π
      stop_time = 0.01
              h = Lx / maximum(resolutions)
    proposal_Δt = 1e-3 * min(h / U, h^2 / κ)

    # Adjust time-step
    stop_iteration = round(Int, stop_time / proposal_Δt)
                Δt = stop_time / stop_iteration

    # Run the tests
    results = [run_test(Nx=Nx, Δt=Δt, stop_iteration=stop_iteration, U=U, κ=κ) for Nx in resolutions]

    return results
end

#####
##### Run test
#####

Nx = 2 .^ (3:7) # N = 64 through N = 256
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

ConvergenceTests.OneDimensionalUtils.plot_solutions!(axs, all_results, names, linestyles, specialcolors)

savefig("figs/cosine_advection_diffusion_solutions.png", dpi=480)

# Error profile
fig, axs = subplots()

ConvergenceTests.OneDimensionalUtils.plot_error_convergence!(axs, Nx, all_results, names)

savefig("figs/cosine_advection_diffusion_error_convergence.png", dpi=480)
