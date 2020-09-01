# "Cosine advection-diffusion" Spatial resolution convergence test

using Test
using PyPlot
using Oceananigans.Grids

# Define a few utilities for running tests and unpacking and plotting results
include("ConvergenceTests/ConvergenceTests.jl")

using .ConvergenceTests
using .ConvergenceTests.OneDimensionalCosineAdvectionDiffusion: run_test
using .ConvergenceTests.OneDimensionalUtils: plot_solutions!, plot_error_convergence!, unpack_errors

""" Run advection-diffusion test for all Nx in resolutions. """
function run_convergence_test(κ, U, resolutions)

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

Nx = 2 .^ (3:7) # N = 8 through N = 256
diffusion_results = run_convergence_test(1e-1, 0, Nx)
advection_results = run_convergence_test(1e-6, 3, Nx)
advection_diffusion_results = run_convergence_test(1e-2, 1, Nx)

#####
##### Plot solution and error profile
#####

all_results = (diffusion_results, advection_results, advection_diffusion_results)
names = ("diffusion only", "advection only", "advection-diffusion")
linestyles = ("-", "--", ":")
specialcolors = ("xkcd:black", "xkcd:indigo", "xkcd:wine red")

# Solutions
close("all")

fig, axs = subplots(nrows=2, figsize=(12, 6), sharex=true)

legends = plot_solutions!(axs, all_results, names, linestyles, specialcolors)

filepath = joinpath(@__DIR__, "figs", "cosine_advection_diffusion_solutions.png")
savefig(filepath, dpi=480, bbox_extra_artists=legends, bbox_inches="tight")

# Error profile
fig, axs = subplots()

legend = plot_error_convergence!(axs, Nx, all_results, names)

filepath = joinpath(@__DIR__, "figs", "cosine_advection_diffusion_error_convergence.png")
savefig(filepath, dpi=480, bbox_extra_artists=(legend,), bbox_inches="tight")

# Test rate of convergence
for (results, name) in zip(all_results, names)
    name = "1D cosine " * name
    @info "Testing rate of convergence for $name..."

    u_L₁, v_L₁, cx_L₁, cy_L₁, u_L∞, v_L∞, cx_L∞, cy_L∞  = unpack_errors(results)

    test_rate_of_convergence(u_L₁,  Nx, expected=-2.0, atol=0.01, name=name*" u_L₁")
    test_rate_of_convergence(v_L₁,  Nx, expected=-2.0, atol=0.01, name=name*" v_L₁")
    test_rate_of_convergence(cx_L₁, Nx, expected=-2.0, atol=0.01, name=name*" cx_L₁")
    test_rate_of_convergence(cy_L₁, Nx, expected=-2.0, atol=0.01, name=name*" cy_L₁")
    test_rate_of_convergence(u_L∞,  Nx, expected=-2.0, atol=0.05, name=name*" u_L∞")
    test_rate_of_convergence(v_L∞,  Nx, expected=-2.0, atol=0.05, name=name*" v_L∞")
    test_rate_of_convergence(cx_L∞, Nx, expected=-2.0, atol=0.05, name=name*" cx_L∞")
    test_rate_of_convergence(cy_L∞, Nx, expected=-2.0, atol=0.05, name=name*" cy_L∞")

    @test u_L₁ ≈ v_L₁ ≈ cx_L₁ ≈ cy_L₁
    @test u_L∞ ≈ v_L∞ ≈ cx_L∞ ≈ cy_L∞
end
