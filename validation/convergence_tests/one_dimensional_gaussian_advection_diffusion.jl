if ENV["CI"] == "true"
    ENV["PYTHON"] = ""
    using Pkg
    Pkg.build("PyCall")
end

using PyPlot
using Oceananigans.Advection

using ConvergenceTests
using ConvergenceTests.OneDimensionalGaussianAdvectionDiffusion: run_test
using ConvergenceTests.OneDimensionalUtils: plot_solutions!, plot_error_convergence!, unpack_errors

""" Run advection-diffusion test for all Nx in resolutions. """
function run_convergence_test(κ, U, resolutions, arch)

    # Determine safe time-step
           Lx = 2.5
    stop_time = 0.25
            h = Lx / maximum(resolutions)
           Δt = min(0.1 * h / U, 0.01 * h^2 / κ)

    # Adjust time-step
    stop_iteration = round(Int, stop_time / Δt)
                Δt = stop_time / stop_iteration

    # Run the tests
    results = [run_test(architecture=arch, Nx=Nx, Δt=Δt, stop_iteration=stop_iteration, U=U, κ=κ, width=0.1)
               for Nx in resolutions]

    return results
end

#####
##### Run test
#####

arch = CUDA.has_cuda() ? GPU() : CPU()

Nx = 2 .^ (6:8) # N = 64 through N = 256
diffusion_results = run_convergence_test(1e-1, 0, Nx, arch)
advection_results = run_convergence_test(1e-6, 3, Nx, arch)
advection_diffusion_results = run_convergence_test(1e-2, 1, Nx, arch)

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

filename = "gaussian_advection_diffusion_solutions_$(typeof(arch)).png"
filepath = joinpath(@__DIR__, "figs", filename)
mkpath(dirname(filepath))
savefig(filepath, dpi=480, bbox_extra_artists=legends, bbox_inches="tight")

# Error profile
fig, axs = subplots()

legend = plot_error_convergence!(axs, Nx, all_results, names)

filename = "gaussian_advection_diffusion_error_convergence_$(typeof(arch)).png"
filepath = joinpath(@__DIR__, "figs", filename)
mkpath(dirname(filepath))
savefig(filepath, dpi=480, bbox_extra_artists=(legend,), bbox_inches="tight")

# Test rate of convergence
for (results, name) in zip(all_results, names)
    atol_L₁, atol_L∞ =
        name == "diffusion only" ? (0.05, 0.05) :
        name == "advection only" ? (0.10, 0.50) :
        name == "advection-diffusion" ? (0.05, 0.01) : nothing

    name = "1D Gaussian " * name
    @info "Testing rate of convergence for $name..."

    u_L₁, v_L₁, cx_L₁, cy_L₁, u_L∞, v_L∞, cx_L∞, cy_L∞  = unpack_errors(results)

    test_rate_of_convergence(u_L₁,  Nx, expected=-2.0, atol=atol_L₁, name=name*" u_L₁")
    test_rate_of_convergence(v_L₁,  Nx, expected=-2.0, atol=atol_L₁, name=name*" v_L₁")
    test_rate_of_convergence(cx_L₁, Nx, expected=-2.0, atol=atol_L₁, name=name*" cx_L₁")
    test_rate_of_convergence(cy_L₁, Nx, expected=-2.0, atol=atol_L₁, name=name*" cy_L₁")
    test_rate_of_convergence(u_L∞,  Nx, expected=-2.0, atol=atol_L∞, name=name*" u_L∞")
    test_rate_of_convergence(v_L∞,  Nx, expected=-2.0, atol=atol_L∞, name=name*" v_L∞")
    test_rate_of_convergence(cx_L∞, Nx, expected=-2.0, atol=atol_L∞, name=name*" cx_L∞")
    test_rate_of_convergence(cy_L∞, Nx, expected=-2.0, atol=atol_L∞, name=name*" cy_L∞")

    @test u_L₁ ≈ v_L₁ ≈ cx_L₁ ≈ cy_L₁
    @test u_L∞ ≈ v_L∞ ≈ cx_L∞ ≈ cy_L∞
end
