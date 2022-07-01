if haskey(ENV, "CI") && ENV["CI"] == "true"
    ENV["PYTHON"] = ""
    using Pkg
    Pkg.build("PyCall")
end

using Test
using CUDA
using PyPlot
using LaTeXStrings

using Oceananigans
using Oceananigans.Advection
using Oceananigans.Advection: boundary_buffer, MultiDimensionalScheme

using ConvergenceTests
using ConvergenceTests.TwoDimensionalGaussianAdvectionDiffusion: run_test
using ConvergenceTests.TwoDimensionalUtils: unpack_errors, defaultcolors, removespines

""" Run advection test for all Nx in resolutions. """
function run_convergence_test(κ, U, resolutions, advection_scheme, arch)

    # Determine safe time-step
           Lx = 2.5
            h = Lx / maximum(resolutions)
           Δt = min(0.01 * h / U, 0.1 * h^2 / κ)
    stop_time = Δt

    # Run the tests
    results = [run_test(architecture=arch, Nx=Nx, Δt=Δt, advection=advection_scheme,
                        stop_iteration=1, U=U, κ=κ) for Nx in resolutions]

    return results
end

#####
##### Run tests
#####

arch = CUDA.has_cuda() ? GPU() : CPU()

advection_schemes = (WENO(order=3), WENO(order=5), WENO(order=7), WENO(order=9), WENO(order=11))

U = 1
κ = 1e-8
Nx = [16, 32, 64, 96, 128, 192, 256] 

results = Dict()
for scheme in advection_schemes
    t_scheme = typeof(scheme)
    results[t_scheme] = run_convergence_test(κ, U, Nx, MultiDimensionalScheme(scheme), arch)
end

rate_of_convergence_1D(::Centered{K}) where K = 2
rate_of_convergence_1D(::UpwindBiased{K}) where K = 2
rate_of_convergence_1D(::WENO{K}) where K = 2

rate_of_convergence_2D(::Centered{K}) where K = 4
rate_of_convergence_2D(::UpwindBiased{K}) where K = 4
rate_of_convergence_2D(::WENO{K}) where K = 4

test_resolution(a) = 256
tolerance(a) = 100.0

colors = ("xkcd:royal blue", "xkcd:light red")

for scheme in advection_schemes

    t_scheme = typeof(scheme)
    name = string(t_scheme.name.wrapper) * "$(boundary_buffer(scheme))"

    @testset "$name" begin

        fig, ax = subplots()

        roc1D = rate_of_convergence(scheme)
        roc2D = rate_of_convergence(scheme)
        atol  = tolerance(scheme)
        Ntest = test_resolution(scheme)
        itest = searchsortedfirst(Nx, Ntest)
        
        (cxy_L₁, cyz_L₁, cxz_L₁, uyz_L₁, vxz_L₁, wxy_L₁, cxy_L∞, cyz_L∞, cxz_L∞, uyz_L∞, vxz_L∞, wxy_L∞) = unpack_errors(results[typeof(scheme)])

        common_kwargs = (linestyle="None", color=colors[1], mfc="None", alpha=0.8)

        loglog(Nx, cxy_L₁; marker="*", label="\$L_1\$-norm, \$c(x)\$ $name", common_kwargs...)
        loglog(Nx, cyz_L₁; marker="+", label="\$L_1\$-norm, \$c(y)\$ $name", common_kwargs...)
        loglog(Nx, cxz_L₁; marker="_", label="\$L_1\$-norm, \$c(z)\$ $name", common_kwargs...)

        loglog(Nx, uyz_L₁; marker="1", label="\$L_1\$-norm, \$u(y)\$ $name", common_kwargs...)
        
        loglog(Nx, vxz_L₁; marker="s", label="\$L_1\$-norm, \$v(x)\$ $name", common_kwargs...)
        
        loglog(Nx, wxy_L₁; marker="X", label="\$L_1\$-norm, \$w(x)\$ $name", common_kwargs...)
        
        common_kwargs = (linestyle="None", color=colors[2], mfc="None", alpha=0.8)

        loglog(Nx, cxy_L∞; marker="*", label="\$L_\\infty\$-norm, \$c(x)\$ $name", common_kwargs...)
        loglog(Nx, cyz_L∞; marker="+", label="\$L_\\infty\$-norm, \$c(y)\$ $name", common_kwargs...)
        loglog(Nx, cxz_L∞; marker="_", label="\$L_\\infty\$-norm, \$c(z)\$ $name", common_kwargs...)

        loglog(Nx, uyz_L∞; marker="1", label="\$L_\\infty\$-norm, \$u(y)\$ $name", common_kwargs...)
        
        loglog(Nx, vxz_L∞; marker="s", label="\$L_\\infty\$-norm, \$v(x)\$ $name", common_kwargs...)
        
        loglog(Nx, wxy_L∞; marker="X", label="\$L_\\infty\$-norm, \$w(x)\$ $name", common_kwargs...)

        label = raw"\sim N_x^{-" * "$roc1D" * raw"}" |> latexstring

        loglog(Nx[itest-3:itest], uyz_L₁[itest] .* (Nx[itest] ./ Nx[itest-3:itest]) .^ roc1D, color=colors[1], alpha=0.8, label=label)

        label = raw"\sim N_x^{-" * "$roc2D" * raw"}" |> latexstring

        loglog(Nx[itest-3:itest], uyz_L₁[itest] .* (Nx[itest] ./ Nx[itest-3:itest]) .^ roc2D, color=colors[1], alpha=0.8, label=label)

        xscale("log", base=2)
        yscale("log", base=10)
        title("Convergence for $name advection scheme")
        xlabel(L"N_x")
        ylabel("\$L\$-norms of \$ | c_\\mathrm{sim} - c_\\mathrm{analytical} |\$")
        removespines("top", "right")
        lgd = legend(loc="lower left", prop=Dict(:size=>6))

        filename = "one_dimensional_convergence_$(name)_$(typeof(arch)).png"
        filepath = joinpath(@__DIR__, "figs", filename)
        mkpath(dirname(filepath))
        savefig(filepath, dpi=240, bbox_extra_artists=(lgd,), bbox_inches="tight")
    end
end
