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
using ConvergenceTests.TwoDimensionalVortexAdvection: run_test
using ConvergenceTests.TwoDimensionalVortexAdvection: unpack_errors, defaultcolors, removespines

""" Run advection test for all Nx in resolutions. """
function run_convergence_test(resolutions, order, arch)

    # Determine safe time-step
           Lx = 10
            h = Lx / maximum(resolutions)
           Δt = min(0.01 * h / U, 0.1)

    # Run the tests
    results = [run_test(architecture=arch, Nx=Nx, Δt=Δt, order = order,
                        stop_iteration=1, U=U) for Nx in resolutions]

    return results
end

#####
##### Run tests
#####

arch = CUDA.has_cuda() ? GPU() : CPU()

advection_schemes = (WENO(order=3), WENO(order=5), WENO(order=7), WENO(order=9), WENO(order=11))

U = 1
Nx = [16, 32, 64, 96, 128, 192, 256] 

results = Dict()
for scheme in advection_schemes
    t_scheme = typeof(scheme)
    results[t_scheme] = run_convergence_test(Nx, 2*boundary_buffer(scheme) - 1, arch)
end

rate_of_convergence_1D(::Centered{K}) where K = 2
rate_of_convergence_1D(::UpwindBiased{K}) where K = 2
rate_of_convergence_1D(::WENO{K}) where K = 2

rate_of_convergence_2D(::Centered{K}) where K = 2K
rate_of_convergence_2D(::UpwindBiased{K}) where K = 2K-1
rate_of_convergence_2D(::WENO{K}) where K = 2K-1

test_resolution(a) = 256
tolerance(a) = 100.0

colors = ("xkcd:royal blue", "xkcd:light red")

for scheme in advection_schemes

    t_scheme = typeof(scheme)
    name = string(t_scheme.name.wrapper) * "$(boundary_buffer(scheme))"

    @testset "$name" begin

        fig, ax = subplots()

        roc1D = rate_of_convergence_1D(scheme)
        roc2D = rate_of_convergence_2D(scheme)
        atol  = tolerance(scheme)
        Ntest = test_resolution(scheme)
        itest = searchsortedfirst(Nx, Ntest)
        
        (uvi_L₁, vvi_L₁, hvi_L₁, ucf_L₁, vcf_L₁, hcf_L₁, uvi_L∞, vvi_L∞, hvi_L∞, ucf_L∞, vcf_L∞, hcf_L∞) = unpack_errors(results[typeof(scheme)])

        common_kwargs = (linestyle="None", color=colors[1], mfc="None", alpha=0.8)

        loglog(Nx, ucf_L₁; marker="*", label="\$L_1\$-norm, \$ucf\$ $name", common_kwargs...)
        loglog(Nx, vcf_L₁; marker="+", label="\$L_1\$-norm, \$vcf\$ $name", common_kwargs...)
        loglog(Nx, hcf_L₁; marker="_", label="\$L_1\$-norm, \$hcf\$ $name", common_kwargs...)

        loglog(Nx, ucf_L₁; marker="1", label="\$L_1\$-norm, \$ucf\$ $name", common_kwargs...)
        loglog(Nx, vcf_L₁; marker="s", label="\$L_1\$-norm, \$vcf\$ $name", common_kwargs...)
        loglog(Nx, hcf_L₁; marker="X", label="\$L_1\$-norm, \$hcf\$ $name", common_kwargs...)
        
        common_kwargs = (linestyle="None", color=colors[2], mfc="None", alpha=0.8)

        loglog(Nx, uvi_L∞; marker="*", label="\$L_\\infty\$-norm, \$uvi\$ $name", common_kwargs...)
        loglog(Nx, vvi_L∞; marker="+", label="\$L_\\infty\$-norm, \$vvi\$ $name", common_kwargs...)
        loglog(Nx, hvi_L∞; marker="_", label="\$L_\\infty\$-norm, \$hvi\$ $name", common_kwargs...)

        loglog(Nx, uvi_L∞; marker="1", label="\$L_\\infty\$-norm, \$uvi\$ $name", common_kwargs...)
        loglog(Nx, vvi_L∞; marker="s", label="\$L_\\infty\$-norm, \$vvi\$ $name", common_kwargs...)
        loglog(Nx, hvi_L∞; marker="X", label="\$L_\\infty\$-norm, \$hvi\$ $name", common_kwargs...)

        label = raw"\sim N_x^{-" * "$roc1D" * raw"}" |> latexstring

        loglog(Nx[itest-3:itest], uvi_L₁[itest] .* (Nx[itest] ./ Nx[itest-3:itest]) .^ roc1D, color=colors[1], alpha=0.8, label=label)

        label = raw"\sim N_x^{-" * "$roc2D" * raw"}" |> latexstring

        loglog(Nx[itest-3:itest], uvi_L₁[itest] .* (Nx[itest] ./ Nx[itest-3:itest]) .^ roc2D, color=colors[1], alpha=0.8, label=label)

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
