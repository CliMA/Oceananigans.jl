using Test

using PyPlot
using LaTeXStrings

using Oceananigans.Advection

# Define a few utilities for running tests and unpacking and plotting results
include("ConvergenceTests/ConvergenceTests.jl")

using .ConvergenceTests
using .ConvergenceTests.OneDimensionalGaussianAdvectionDiffusion: run_test
using .ConvergenceTests.OneDimensionalUtils: unpack_errors, defaultcolors, removespines

""" Run advection test for all Nx in resolutions. """
function run_convergence_test(κ, U, resolutions, advection_scheme)

    # Determine safe time-step
           Lx = 2.5
            h = Lx / maximum(resolutions)
           Δt = min(0.01 * h / U, 0.1 * h^2 / κ)
    stop_time = Δt

    # Run the tests
    results = [run_test(Nx=Nx, Δt=Δt, advection=advection_scheme, stop_iteration=1,
                        U=U, κ=κ) for Nx in resolutions]

    return results
end

#####
##### Run test
#####

advection_schemes = (CenteredSecondOrder(), CenteredFourthOrder(), UpwindBiasedThirdOrder(), WENO5())
#advection_schemes = (UpwindBiasedThirdOrder(),)

U = 1
κ = 1e-8
Nx = [8, 16, 32, 64, 96, 128, 192, 256, 384, 512]

tolerance(::CenteredSecondOrder)    = 0.05
tolerance(::CenteredFourthOrder)    = 0.05
tolerance(::UpwindBiasedThirdOrder) = 0.30
tolerance(::WENO5)                  = 0.40

test_resolution(::CenteredSecondOrder)    = 512
test_resolution(::CenteredFourthOrder)    = 512
test_resolution(::UpwindBiasedThirdOrder) = 128
test_resolution(::WENO5)                  = 512

rate_of_convergence(::CenteredSecondOrder) = 2
rate_of_convergence(::CenteredFourthOrder) = 4
rate_of_convergence(::UpwindBiasedThirdOrder) = 3
rate_of_convergence(::WENO5) = 5
rate_of_convergence(::WENO{K}) where K = 2K-1

results = Dict()
for scheme in advection_schemes
    t_scheme = typeof(scheme)
    results[t_scheme] = run_convergence_test(κ, U, Nx, scheme)
end

colors = ("xkcd:royal blue", "xkcd:light red")

@testset "tmp" begin
    for scheme in advection_schemes

        fig, ax = subplots()

        t_scheme = typeof(scheme)
        name = string(t_scheme)
        roc = rate_of_convergence(scheme)
        atol = tolerance(scheme)
        atol = tolerance(scheme)
        Ntest = test_resolution(scheme)
        itest = searchsortedfirst(Nx, Ntest)
        
        (cx_L₁, cy_L₁, cz_L₁, uy_L₁, uz_L₁, vx_L₁, vz_L₁, wx_L₁, wy_L₁,
         cx_L∞, cy_L∞, cz_L∞, uy_L∞, uz_L∞, vx_L∞, vz_L∞, wx_L∞, wy_L∞) = unpack_errors(results[typeof(scheme)])

        test_rate_of_convergence(cx_L₁, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" cx_L₁")
        test_rate_of_convergence(cy_L₁, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" cy_L₁")
        test_rate_of_convergence(cz_L₁, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" cz_L₁")

        test_rate_of_convergence(uy_L₁, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" uy_L₁")
        test_rate_of_convergence(uz_L₁, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" uz_L₁")

        test_rate_of_convergence(vx_L₁, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" vx_L₁")
        test_rate_of_convergence(vz_L₁, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" vz_L₁")

        test_rate_of_convergence(wx_L₁, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" wx_L₁")
        test_rate_of_convergence(wy_L₁, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" wy_L₁")

        #=
        test_rate_of_convergence(cx_L∞, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" cx_L∞")
        test_rate_of_convergence(cy_L∞, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" cy_L∞")
        test_rate_of_convergence(cz_L∞, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" cz_L∞")

        test_rate_of_convergence(uy_L∞, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" uy_L∞")
        test_rate_of_convergence(uz_L∞, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" uz_L∞")

        test_rate_of_convergence(vx_L∞, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" vx_L∞")
        test_rate_of_convergence(vz_L∞, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" vz_L∞")

        test_rate_of_convergence(wx_L∞, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" wx_L∞")
        test_rate_of_convergence(wy_L∞, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" wy_L∞")
        =#

        @test cx_L₁ ≈ cy_L₁
        @test cx_L₁ ≈ cz_L₁
        @test uy_L₁ ≈ uz_L₁
        @test vx_L₁ ≈ vz_L₁
        @test wx_L₁ ≈ wy_L₁

        @test cx_L∞ ≈ cy_L∞
        @test cx_L∞ ≈ cz_L∞
        @test uy_L∞ ≈ uz_L∞
        @test vx_L∞ ≈ vz_L∞
        @test wx_L∞ ≈ wy_L∞

        common_kwargs = (linestyle="None", color=colors[1], mfc="None", alpha=0.8)

        loglog(Nx, cx_L₁; basex=2, marker="*", label="\$L_1\$-norm, \$c(x)\$ $name", common_kwargs...)
        loglog(Nx, cy_L₁; basex=2, marker="+", label="\$L_1\$-norm, \$c(y)\$ $name", common_kwargs...)
        loglog(Nx, cz_L₁; basex=2, marker="_", label="\$L_1\$-norm, \$c(z)\$ $name", common_kwargs...)

        loglog(Nx, uy_L₁; basex=2, marker="1", label="\$L_1\$-norm, \$u(y)\$ $name", common_kwargs...)
        loglog(Nx, uz_L₁; basex=2, marker="^", label="\$L_1\$-norm, \$u(z)\$ $name", common_kwargs...)

        loglog(Nx, vx_L₁; basex=2, marker="s", label="\$L_1\$-norm, \$v(x)\$ $name", common_kwargs...)
        loglog(Nx, vz_L₁; basex=2, marker="v", label="\$L_1\$-norm, \$v(z)\$ $name", common_kwargs...)

        loglog(Nx, wx_L₁; basex=2, marker="X", label="\$L_1\$-norm, \$w(x)\$ $name", common_kwargs...)
        loglog(Nx, wy_L₁; basex=2, marker="D", label="\$L_1\$-norm, \$w(y)\$ $name", common_kwargs...)

        common_kwargs = (linestyle="None", color=colors[2], mfc="None", alpha=0.8)

        loglog(Nx, cx_L∞; basex=2, marker="*", label="\$L_\\infty\$-norm, \$c(x)\$ $name", common_kwargs...)
        loglog(Nx, cy_L∞; basex=2, marker="+", label="\$L_\\infty\$-norm, \$c(y)\$ $name", common_kwargs...)
        loglog(Nx, cz_L∞; basex=2, marker="_", label="\$L_\\infty\$-norm, \$c(z)\$ $name", common_kwargs...)

        loglog(Nx, uy_L∞; basex=2, marker="1", label="\$L_\\infty\$-norm, \$u(y)\$ $name", common_kwargs...)
        loglog(Nx, uz_L∞; basex=2, marker="^", label="\$L_\\infty\$-norm, \$u(z)\$ $name", common_kwargs...)

        loglog(Nx, vx_L∞; basex=2, marker="s", label="\$L_\\infty\$-norm, \$v(x)\$ $name", common_kwargs...)
        loglog(Nx, vz_L∞; basex=2, marker="v", label="\$L_\\infty\$-norm, \$v(z)\$ $name", common_kwargs...)

        loglog(Nx, wx_L∞; basex=2, marker="X", label="\$L_\\infty\$-norm, \$w(x)\$ $name", common_kwargs...)
        loglog(Nx, wy_L∞; basex=2, marker="D", label="\$L_\\infty\$-norm, \$w(y)\$ $name", common_kwargs...)

        label = raw"\sim N_x^{-" * "$roc" * raw"}" |> latexstring

        loglog(Nx[itest-3:itest], uy_L₁[itest] .* (Nx[itest] ./ Nx[itest-3:itest]) .^ roc, color=colors[1], basex=2, alpha=0.8, label=label)

        xlabel(L"N_x")
        ylabel("\$L\$-norms of \$ | c_\\mathrm{sim} - c_\\mathrm{analytical} |\$")
        removespines("top", "right")
        lgd = legend(loc="upper right", bbox_to_anchor=(1.4, 1.0), prop=Dict(:size=>6))

        filepath = joinpath(@__DIR__, "figs", "one_dimensional_convergence_$name.png")
        savefig(filepath, dpi=480, bbox_extra_artists=(lgd,), bbox_inches="tight")
        close(fig)
    end
end
