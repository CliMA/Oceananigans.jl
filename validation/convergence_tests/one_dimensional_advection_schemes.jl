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

using ConvergenceTests
using ConvergenceTests.OneDimensionalGaussianAdvectionDiffusion: run_test
using ConvergenceTests.OneDimensionalUtils: unpack_errors, defaultcolors, removespines

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

advection_schemes = (CenteredSecondOrder(), CenteredFourthOrder(), UpwindBiasedThirdOrder(),
                     UpwindBiasedFifthOrder(), WENO5())

U = 1
κ = 1e-8
Nx = [8, 16, 32, 64, 96, 128, 192, 256, 384, 512]

results = Dict()
for scheme in advection_schemes
    t_scheme = typeof(scheme)
    results[t_scheme] = run_convergence_test(κ, U, Nx, scheme, arch)
end

rate_of_convergence(::CenteredSecondOrder) = 2
rate_of_convergence(::CenteredFourthOrder) = 4
rate_of_convergence(::UpwindBiasedThirdOrder) = 3
rate_of_convergence(::UpwindBiasedFifthOrder) = 5
rate_of_convergence(::WENO5) = 5
# rate_of_convergence(::WENO{K}) where K = 2K-1

test_resolution(::CenteredSecondOrder)    = 512
test_resolution(::CenteredFourthOrder)    = 512
test_resolution(::UpwindBiasedThirdOrder) = 512
test_resolution(::UpwindBiasedFifthOrder) = 512
test_resolution(::WENO5)                  = 512

tolerance(::CenteredSecondOrder)    = 0.02
tolerance(::CenteredFourthOrder)    = 0.06
tolerance(::UpwindBiasedThirdOrder) = 0.08
tolerance(::UpwindBiasedFifthOrder) = 0.2
tolerance(::WENO5)                  = 0.4

colors = ("xkcd:royal blue", "xkcd:light red")

for scheme in advection_schemes

    t_scheme = typeof(scheme)
    name = string(t_scheme)

    @testset "$name" begin

        fig, ax = subplots()

        roc = rate_of_convergence(scheme)
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

        test_rate_of_convergence(cx_L∞, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" cx_L∞")
        test_rate_of_convergence(cy_L∞, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" cy_L∞")
        test_rate_of_convergence(cz_L∞, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" cz_L∞")

        test_rate_of_convergence(uy_L∞, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" uy_L∞")
        test_rate_of_convergence(uz_L∞, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" uz_L∞")

        test_rate_of_convergence(vx_L∞, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" vx_L∞")
        test_rate_of_convergence(vz_L∞, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" vz_L∞")

        test_rate_of_convergence(wx_L∞, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" wx_L∞")
        test_rate_of_convergence(wy_L∞, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" wy_L∞")

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

        loglog(Nx, cx_L₁; marker="*", label="\$L_1\$-norm, \$c(x)\$ $name", common_kwargs...)
        loglog(Nx, cy_L₁; marker="+", label="\$L_1\$-norm, \$c(y)\$ $name", common_kwargs...)
        loglog(Nx, cz_L₁; marker="_", label="\$L_1\$-norm, \$c(z)\$ $name", common_kwargs...)

        loglog(Nx, uy_L₁; marker="1", label="\$L_1\$-norm, \$u(y)\$ $name", common_kwargs...)
        loglog(Nx, uz_L₁; marker="^", label="\$L_1\$-norm, \$u(z)\$ $name", common_kwargs...)

        loglog(Nx, vx_L₁; marker="s", label="\$L_1\$-norm, \$v(x)\$ $name", common_kwargs...)
        loglog(Nx, vz_L₁; marker="v", label="\$L_1\$-norm, \$v(z)\$ $name", common_kwargs...)

        loglog(Nx, wx_L₁; marker="X", label="\$L_1\$-norm, \$w(x)\$ $name", common_kwargs...)
        loglog(Nx, wy_L₁; marker="D", label="\$L_1\$-norm, \$w(y)\$ $name", common_kwargs...)

        common_kwargs = (linestyle="None", color=colors[2], mfc="None", alpha=0.8)

        loglog(Nx, cx_L∞; marker="*", label="\$L_\\infty\$-norm, \$c(x)\$ $name", common_kwargs...)
        loglog(Nx, cy_L∞; marker="+", label="\$L_\\infty\$-norm, \$c(y)\$ $name", common_kwargs...)
        loglog(Nx, cz_L∞; marker="_", label="\$L_\\infty\$-norm, \$c(z)\$ $name", common_kwargs...)

        loglog(Nx, uy_L∞; marker="1", label="\$L_\\infty\$-norm, \$u(y)\$ $name", common_kwargs...)
        loglog(Nx, uz_L∞; marker="^", label="\$L_\\infty\$-norm, \$u(z)\$ $name", common_kwargs...)

        loglog(Nx, vx_L∞; marker="s", label="\$L_\\infty\$-norm, \$v(x)\$ $name", common_kwargs...)
        loglog(Nx, vz_L∞; marker="v", label="\$L_\\infty\$-norm, \$v(z)\$ $name", common_kwargs...)

        loglog(Nx, wx_L∞; marker="X", label="\$L_\\infty\$-norm, \$w(x)\$ $name", common_kwargs...)
        loglog(Nx, wy_L∞; marker="D", label="\$L_\\infty\$-norm, \$w(y)\$ $name", common_kwargs...)

        label = raw"\sim N_x^{-" * "$roc" * raw"}" |> latexstring

        loglog(Nx[itest-3:itest], uy_L₁[itest] .* (Nx[itest] ./ Nx[itest-3:itest]) .^ roc, color=colors[1], alpha=0.8, label=label)

        xscale("log", base=2)
        yscale("log", base=10)
        title("Convergence for $name advection scheme")
        xlabel(L"N_x")
        ylabel("\$L\$-norms of \$ | c_\\mathrm{sim} - c_\\mathrm{analytical} |\$")
        removespines("top", "right")
        lgd = legend(loc="upper right", bbox_to_anchor=(1.4, 1.0), prop=Dict(:size=>6))

        filename = "one_dimensional_convergence_$(name)_$(typeof(arch)).png"
        filepath = joinpath(@__DIR__, "figs", filename)
        mkpath(dirname(filepath))
        savefig(filepath, dpi=480, bbox_extra_artists=(lgd,), bbox_inches="tight")
    end
end
