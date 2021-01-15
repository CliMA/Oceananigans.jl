using Test
using Plots
using LaTeXStrings
using Printf
using Oceananigans.Advection

# Define a few utilities for running tests and unpacking and plotting results
include("ConvergenceTests/ConvergenceTests.jl")

using .ConvergenceTests
using .ConvergenceTests.OneDimensionalGaussianAdvectionDiffusion: run_test
using .ConvergenceTests.OneDimensionalUtils: plot_solutions!, unpack_solutions, unpack_errors,  unpack_grids

#####
##### Run test
#####

schemes = (CenteredSecondOrder(),
#           CenteredFourthOrder(),
#           UpwindBiasedThirdOrder(),
#           UpwindBiasedFifthOrder(),
#           WENO5()
          )

 U = 1
 κ = 1e-8
Ns = 2 .^ (5:10)

tolerance(::CenteredSecondOrder)    = 0.05
tolerance(::CenteredFourthOrder)    = 0.05
tolerance(::UpwindBiasedThirdOrder) = 0.30
tolerance(::UpwindBiasedFifthOrder) = 5.30   # this is bad!!!!!
tolerance(::WENO5)                  = 0.40

test_resolution(::CenteredSecondOrder)    = 1024
test_resolution(::CenteredFourthOrder)    = 1024
test_resolution(::UpwindBiasedThirdOrder) = 1024
test_resolution(::UpwindBiasedFifthOrder) = 1024
test_resolution(::WENO5)                  = 1024

rate_of_convergence(::CenteredSecondOrder) = 2
rate_of_convergence(::CenteredFourthOrder) = 4
rate_of_convergence(::UpwindBiasedThirdOrder) = 3
rate_of_convergence(::UpwindBiasedFifthOrder) = 5
rate_of_convergence(::WENO5) = 5

results = Dict()
for scheme in schemes

    t_scheme = typeof(scheme)
    Lx = 2.5
    h = Lx / maximum(Ns[end])
    Δt = min(0.01 * h / U)
    stop_time = Δt

    results[t_scheme] = [run_test( Nx = N,
                                      Δt = Δt,
                                      advection = scheme,
                                      stop_iteration = 1,
                                      U = 1.0
                                      ) for N in Ns]

    #results[t_scheme] = run_convergence_test(κ, U, Nx, scheme)
    plot_solutions!(results, t_scheme)
end

for scheme in advection_schemes

    t_scheme = typeof(scheme)
    name = string(t_scheme)
    roc = rate_of_convergence(scheme)
    atol = tolerance(scheme)
    Ntest = test_resolution(scheme)
    itest = searchsortedfirst(Nx, Ntest)

    (cx_L₁, cx_L∞) = unpack_errors(results[t_scheme])

    #test_rate_of_convergence(cx_L₁, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" cx_L₁")
    #test_rate_of_convergence(cx_L∞, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" cx_L∞")

    plt1 = plot(log2.(Nx),
                cx_L₁,
                seriestype = :scatter,
                shape = :star4,
                markersize = 6,
                markercolor = :blue,
                xlabel = "log₂N",
                ylabel = "L-norms of |cₛᵢₘ - cₐₙₐₗ|",
                label = "L₁-norm, c(x) "*string(t_scheme),
                title = "Convergence plots for advection", 
                yaxis = :log10
                )
    plot!(plt1,
          log2.(Nx),
          cx_L∞,
          seriestype = :scatter,
          shape = :star6,
          markersize = 6,
          markercolor = :red,
          label = "L∞-norm, c(x) "*string(t_scheme), 
          )

    plot!(plt1,
          log2.(Nx[itest-3:itest]),
          cx_L₁[itest] .* (Nx[itest] ./ Nx[itest-3:itest]) .^ roc,
          linestyle = :solid,
          lw = 3,
          label = raw"\sim N_x^{-" * "$roc" * raw"}" |> latexstring
          )

    display(plt1)

    savefig(plt1, string("convergence_rates", t_scheme))

    print(Nx,"\n")
    print(cx_L₁,"\n")
    #print(cx_L∞)
end

