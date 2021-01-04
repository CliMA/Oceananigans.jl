using Test
using Plots
using LaTeXStrings
using Printf
using Polynomials

using Oceananigans.Advection
using Oceananigans.Grids: xnodes

# Define a few utilities for running tests and unpacking and plotting results
include("ConvergenceTests/ConvergenceTests.jl")

using .ConvergenceTests
using .ConvergenceTests.OneDimensionalGaussianAdvectionDiffusion: run_test
using .ConvergenceTests.OneDimensionalUtils: new_plot_solutions!, unpack_solutions, unpack_errors,  unpack_grids

""" Run advection test for all Nx in resolutions. """
function run_convergence_test(κ, U, resolutions, advection_scheme)

    # Determine safe time-step
           Lx = 2.5
            h = Lx / maximum(resolutions[end])
           Δt = min(0.01 * h / U)
    stop_time = Δt

    # Run the tests
    print("advection_scheme = ", advection_scheme, "\n")    
    results = [run_test(             Nx = Nx,
                                     Δt = Δt,
                              advection = advection_scheme,
                         stop_iteration = 1,
                                      U = U,
#                                      κ = κ
                         ) for Nx in resolutions]

    return results
end

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
Ns = 2 .^ (6:10)

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
ROC = Dict()

for scheme in schemes
    
    print("scheme = ", scheme, "\n")    
    t_scheme = typeof(scheme)
    
    Lx = 2.5
    Δt = 0.01 * minimum(Lx/Ns) / U
    stop_time = Δt
    
    for N in Ns

        results[(N,t_scheme)] = run_test( Nx = N,
                                        Δt = Δt,
                                        advection = scheme,
                                        stop_iteration = 1,
                                        U = U) 
    end
    
    new_plot_solutions!(results, t_scheme, Ns)
    
end

for scheme in schemes

    t_scheme = typeof(scheme)
    name = string(t_scheme)
    roc = rate_of_convergence(scheme)
    atol = tolerance(scheme)
    Ntest = test_resolution(scheme)
    #itest = searchsortedfirst(Ns, Ntest)

    (cx_L₁, cx_L∞) = unpack_errors( [results[(N,t_scheme)] for N in Ns] )

    #test_rate_of_convergence(cx_L₁, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" cx_L₁")
    #test_rate_of_convergence(cx_L∞, Nx, Ntest=Ntest, expected=-roc, atol=atol, name=name*" cx_L∞")

    plt1 = plot(log2.(Ns),
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
    #=
    plot!(plt1,
          log2.(Ns),
          cx_L∞,
          seriestype = :scatter,
          shape = :star6,
          markersize = 6,
          markercolor = :red,
          label = "L∞-norm, c(x) "*string(t_scheme), 
          )
    =#
    
    name = t_scheme
    roc = rate_of_convergence(scheme)
    j = 3
    best_fit = fit(log10.(Ns[2:end]), log10.(cx_L₁[2:end]), 1)
    ROC[scheme] = best_fit[1]
    println("Method = ", name, ",           Rate of Convergence = ", @sprintf("%.2f", -ROC[scheme]), ", Expected = ", roc)

    itest = length(Ns)
    plot!(plt1,
          log2.(Ns[itest-j:itest]),
          cx_L₁[itest] .* (Ns[itest] ./ Ns[itest-j:itest]) .^ roc,
          linestyle = :solid,
          lw = 3,
          label = raw"\sim N_x^{-" * "$roc" * raw"}" |> latexstring
          )

    display(plt1)

    savefig(plt1, string("convergence_rates", t_scheme))

end

