using Plots
using LaTeXStrings
using Printf
using Polynomials
using LinearAlgebra
using OffsetArrays

using Oceananigans
using Oceananigans.Models: ShallowWaterModel

rate_of_convergence(::UpwindBiasedFirstOrder) = 1
rate_of_convergence(::CenteredSecondOrder)    = 2
rate_of_convergence(::UpwindBiasedThirdOrder) = 3
rate_of_convergence(::CenteredFourthOrder)    = 4
rate_of_convergence(::UpwindBiasedFifthOrder) = 5
rate_of_convergence(::WENO5)                  = 5

labels(::UpwindBiasedFirstOrder) = "Upwind1ˢᵗ"
labels(::CenteredSecondOrder)    = "Center2ⁿᵈ"
labels(::UpwindBiasedThirdOrder) = "Upwind3ʳᵈ"
labels(::CenteredFourthOrder)    = "Center4ᵗʰ"
labels(::UpwindBiasedFifthOrder) = "Upwind5ᵗʰ"
labels(::WENO5)                  = "WENO5ᵗʰ "

shapes(::UpwindBiasedFirstOrder) = :square
shapes(::CenteredSecondOrder)    = :diamond
shapes(::UpwindBiasedThirdOrder) = :dtriangle
shapes(::CenteredFourthOrder)    = :rect
shapes(::UpwindBiasedFifthOrder) = :star5
shapes(::WENO5)                  = :star6

colors(::UpwindBiasedFirstOrder) = :black
colors(::CenteredSecondOrder)    = :green
colors(::UpwindBiasedThirdOrder) = :red
colors(::CenteredFourthOrder)    = :cyan
colors(::UpwindBiasedFifthOrder) = :magenta
colors(::WENO5)                  = :purple

halos(::UpwindBiasedFirstOrder) = 1
halos(::CenteredSecondOrder)    = 1
halos(::UpwindBiasedThirdOrder) = 2
halos(::CenteredFourthOrder)    = 2
halos(::UpwindBiasedFifthOrder) = 3
halos(::WENO5)                  = 3 

L  = 2
U  = 1
W  = 0.1
Ns = 2 .^ (6:10)
Δt = 0.01 * minimum(L/Ns) / U
pnorm = 1

c(x, y, z, t, U, W) = exp( - (x - U * t)^2 / W^2 )
   h(x, y, z) = 1
  uh(x, y, z) = U * h(x, y, z)

schemes = (
 UpwindBiasedFirstOrder(), 
 CenteredSecondOrder(), 
 UpwindBiasedThirdOrder(), 
 CenteredFourthOrder(), 
 UpwindBiasedFifthOrder(), 
 WENO5()
);

error = Dict()
ROC   = Dict()

for N in Ns, scheme in schemes

    grid = RegularRectilinearGrid(Float64; size=N, 
                                x=(-1, 1), 
                                halo=(halos(scheme)),
                                topology=(Periodic, Flat, Flat))

    model = ShallowWaterModel(architecture = CPU(), grid = grid,
                                advection = scheme,
                                coriolis = nothing,
                                gravitational_acceleration = 0,
                                tracers = (:c))

    set!(model, h = h, uh = uh, c = (x, y, z) -> c(x, y, z, 0, U, W))

    simulation = Simulation(model, Δt=Δt, stop_iteration=1, iteration_interval=1)

    run!(simulation)

    c₁  = c.(grid.xC[:,1,1], 0, 0, Δt, U, W);

    error[(N, scheme)] = norm(abs.(model.tracers.c[1:N, 1, 1] .- c₁[1:N]), pnorm)/N^(1/pnorm)   

end

println("\nResults are for the L"*string(pnorm)*"-norm:\n")

for scheme in schemes

    local best_fit = fit(log10.(Ns[2:end]),
                          log10.([error[(N, scheme)] for N in Ns][2:end]), 1)

    ROC[scheme] = best_fit[1]
    
    @printf("Method = % 24s, Rate of Convergence = %.2f, Expected = %d \n", 
    scheme, -ROC[scheme], rate_of_convergence(scheme))
end

function plot_solutions!(error, Ns, schemes, rate_of_convergence, shapes, colors, labels, pnorm, ROC)

    plt = plot()

    for scheme in schemes

        plot!(
            plt,
            log2.(Ns),
            [error[(N, scheme)] for N in Ns],
            seriestype = :scatter,
            shape = shapes(scheme),
            markersize = 8,
            markercolor = colors(scheme),
            xscale = :log10,
            yscale = :log10,
            xlabel = "log₂N",
            ylabel = "L"*string(pnorm)*"-norm: |cₛᵢₘ - c₁|",
            xticks = (log2.(Ns), string.(Int.(log2.(Ns)))),
            label =  string(labels(scheme))*" slope = "*@sprintf("%.2f", ROC[scheme]),
            legend = :bottomleft,
            title = "Rates of Convergence"
        )

    end

    for scheme in schemes

        roc = rate_of_convergence(scheme)

        best_line = [error[(Ns[1], scheme)]] .* (Ns[1] ./ Ns) .^ roc

        plot!(
            plt,
            log2.(Ns),
            best_line,
            linestyle = :solid,
            lw = 3,
            linecolor = colors(scheme),
            label = "Expected slope = "*@sprintf("%d",-roc)
        )

    end

    return plt
end

plt = plot_solutions!(
    error,
    Ns,
    schemes,
    rate_of_convergence,
    shapes,
    colors,
    labels,
    pnorm,
    ROC)
savefig(plt, "convergence_rates")

