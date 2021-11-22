using Plots
using LaTeXStrings
using Printf
using Polynomials
using LinearAlgebra
using OffsetArrays

using Oceananigans
using Oceananigans.Grids: min_Δx    
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

for N in Ns, (adv, scheme) in enumerate(schemes)


    function Δx_str2(i, N)
        if i < N/4
            return 1
        elseif i > 3N/4
            return 1
        elseif i < N/2
            return 1.2 * (i - N/4) + 1
        else
            return 1.2 * (3N/4 - i) + 1
        end
     end   
    
     xF = zeros(Float64, N+1)
    
     for i = 2:N+1
         xF[i] = xF[i-1] + Δx_str2(i-1, N)
     end
    
     xF ./= xF[end]
     xF = xF .*2 .- 1

    grid = RectilinearGrid(Float64; size=N, 
                                x=xF, 
                                halo=(halos(scheme)),
                                topology=(Periodic, Flat, Flat))

    Δt = 0.1 * min_Δx(grid)

    if adv == 7 
        scheme = WENO5( grid = grid )
    end

    model = ShallowWaterModel(architecture = CPU(), grid = grid,
                                advection = scheme,
                                coriolis = nothing,
                                gravitational_acceleration = 0,
                                tracers = (:c))

    set!(model, h = h, uh = uh, c = (x, y, z) -> c(x, y, z, 0, U, W))

    simulation = Simulation(model, Δt=Δt, stop_iteration=1)

    run!(simulation)

    c₁  = c.(grid.xᶜᵃᵃ[:,1,1], 0, 0, Δt, U, W);

    error[(N, adv)] = norm(abs.(model.tracers.c[1:N, 1, 1] .- c₁[1:N]), pnorm)/N^(1/pnorm)   

end

println("\nResults are for the L"*string(pnorm)*"-norm:\n")

for (adv, scheme) in enumerate(schemes)

    local best_fit = fit(log10.(Ns[2:end]),
                          log10.([error[(N, adv)] for N in Ns][2:end]), 1)

    ROC[adv] = best_fit[1]
    
    @printf("Method = % 24s, Rate of Convergence = %.2f, Expected = %d \n", 
    scheme, -ROC[adv], rate_of_convergence(scheme))
end

function plot_solutions!(error, Ns, schemes, rate_of_convergence, shapes, colors, labels, pnorm, ROC)

    plt = plot()

    for (adv, scheme) in enumerate(schemes)

        plot!(
            plt,
            log2.(Ns),
            [error[(N, adv)] for N in Ns],
            seriestype = :scatter,
            shape = shapes(scheme),
            markersize = 8,
            markercolor = colors(scheme),
            xscale = :log10,
            yscale = :log10,
            xlabel = "log₂N",
            ylabel = "L"*string(pnorm)*"-norm: |cₛᵢₘ - c₁|",
            xticks = (log2.(Ns), string.(Int.(log2.(Ns)))),
            label =  string(labels(scheme))*" slope = "*@sprintf("%.2f", ROC[adv]),
            legend = :bottomleft,
            title = "Rates of Convergence"
        )

    end

    for (adv, scheme) in enumerate(schemes)

        roc = rate_of_convergence(scheme)

        best_line = [error[(Ns[1], adv)]] .* (Ns[1] ./ Ns) .^ roc

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

