using Plots
using LaTeXStrings
using Printf
using Polynomials
using LinearAlgebra
using OffsetArrays

using Oceananigans
using Oceananigans.Models: ShallowWaterModel

rate_of_convergence(::UpwindBiased) = 5
rate_of_convergence(::Centered)     = 4
rate_of_convergence(::WENO{2})      = 3
rate_of_convergence(::WENO{3})      = 5
rate_of_convergence(::WENO{4})      = 7
rate_of_convergence(::WENO{5})      = 9
rate_of_convergence(::WENO{6})      = 11

labels(::Centered)     = "Center4ᵗʰ"
labels(::UpwindBiased) = "Upwind5ᵗʰ"
labels(::WENO)         = "WENOᵗʰ "

shapes(::Centered)     = :diamond
shapes(::UpwindBiased) = :square
shapes(::WENO)         = :star6
shapes(::WENO{2})      = :star5
shapes(::WENO{3})      = :diamond
shapes(::WENO{4})      = :diamond
shapes(::WENO{5})      = :diamond
shapes(::WENO{6})      = :diamond

colors(::Centered)     = :red
colors(::UpwindBiased) = :green
colors(::WENO{2})      = :blue
colors(::WENO{3})      = :cyan
colors(::WENO{4})      = :black
colors(::WENO{5})      = :yellow
colors(::WENO{6})      = :magenta

halos(::Centered)     = 2
halos(::UpwindBiased) = 3
halos(::WENO{2})      = 2
halos(::WENO{3})      = 3
halos(::WENO{4})      = 4
halos(::WENO{5})      = 5
halos(::WENO{6})      = 6

L  = 2
U  = 1
W  = 0.1
Ns = 2 .^ (5:8)
pnorm = 1

 c(x, y, z, t, U, W) = exp( - (x - U * t)^2 / W^2 )
 h(x, y, z) = 1
uh(x, y, z) = U * h(x, y, z)

schemes = (
    CenteredFourthOrder(), 
    UpwindBiasedFifthOrder(), 
    WENO(order = 3),
    WENO(order = 5),
    WENO(order = 7),
    WENO(order = 9),
    WENO(order = 11)
)s

error = Dict()
ROC   = Dict()

for N in Ns, (adv, scheme) in enumerate(schemes)

    grid = RectilinearGrid(Float64;
                           size = N, 
                           x = (-1, 1), 
                           halo = (halos(scheme)),
                           topology = (Periodic, Flat, Flat))

    Δt = 0.1 * minimum_xspacing(grid, Center(), Center(), Center())

    model = ShallowWaterModel(grid = grid,
                              momentum_advection = scheme,
                              tracer_advection = scheme,
                              coriolis = nothing,
                              gravitational_acceleration = 0,
                              tracers = :c)

    set!(model, h = h, uh = uh, c = (x, y, z) -> c(x, y, z, 0, U, W))

    simulation = Simulation(model, Δt=Δt, stop_iteration=100)

    run!(simulation)

    c₁  = c.(grid.xᶜᵃᵃ[:], 0, 0, Δt*100, U, W);

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

