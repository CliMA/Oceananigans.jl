using Plots
using LaTeXStrings
using Printf

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


