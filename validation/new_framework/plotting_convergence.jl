function plot_solutions!(error, Ns, schemes, rate_of_convergence, shapes, colors, labels, pnorm)

    plt = plot()

    for scheme in schemes
    
        plot!(
            plt,
            log2.(Ns),
            [error[(N, scheme)] for N in Ns],
            seriestype = :scatter,
            shape = shapes(scheme()),
            markersize = 6,
            markercolor = colors(scheme()),
            xscale = :log10,
            yscale = :log10,
            xlabel = "log₂N",
            xticks = (log2.(Ns), string.(Int.(log2.(Ns)))),
            ylabel = "L"*string(pnorm)*"-norm: |cₛᵢₘ - c₁|",
            label =  string(labels(scheme()))*" slope = "*@sprintf("%.2f", ROC[scheme]),
            legend = :outertopright,
            title = "Rates of Convergence"
        )

    end

    for scheme in schemes
        
        roc = rate_of_convergence(scheme())
    
        plot!(
            plt,
            log2.(Ns[end-3:end]),
            [error[(N, scheme)] for N in Ns][end-3] .* (Ns[end-3] ./ Ns[end-3:end]) .^ roc,
            linestyle = :solid,
            lw = 3,
            linecolor = colors(scheme()),
            label = "Expected slope = "*@sprintf("%d",-roc)
        )
    
    end

    display(plt)
    savefig(plt, "convergence_rates")

    return
end


