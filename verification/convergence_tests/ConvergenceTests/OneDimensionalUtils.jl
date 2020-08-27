module OneDimensionalUtils

using PyPlot

using Oceananigans.Grids

defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
removespine(side) = gca().spines[side].set_visible(false)
removespines(sides...) = [removespine(side) for side in sides]

""" Unpack a vector of results associated with a convergence test. """
function unpack_solutions(results)
    c_ana = map(r -> r.cx.analytical[:], results)
    c_sim = map(r -> r.cx.simulation[:], results)
    return c_ana, c_sim
end

function unpack_errors(results)
     u_L₁ = map(r -> r.u.L₁, results)
     v_L₁ = map(r -> r.v.L₁, results)
    cx_L₁ = map(r -> r.cx.L₁, results)
    cy_L₁ = map(r -> r.cy.L₁, results)

     u_L∞ = map(r ->  r.u.L∞, results)
     v_L∞ = map(r ->  r.v.L∞, results)
    cx_L∞ = map(r -> r.cx.L∞, results)
    cy_L∞ = map(r -> r.cy.L∞, results)

    return (u_L₁, v_L₁, cx_L₁, cy_L₁,
            u_L∞, v_L∞, cx_L∞, cy_L∞)
end

""" Unpack a vector of grids associated with a convergence test. """
unpack_grids(results) = map(r -> r.grid, results)

function plot_solutions!(axs, all_results, names, linestyles, specialcolors)

    for j = 1:length(all_results)

        results = all_results[j]
        name = names[j]
        linestyle = linestyles[j]

        c_ana, c_sim = unpack_solutions(results)

        grids = unpack_grids(results)

        x = xnodes(Cell, grids[end])[:]

        sca(axs[1])

        plot(x, c_ana[end], "-", color=specialcolors[j], alpha=0.2, linewidth=3, label="$name analytical")

        for i in 1:length(c_sim)
            x = xnodes(Cell, grids[i])[:]

            plot(x, c_sim[i], linestyle, color=defaultcolors[i], alpha=0.8, linewidth=1,
                 label="$name simulated, \$ N_x \$ = $(grids[i].Nx)")
        end

        sca(axs[2])

        for i in 1:length(c_sim)
            x = xnodes(Cell, grids[i])[:]

            plot(x, abs.(c_sim[i] .- c_ana[i]), linestyle, color=defaultcolors[i], alpha=0.8,
                 label="$name, \$ N_x \$ = $(grids[i].Nx)")
        end

    end

    sca(axs[1])
    ylabel(L"c")
    removespines("top", "right", "bottom")
    axs[1].tick_params(bottom=false, labelbottom=false)
    legend(loc="upper left", prop=Dict(:size=>7), bbox_to_anchor=(0.01, 0.3, 1.5, 1.0))

    sca(axs[2])
    axs[2].set_yscale("log")
    removespines("top", "right")
    xlabel(L"x")
    ylabel("Absolute error \$ | c_\\mathrm{sim} - c_\\mathrm{analytical} | \$")
    legend(loc="upper left", prop=Dict(:size=>10), bbox_to_anchor=(0.01, 0.3, 1.5, 1.0))

    return nothing
end

function plot_error_convergence!(axs, Nx, all_results, names)

    for j = 1:length(all_results)
        results = all_results[j]
        name = names[j]
        u_L₁, v_L₁, cx_L₁, cy_L₁, u_L∞, v_L∞, cx_L∞, cy_L∞  = unpack_errors(results)

        @show u_L₁, u_L∞

        common_kwargs = (linestyle="None", color=defaultcolors[j], mfc="None", alpha=0.8)
        loglog(Nx,  u_L₁; marker="o", label="\$L_1\$-norm, \$u\$ $name", common_kwargs...)
        loglog(Nx,  v_L₁; marker="2", label="\$L_1\$-norm, \$v\$ $name", common_kwargs...)
        loglog(Nx, cx_L₁; marker="*", label="\$L_1\$-norm, \$x\$ tracer $name", common_kwargs...)
        loglog(Nx, cy_L₁; marker="+", label="\$L_1\$-norm, \$y\$ tracer $name", common_kwargs...)

        loglog(Nx,  u_L∞; marker="1", label="\$L_\\infty\$-norm, \$u\$ $name", common_kwargs...)
        loglog(Nx,  v_L∞; marker="_", label="\$L_\\infty\$-norm, \$v\$ $name", common_kwargs...)
        loglog(Nx, cx_L∞; marker="^", label="\$L_\\infty\$-norm, \$x\$ tracer $name", common_kwargs...)
        loglog(Nx, cy_L∞; marker="s", label="\$L_\\infty\$-norm, \$y\$ tracer $name", common_kwargs...)
    end

    # Guide line to confirm second-order scaling
    u_L₁, v_L₁, cx_L₁, cy_L₁, u_L∞, v_L∞, cx_L∞, cy_L∞  = unpack_errors(all_results[1])
    loglog(Nx, cx_L₁[1] .* (Nx[1] ./ Nx).^2, "k-", alpha=0.8, label=L"\sim N_x^{-2}")

    axs.grid(which="both", linewidth=1)
    xlabel(L"N_x")
    ylabel("\$L\$-norms of \$ | c_\\mathrm{sim} - c_\\mathrm{analytical} |\$")
    removespines("top", "right")
    legend(loc="upper right", prop=Dict(:size=>6))
    xticks(Nx, ["\$ 2^{$(round(Int, log2(n)))} \$" for n in Nx])

end

end # module
