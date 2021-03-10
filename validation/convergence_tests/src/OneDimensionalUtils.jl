module OneDimensionalUtils

using Oceananigans.Grids
using Oceananigans.Advection
using Plots
using Printf

""" Unpack a vector of results associated with a convergence test. """
function unpack_solutions(results)
    c_ana = map(r -> r.cx.analytical[:], results)
    c_sim = map(r -> r.cx.simulation[:], results)
    return c_ana, c_sim
end

function unpack_errors(results)
    
    cx_L₁ = map(r -> r.cx.L₁, results)
    cx_L∞ = map(r -> r.cx.L∞, results)

    return (cx_L₁, cx_L∞)
end

""" Unpack a vector of grids associated with a convergence test. """
unpack_grids(results) = map(r -> r.grid, results)


function new_plot_solutions!(all_results, t_scheme, Ns)

    results = [all_results[(N,t_scheme)] for N in Ns]
        
    c_ana, c_sim = unpack_solutions(results)

    grids = unpack_grids(results)

    x = xnodes(Cell, grids[end])[:]

    ### Plot analytical solution
    plt = plot(x,
               c_ana[end],
               lw = 3,
               linestyle = :solid,
               label = "analytical",
               xlabel = "x",
               xlims = (minimum(x), maximum(x))
        )

    ### Plot simultion solution
    for i in 1:length(c_sim)
        x = xnodes(Cell, grids[i])[:]
        Nx = length(x)
            
        plot!(plt,
              x,
              c_sim[i],
              lw = 2,
              linestyle = :dash,
              label = @sprintf("simulation for Nx = %d\n", Nx)
              )
    end

    display(plt)
    savefig(plt, string("test1", t_scheme))

    plt2 = plot()
        
    ### Plot error for each simulation solution
    for i in 1:length(c_sim)
            
        x = xnodes(Cell, grids[i])[:]
        Nx = length(x)
        error = abs.(c_sim[i] .- c_ana[i]) .+ 1e-16

        plot!(plt2,
              x,
              error,
              lw = 2,
              linestyle = :solid,
              xlabel = "x",
              xlims = (minimum(x), maximum(x)),
              label = @sprintf("error for Nx = %d", Nx),
              yaxis = :log
              )
            
    end
        
    display(plt2)
    savefig(plt2, string("test2", t_scheme))
        
    return 
end


function plot_solutions!(all_results, t_scheme)

    for j = 1:length(all_results)

        results = all_results[t_scheme]
        
        c_ana, c_sim = unpack_solutions(results)

        grids = unpack_grids(results)

        x = xnodes(Center, grids[end])[:]

        ### Plot analytical solution
        plt = plot(x,
                   c_ana[end],
                   lw = 3,
                   linestyle = :solid,
                   label = "analytical",
                   xlabel = "x",
                   xlims = (minimum(x), maximum(x))
        )

        ### Plot simultion solution
        for i in 1:length(c_sim)
            x = xnodes(Center, grids[i])[:]

            plot(x, c_sim[i], linestyle, color=defaultcolors[i], alpha=0.8, linewidth=1,
                 label="$name simulated, \$ N_x \$ = $(grids[i].Nx)")

            xlim(minimum(x), maximum(x))
        end

        display(plt)
        savefig(plt, string("test1", t_scheme))

        plt2 = plot()
        
        ### Plot error for each simulation solution
        for i in 1:length(c_sim)
            x = xnodes(Center, grids[i])[:]

            semilogy(x, abs.(c_sim[i] .- c_ana[i]), linestyle, color=defaultcolors[i], alpha=0.8,
                     label="$name, \$ N_x \$ = $(grids[i].Nx)")

            xlim(minimum(x), maximum(x))
        end
        
        display(plt2)
        savefig(plt2, string("test2", t_scheme))
        
    end
    
    return 
end


function plot_error_convergence!(Nx, all_results, t_scheme)

    for j = 1:length(all_results)
        
        results = all_results[t_scheme]
       
        u_L₁, v_L₁, cx_L₁, cy_L₁, u_L∞, v_L∞, cx_L∞, cy_L∞ = unpack_errors(results)

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
    u_L₁, v_L₁, cx_L₁, cy_L₁, u_L∞, v_L∞, cx_L∞, cy_L∞ = unpack_errors(all_results[1])
    loglog(Nx, cx_L₁[1] .* (Nx[1] ./ Nx).^2, "k-", alpha=0.8, label=L"\sim N_x^{-2}")

    xscale("log", base=2)
    yscale("log", base=10)
    xlabel(L"N_x")
    ylabel("\$L\$-norms of \$ | c_\\mathrm{sim} - c_\\mathrm{analytical} |\$")
    removespines("top", "right")
    lgd = legend(loc="upper right", bbox_to_anchor=(1.4, 1.0), prop=Dict(:size=>6))

    return lgd
end

end # module
