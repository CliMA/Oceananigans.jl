module OneDimensionalUtils

#using PyPlot
using Plots

using Oceananigans.Grids
using Oceananigans.Advection

using Printf

#defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#removespine(side) = gca().spines[side].set_visible(false)
#removespines(sides...) = [removespine(side) for side in sides]

""" Unpack a vector of results associated with a convergence test. """
function unpack_solutions(results)
    c_ana = map(r -> r.cx.analytical[:], results)
    c_sim = map(r -> r.cx.simulation[:], results)
    return c_ana, c_sim
end

function unpack_errors(results)
    cx_L₁ = map(r -> r.cx.L₁, results)
    #=
    vx_L₁ = map(r -> r.vx.L₁, results)
    wx_L₁ = map(r -> r.vx.L₁, results)

    cy_L₁ = map(r -> r.cy.L₁, results)
    uy_L₁ = map(r -> r.uy.L₁, results)
    wy_L₁ = map(r -> r.wy.L₁, results)

    cz_L₁ = map(r -> r.cz.L₁, results)
    uz_L₁ = map(r -> r.uz.L₁, results)
    vz_L₁ = map(r -> r.vz.L₁, results)
    =#
    
    cx_L∞ = map(r -> r.cx.L∞, results)
    #=
    vx_L∞ = map(r -> r.vx.L∞, results)
    wx_L∞ = map(r -> r.vx.L∞, results)

    cy_L∞ = map(r -> r.cy.L∞, results)
    uy_L∞ = map(r -> r.uy.L∞, results)
    wy_L∞ = map(r -> r.wy.L∞, results)

    cz_L∞ = map(r -> r.cz.L∞, results)
    uz_L∞ = map(r -> r.uz.L∞, results)
    vz_L∞ = map(r -> r.vz.L∞, results)
    
    return (cx_L₁, cy_L₁, cz_L₁,
            uy_L₁, uz_L₁,
            vx_L₁, vz_L₁,
            wx_L₁, wy_L₁,
            cx_L∞, cy_L∞, cz_L∞,
            uy_L∞, uz_L∞,
            vx_L∞, vz_L∞,
            wx_L∞, wy_L∞)
    =#
    return (cx_L₁, cx_L∞)
end

""" Unpack a vector of grids associated with a convergence test. """
unpack_grids(results) = map(r -> r.grid, results)


#function plot_solutions!(axs, all_results, names, linestyles, specialcolors)
function plot_solutions!(all_results, t_scheme)

    for j = 1:length(all_results)

        results = all_results[t_scheme]
        #name = names[j]
        #linestyle = linestyles[j]

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

        #plt2 = plot(x,
        #            0 .* x .+ 1e-16,
        #            yaxis = :log,
        #            label = "zero"
        #            )
        plt2 = plot()
        
        ### Plot error for each simulation solution
        for i in 1:length(c_sim)
            x = xnodes(Cell, grids[i])[:]
            Nx = length(x)

            error = abs.(c_sim[i] .- c_ana[i]) .+ 1e-16
            println("Error for ", t_scheme, " with Nx = ", Nx, " is ", maximum(error), "\n")

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
        
    end
    
    return 
end


function plot_error_convergence!(Nx, all_results, t_scheme)

    for j = 1:length(all_results)
        results = all_results[t_scheme]
        #results = all_results[j]
        #name = names[j]
        u_L₁, v_L₁, cx_L₁, cy_L₁, u_L∞, v_L∞, cx_L∞, cy_L∞ = unpack_errors(results)

        #common_kwargs = (linestyle="None", color=defaultcolors[j], mfc="None", alpha=0.8)
        #loglog(Nx,  u_L₁; basex=2, marker="o", label="\$L_1\$-norm, \$u\$ $name", common_kwargs...)
        #loglog(Nx,  v_L₁; basex=2, marker="2", label="\$L_1\$-norm, \$v\$ $name", common_kwargs...)
        #loglog(Nx, cx_L₁; basex=2, marker="*", label="\$L_1\$-norm, \$x\$ tracer $name", common_kwargs...)
        #loglog(Nx, cy_L₁; basex=2, marker="+", label="\$L_1\$-norm, \$y\$ tracer $name", common_kwargs...)

        #loglog(Nx,  u_L∞; basex=2, marker="1", label="\$L_\\infty\$-norm, \$u\$ $name", common_kwargs...)
        #loglog(Nx,  v_L∞; basex=2, marker="_", label="\$L_\\infty\$-norm, \$v\$ $name", common_kwargs...)
        #loglog(Nx, cx_L∞; basex=2, marker="^", label="\$L_\\infty\$-norm, \$x\$ tracer $name", common_kwargs...)
        #loglog(Nx, cy_L∞; basex=2, marker="s", label="\$L_\\infty\$-norm, \$y\$ tracer $name", common_kwargs...)
    end

    # Guide line to confirm second-order scaling
    #u_L₁, v_L₁, cx_L₁, cy_L₁, u_L∞, v_L∞, cx_L∞, cy_L∞ = unpack_errors(all_results[1])
    #loglog(Nx, cx_L₁[1] .* (Nx[1] ./ Nx).^2, "k-", basex=2, alpha=0.8, label=L"\sim N_x^{-2}")

    #xlabel(L"N_x")
    #ylabel("\$L\$-norms of \$ | c_\\mathrm{sim} - c_\\mathrm{analytical} |\$")
    #removespines("top", "right")
    #lgd = legend(loc="upper right", bbox_to_anchor=(1.4, 1.0), prop=Dict(:size=>6))

    return #lgd
end

end # module
