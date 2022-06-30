module TwoDimensionalUtils

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
    cxy_L₁ = map(r -> r.cxy.L₁, results)
    cyz_L₁ = map(r -> r.cyz.L₁, results)
    cxz_L₁ = map(r -> r.cxz.L₁, results)

    uyz_L₁ = map(r -> r.uyz.L₁, results)
    vxz_L₁ = map(r -> r.vxz.L₁, results)
    wxy_L₁ = map(r -> r.wxy.L₁, results)
    
    cxy_L∞ = map(r -> r.cxy.L∞, results)
    cyz_L∞ = map(r -> r.cyz.L∞, results)
    cxz_L∞ = map(r -> r.cxz.L∞, results)

    uyz_L∞ = map(r -> r.uyz.L∞, results)
    vxz_L∞ = map(r -> r.vxz.L∞, results)
    wxy_L∞ = map(r -> r.wxy.L∞, results)

    return (cxy_L₁, cyz_L₁, cxz_L₁,
            uyz_L₁,
            vxz_L₁,
            wxy_L₁,
            cxy_L∞, cyz_L∞, cxz_L∞,
            uyz_L∞,
            vxz_L∞,
            wxy_L∞)
end

""" Unpack a vector of grids associated with a convergence test. """
unpack_grids(results) = map(r -> r.grid, results)

function plot_error_convergence!(axs, Nx, all_results, names)

    for j = 1:length(all_results)
        results = all_results[j]
        name = names[j]
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
