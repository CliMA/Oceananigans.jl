module TwoDimensionalUtils

using PyPlot

using Oceananigans.Grids

defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
removespine(side) = gca().spines[side].set_visible(false)
removespines(sides...) = [removespine(side) for side in sides]

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

end # module
