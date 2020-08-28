using Oceananigans.Grids

using PyPlot

include("ConvergenceTests/ConvergenceTests.jl")

defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
removespine(side) = gca().spines[side].set_visible(false)
removespines(sides...) = [removespine(side) for side in sides]

run = ConvergenceTests.TwoDimensionalDiffusion.run_simulation
run_and_analyze = ConvergenceTests.TwoDimensionalDiffusion.run_and_analyze

function convergence_test(Nx, Δt, stop_iteration, topo)
    results = [run_and_analyze(Nx=N, Δt=Δt, stop_iteration=stop_iteration, topo=topo,
                               output=false) for N in Nx]

    L₁ = map(r -> r.L₁, results)
    L∞ = map(r -> r.L∞, results)

    return (L₁=L₁, L∞=L∞)
end

# Setup and run 4 simulations
Nx = [8, 16, 32, 64, 128, 256]
stop_time = 1e-4

# Calculate time step based on diffusive time-step constraint for finest mesh
        min_Δx = 2π / maximum(Nx)
   proposal_Δt = 1e-3 * min_Δx^2 # proposal time-step
stop_iteration = round(Int, stop_time / proposal_Δt)
            Δt = stop_time / stop_iteration # ensure time-stepping to exact finish time.

topologies = (
              (Periodic, Periodic, Bounded),
              (Periodic, Bounded, Bounded),
              (Bounded, Bounded, Bounded)
             )

errors = [convergence_test(Nx, Δt, stop_iteration, topo) for topo in topologies]

fig, axs = subplots()

for (itopo, topo) in enumerate(topologies)
    L₁ = errors[itopo].L₁
    L∞ = errors[itopo].L∞
    name ="$(topo[1]), $(topo[2])"

    loglog(Nx, L₁, "o", color=defaultcolors[itopo], mfc="None", label="\$L_1\$-norm, $name")
    loglog(Nx, L∞, "^", color=defaultcolors[itopo], mfc="None", label="\$L_\\infty\$-norm, $name")
end

L₁ = errors[1].L₁
L∞ = errors[1].L∞
loglog(Nx, L₁[1] * (Nx[1]./Nx).^2, "k-", alpha=0.6, linewidth=1, label=L"\sim N_x^{-2}")

xlabel(L"N_x")
ylabel("\$L\$-norms of \$ | c_\\mathrm{sim} - c_\\mathrm{analytical} |\$")
title("Two dimensional diffusion convergence test")
removespines("top", "right")
legend(loc="upper right")
xticks(Nx, ["\$ 2^{$(round(Int, log2(n)))} \$" for n in Nx])

savefig("figs/two_dimensional_diffusion_convergence.png", dpi=480)
