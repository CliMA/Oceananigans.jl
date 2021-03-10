if haskey(ENV, "CI") && ENV["CI"] == "true"
    ENV["PYTHON"] = ""
    using Pkg
    Pkg.build("PyCall")
end

using CUDA
using PyPlot
using Oceananigans

using ConvergenceTests
using ConvergenceTests.TwoDimensionalDiffusion: run_and_analyze

defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
removespine(side) = gca().spines[side].set_visible(false)
removespines(sides...) = [removespine(side) for side in sides]

function convergence_test(Nx, Δt, stop_iteration, arch, topo)

    results = [run_and_analyze(architecture=arch, Nx=N, Δt=Δt, stop_iteration=stop_iteration,
                               topo=topo, output=false) for N in Nx]

    L₁ = map(r -> r.L₁, results)
    L∞ = map(r -> r.L∞, results)

    return (L₁=L₁, L∞=L∞)
end

# Setup and run simulations
arch = CUDA.has_cuda() ? GPU() : CPU()
Nx = [8, 16, 32, 64, 128, 256]
stop_time = 1e-4

# Calculate time step based on diffusive time-step constraint for finest mesh
        min_Δx = 2π / maximum(Nx)
   proposal_Δt = 1e-3 * min_Δx^2 # proposal time-step
stop_iteration = round(Int, stop_time / proposal_Δt)
            Δt = stop_time / stop_iteration # ensure time-stepping to exact finish time.

all_topologies = ((Periodic, Periodic, Bounded),
                  (Periodic, Bounded,  Bounded),
                  (Bounded,  Bounded,  Bounded))

topologies = arch isa CPU ? all_topologies : all_topologies[1:2]

errors = [convergence_test(Nx, Δt, stop_iteration, arch, topo) for topo in topologies]

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

xscale("log", base=2)
yscale("log", base=10)
xlabel(L"N_x")
ylabel("\$L\$-norms of \$ | c_\\mathrm{sim} - c_\\mathrm{analytical} |\$")
title("Two dimensional diffusion convergence test")
removespines("top", "right")
legend(loc="upper right")

filename = "two_dimensional_diffusion_convergence_$(typeof(arch)).png"
filepath = joinpath(@__DIR__, "figs", filename)
mkpath(dirname(filepath))
savefig(filepath, dpi=480)

for (itopo, topo) in enumerate(topologies)
    L₁_topo = errors[itopo].L₁
    L∞_topo = errors[itopo].L∞
    test_rate_of_convergence(L₁_topo, Nx, expected=-2.0, atol=0.01, name="2D diffusion $topo L₁")
    test_rate_of_convergence(L∞_topo, Nx, expected=-2.0, atol=0.06, name="2D diffusion $topo L∞")
end
