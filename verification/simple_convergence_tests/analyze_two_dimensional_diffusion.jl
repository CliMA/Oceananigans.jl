using PyPlot, Glob, JLD2

defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

include("ConvergenceTests/ConvergenceTests.jl")

free_slip_simulation_files = glob("data/free_decay*.jld2")

errors = ConvergenceTests.compute_errors((x, y, z, t) -> ConvergenceTests.TwoDimensionalDiffusion.c(x, y, t), 
                                         free_slip_simulation_files...; name=:c)

sizes = ConvergenceTests.extract_sizes(free_slip_simulation_files...)

Nx = map(sz -> sz[1], sizes)
L₁ = map(err -> err.L₁, errors)
L∞ = map(err -> err.L∞, errors)

close("all")
fig, ax = subplots()

ax.tick_params(bottom=false, labelbottom=false)

ax.loglog(Nx, L₁, linestyle="None", marker="o", label="error, \$L_1\$-norm")
ax.loglog(Nx, L∞, linestyle="None", marker="^", label="error, \$L_\\infty\$-norm")

ax.loglog(Nx, 200 * Nx.^(-2), "k--", linewidth=1, alpha=0.6, label="\$ 200 N_x^{-2} \$")
ax.loglog(Nx, 5 * Nx.^(-1),   "k-", linewidth=1, alpha=0.6, label="\$ 5 N_x^{-1} \$")

legend()

title("Convergence for free decay")
ylabel("Norms of the absolute error, \$ | u_\\mathrm{simulation} - u_\\mathrm{analytical} | \$")
xlabel(L"N_x")

#xticks(Nx, ["\$ 2^$(round(Int, log2(N))) \$" for N in Nx])

pause(0.1)
