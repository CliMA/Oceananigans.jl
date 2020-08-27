using PyPlot, Glob, JLD2

defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
removespine(side) = gca().spines[side].set_visible(false)
removespines(sides...) = [removespine(side) for side in sides]

include("ConvergenceTests/ConvergenceTests.jl")

filenames = glob("data/taylor_green*.jld2")

errors = ConvergenceTests.compute_errors(
            (x, y, z, t) -> ConvergenceTests.DoublyPeriodicTaylorGreen.u(x, y, t),
            filenames...)

sizes = ConvergenceTests.extract_sizes(filenames...)

Nx = map(sz -> sz[1], sizes)
L₁ = map(err -> err.L₁, errors)
L∞ = map(err -> err.L∞, errors)

close("all")
fig, ax = subplots()

ax.loglog(Nx, L₁, linestyle="None", marker="o", label="error, \$L_1\$-norm")
ax.loglog(Nx, L∞, linestyle="None", marker="^", label="error, \$L_\\infty\$-norm")
ax.loglog(Nx, L₁[end] * (Nx[end] ./ Nx).^2, "k-", linewidth=1, alpha=0.6,  label=L"\sim N_x^{-2}")

legend()

title("Convergence for freely-decaying Taylor-Green vortex")
removespines("top", "right")
ylabel("Norms of the absolute error, \$ | u_\\mathrm{simulation} - u_\\mathrm{analytical} | \$")
xlabel(L"N_x")
xticks(sort(Nx), ["\$ 2^{$(round(Int, log2(n)))} \$" for n in sort(Nx)])

savefig("figs/taylor_green_convergence.png", dpi=480)
