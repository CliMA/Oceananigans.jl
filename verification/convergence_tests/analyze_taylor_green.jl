using PyPlot
using Glob
using JLD2

defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
removespine(side) = gca().spines[side].set_visible(false)
removespines(sides...) = [removespine(side) for side in sides]

include("ConvergenceTests/ConvergenceTests.jl")

using .ConvergenceTests
using .ConvergenceTests: compute_errors, extract_sizes

filenames = glob(joinpath(@__DIR__, "data", "taylor_green*.jld2"))

errors = compute_errors(
            (x, y, z, t) -> ConvergenceTests.DoublyPeriodicTaylorGreen.u(x, y, t),
            filenames...)

sizes = extract_sizes(filenames...)

Nx = map(sz -> sz[1], sizes)
L₁ = map(err -> err.L₁, errors)
L∞ = map(err -> err.L∞, errors)

close("all")
fig, ax = subplots()

ax.loglog(Nx, L₁, basex=2, linestyle="None", marker="o", label="error, \$L_1\$-norm")
ax.loglog(Nx, L∞, basex=2, linestyle="None", marker="^", label="error, \$L_\\infty\$-norm")
ax.loglog(Nx, L₁[end] * (Nx[end] ./ Nx).^2, "k-", basex=2, linewidth=1, alpha=0.6,  label=L"\sim N_x^{-2}")

legend()

title("Convergence for freely-decaying Taylor-Green vortex")
removespines("top", "right")
ylabel("Norms of the absolute error, \$ | u_\\mathrm{simulation} - u_\\mathrm{analytical} | \$")
xlabel(L"N_x")

filepath = joinpath(@__DIR__, "figs", "taylor_green_convergence.png")
savefig(filepath, dpi=480)

test_rate_of_convergence(L₁, Nx, expected=-2.0, atol=0.001, name="Taylor-Green L₁")
test_rate_of_convergence(L∞, Nx, expected=-2.0, atol=0.050, name="Taylor-Green L∞")
