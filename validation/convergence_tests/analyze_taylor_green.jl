if haskey(ENV, "CI") && ENV["CI"] == "true"
    ENV["PYTHON"] = ""
    using Pkg
    Pkg.build("PyCall")
end

using PyPlot
using Glob
using JLD2
using Oceananigans

using ConvergenceTests
using ConvergenceTests: compute_errors, extract_sizes

arch = CUDA.functional() ? GPU() : CPU()

defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
removespine(side) = gca().spines[side].set_visible(false)
removespines(sides...) = [removespine(side) for side in sides]

filenames = glob("taylor_green*.jld2", joinpath(@__DIR__, "data"))

errors = compute_errors(
            (x, y, z, t) -> ConvergenceTests.DoublyPeriodicTaylorGreen.u(x, y, t),
            filenames...)

sizes = extract_sizes(filenames...)

Nx = map(sz -> sz[1], sizes)
L₁ = map(err -> err.L₁, errors)
L∞ = map(err -> err.L∞, errors)

close("all")
fig, ax = subplots()

ax.loglog(Nx, L₁, linestyle="None", marker="o", label="error, \$L_1\$-norm")
ax.loglog(Nx, L∞, linestyle="None", marker="^", label="error, \$L_\\infty\$-norm")
ax.loglog(Nx, L₁[end] * (Nx[end] ./ Nx).^2, "k-", linewidth=1, alpha=0.6,  label=L"\sim N_x^{-2}")

ax.set_xscale("log", base=2)
ax.set_yscale("log", base=10)

legend()

title("Convergence for freely-decaying Taylor-Green vortex")
removespines("top", "right")
ylabel("Norms of the absolute error, \$ | u_\\mathrm{simulation} - u_\\mathrm{analytical} | \$")
xlabel(L"N_x")

filename = "taylor_green_convergence_$(typeof(arch)).png"
filepath = joinpath(@__DIR__, "figs", filename)
mkpath(dirname(filepath))
savefig(filepath, dpi=480)

p = sortperm(Nx)
test_rate_of_convergence(L₁[p], Nx[p], expected=-2.0, atol=0.05, name="Taylor-Green L₁")
test_rate_of_convergence(L∞[p], Nx[p], expected=-2.0, atol=0.05, name="Taylor-Green L∞")
