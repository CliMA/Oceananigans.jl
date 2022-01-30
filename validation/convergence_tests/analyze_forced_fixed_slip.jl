if haskey(ENV, "CI") && ENV["CI"] == "true"
    ENV["PYTHON"] = ""
    using Pkg
    Pkg.build("PyCall")
end

using Printf
using Glob
using PyPlot
using Oceananigans

using ConvergenceTests
using ConvergenceTests.ForcedFlowFixedSlip: u

arch = CUDA.functional() ? GPU() : CPU()

defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
removespine(side) = gca().spines[side].set_visible(false)
removespines(sides...) = [removespine(side) for side in sides]

xy_filenames = glob("forced_fixed_slip_xy*", joinpath(@__DIR__, "data"))
xz_filenames = glob("forced_fixed_slip_xz*", joinpath(@__DIR__, "data"))

labels = ["(x, y)", "(x, z)"]

filenameses = [ xy_filenames,
                xz_filenames ]

errorses = [ ConvergenceTests.compute_errors((x, y, z, t) -> u(x, y, t), xy_filenames...),
             ConvergenceTests.compute_errors((x, y, z, t) -> u(x, z, t), xz_filenames...) ]

sizeses = [ ConvergenceTests.extract_sizes(xy_filenames...),
            ConvergenceTests.extract_sizes(xz_filenames...) ]

close("all")
fig, ax = subplots()

for j = 1:2
    filenames = filenameses[j]
    errors = errorses[j]
    sizes = sizeses[j]

    Nx = map(sz -> sz[1], sizes)

    L₁ = map(err -> err.L₁, errors)
    L∞ = map(err -> err.L∞, errors)

    @show size(L₁) size(Nx)

    ax.loglog(Nx, L₁, color=defaultcolors[j], alpha=0.6, mfc="None",
            linestyle="None", marker="o", label="\$L_1\$-norm, $(labels[j])")

    ax.loglog(Nx, L∞, color=defaultcolors[j], alpha=0.6, mfc="None",
            linestyle="None", marker="^", label="\$L_\\infty\$-norm, $(labels[j])")

    if j == 2
        L₁ = map(err -> err.L₁, errors)
        ii = sortperm(Nx)
        Nx = Nx[ii]
        L₁ = L₁[ii]
        ax.loglog(Nx, L₁[end] * (Nx[end] ./ Nx).^2, "k-", linewidth=1, alpha=0.6, label=L"\sim N_x^{-2}")
    end

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
end

filenames = filenameses[1]
errors = errorses[1]
sizes = sizeses[1]
Nx = map(sz -> sz[1], sizes)

legend()

xlabel(L"N_x")
ylabel("Norms of the absolute error, \$ | u_{\\mathrm{sim}} - u_{\\mathrm{exact}} | \$")

removespines("top", "right")
title("Convergence for forced fixed slip")

filename = "forced_fixed_slip_convergence_$(typeof(arch)).png"
filepath = joinpath(@__DIR__, "figs", filename)
mkpath(dirname(filepath))
savefig(filepath, dpi=480)

p = sortperm(Nx)
for (label, error) in zip(labels, errorses)
    L₁ = map(e -> e.L₁, error)
    L∞ = map(e -> e.L∞, error)
    name = "Forced flow fixed slip " * label
    test_rate_of_convergence(L₁[p], Nx[p], expected=-2.0, atol=0.05, name=name * " L₁")
    test_rate_of_convergence(L∞[p], Nx[p], expected=-2.0, atol=0.10, name=name * " L∞")
end

