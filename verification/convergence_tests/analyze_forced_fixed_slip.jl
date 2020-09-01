using PyPlot, Glob, Printf

include("ConvergenceTests/ConvergenceTests.jl")

using .ConvergenceTests
using .ConvergenceTests.ForcedFlowFixedSlip: u

defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
removespine(side) = gca().spines[side].set_visible(false)
removespines(sides...) = [removespine(side) for side in sides]

xy_filenames = glob(joinpath(@__DIR__, "data", "forced_fixed_slip_xy*"))
xz_filenames = glob(joinpath(@__DIR__, "data", "forced_fixed_slip_xz*"))

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

    ax.loglog(Nx, L₁, basex=2, color=defaultcolors[j], alpha=0.6, mfc="None",
            linestyle="None", marker="o", label="\$L_1\$-norm, $(labels[j])")

    ax.loglog(Nx, L∞, basex=2, color=defaultcolors[j], alpha=0.6, mfc="None",
            linestyle="None", marker="^", label="\$L_\\infty\$-norm, $(labels[j])")

    if j == 2
        L₁ = map(err -> err.L₁, errors)
        ii = sortperm(Nx)
        Nx = Nx[ii]
        L₁ = L₁[ii]
        ax.loglog(Nx, L₁[end] * (Nx[end] ./ Nx).^2, "k-", basex=2, linewidth=1, alpha=0.6, label=L"\sim N_x^{-2}")
    end
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

filepath = joinpath(@__DIR__, "figs", "forced_fixed_slip_convergence.png")
savefig(filepath, dpi=480)

for (label, error) in zip(labels, errorses)
    L₁ = map(e -> e.L₁, error)
    L∞ = map(e -> e.L∞, error)
    name = "Forced fixed slip " * label
    test_rate_of_convergence(L₁, Nx, expected=-2.0, atol=Inf, name=name * " L₁")
    test_rate_of_convergence(L∞, Nx, expected=-2.0, atol=Inf, name=name * " L∞")
end
