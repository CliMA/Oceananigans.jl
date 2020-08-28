using PyPlot, Glob, Printf

include("ConvergenceTests/ConvergenceTests.jl")

defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
removespine(side) = gca().spines[side].set_visible(false)
removespines(sides...) = [removespine(side) for side in sides]

u = ConvergenceTests.ForcedFlowFixedSlip.u

xy_filenames = glob("data/forced_fixed_slip_xy*")
xz_filenames = glob("data/forced_fixed_slip_xz*")

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

    ax.plot(Nx, L₁, color=defaultcolors[j], alpha=0.6, mfc="None",
            linestyle="None", marker="o", label="\$L_1\$-norm, $(labels[j])")

    ax.plot(Nx, L∞, color=defaultcolors[j], alpha=0.6, mfc="None",
            linestyle="None", marker="^", label="\$L_\\infty\$-norm, $(labels[j])")

    if j == 2
        L₁ = map(err -> err.L₁, errors)
        ii = sortperm(Nx)
        Nx = Nx[ii]
        L₁ = L₁[ii]
        ax.plot(Nx, L₁[end] * (Nx[end] ./ Nx).^2, "k-", linewidth=1, alpha=0.6, label=L"\sim N_x^{-2}")
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

ax.set_xscale("log")
ax.set_yscale("log")

xticks(sort(Nx), ["\$ 2^{$(round(Int, log2(n)))} \$" for n in sort(Nx)])

savefig("figs/forced_fixed_slip_convergence.png", dpi=480)

