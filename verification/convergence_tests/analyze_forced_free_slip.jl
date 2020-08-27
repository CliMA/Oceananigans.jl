using PyPlot, Glob

include("ConvergenceTests/ConvergenceTests.jl")

defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
removespine(side) = gca().spines[side].set_visible(false)
removespines(sides...) = [removespine(side) for side in sides]

u = ConvergenceTests.ForcedFlowFreeSlip.u

xy_filenames = glob("data/forced_free_slip_xy*.jld2")
xz_filenames = glob("data/forced_free_slip_xz*.jld2")

xy_errors = ConvergenceTests.compute_errors((x, y, z, t) -> u(x, y, t), xy_filenames...)
xz_errors = ConvergenceTests.compute_errors((x, y, z, t) -> u(x, z, t), xz_filenames...)

sizes = ConvergenceTests.extract_sizes(xy_filenames...)

Nx = map(sz -> sz[1], sizes)

names = (L"(x, y)", L"(x, z)")
errors = (xy_errors, xz_errors)

close("all")
fig, ax = subplots()

for i = 1:length(errors)

    error = errors[i]
    name = names[i]

    L₁ = map(err -> err.L₁, error)
    L∞ = map(err -> err.L∞, error)

    loglog(Nx, L₁, color=defaultcolors[i], alpha=0.6, mfc="None",
           linestyle="None", marker="o", label="\$L_1\$-norm, $name")

    loglog(Nx, L∞, color=defaultcolors[i], alpha=0.6, mfc="None",
           linestyle="None", marker="^", label="\$L_\\infty\$-norm, $name")
end

L₁ = map(err -> err.L₁, errors[1])
loglog(Nx, L₁[end] * (Nx[end] ./ Nx).^2, "k-", linewidth=1, alpha=0.6, label=L"\sim N_x^{-2}")

legend()
xlabel(L"N_x")
ylabel("Norms of the absolute error, \$ | u_{\\mathrm{sim}} - u_{\\mathrm{exact}} | \$")
removespines("top", "right")
title("Convergence for forced free slip")
xticks(sort(Nx), ["\$ 2^{$(round(Int, log2(n)))} \$" for n in sort(Nx)])

savefig("figs/forced_free_slip_convergence.png", dpi=480)
