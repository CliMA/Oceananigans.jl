using PyPlot, Glob

include("ConvergenceTests/ConvergenceTests.jl")

using .ConvergenceTests
using .ConvergenceTests.ForcedFlowFreeSlip: u

defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
removespine(side) = gca().spines[side].set_visible(false)
removespines(sides...) = [removespine(side) for side in sides]

xy_filenames = glob("forced_free_slip_xy*.jld2", joinpath(@__DIR__, "data"))
xz_filenames = glob("forced_free_slip_xz*.jld2", joinpath(@__DIR__, "data"))

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

    loglog(Nx, L₁, basex=2, color=defaultcolors[i], alpha=0.6, mfc="None",
           linestyle="None", marker="o", label="\$L_1\$-norm, $name")
    loglog(Nx, L∞, basex=2, color=defaultcolors[i], alpha=0.6, mfc="None",
           linestyle="None", marker="^", label="\$L_\\infty\$-norm, $name")
end

L₁ = map(err -> err.L₁, errors[1])
loglog(Nx, L₁[end] * (Nx[end] ./ Nx).^2, "k-", basex=2, linewidth=1, alpha=0.6, label=L"\sim N_x^{-2}")

legend()
xlabel(L"N_x")
ylabel("Norms of the absolute error, \$ | u_{\\mathrm{sim}} - u_{\\mathrm{exact}} | \$")
removespines("top", "right")
title("Convergence for forced free slip")

filepath = joinpath(@__DIR__, "figs", "forced_free_slip_convergence.png")
savefig(filepath, dpi=480)

for (name, error) in zip(names, errors)
    L₁ = map(e -> e.L₁, error)
    L∞ = map(e -> e.L∞, error)
    name = "Forced flow free slip " * strip(name.s, '$')
    test_rate_of_convergence(L₁, Nx, expected=-2.0, atol=0.001, name=name * " L₁")
    test_rate_of_convergence(L∞, Nx, expected=-2.0, atol=0.001, name=name * " L∞")
end
