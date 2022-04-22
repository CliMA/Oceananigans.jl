if haskey(ENV, "CI") && ENV["CI"] == "true"
    ENV["PYTHON"] = ""
    using Pkg
    Pkg.build("PyCall")
end

using PyPlot
using Glob
using Oceananigans

using ConvergenceTests
using ConvergenceTests.ForcedFlowFreeSlip: u

arch = CUDA.has_cuda() ? GPU() : CPU()

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

    loglog(Nx, L₁, color=defaultcolors[i], alpha=0.6, mfc="None",
           linestyle="None", marker="o", label="\$L_1\$-norm, $name")
    loglog(Nx, L∞, color=defaultcolors[i], alpha=0.6, mfc="None",
           linestyle="None", marker="^", label="\$L_\\infty\$-norm, $name")
end

L₁ = map(err -> err.L₁, errors[1])
loglog(Nx, L₁[end] * (Nx[end] ./ Nx).^2, "k-", linewidth=1, alpha=0.6, label=L"\sim N_x^{-2}")

legend()
xscale("log", base=2)
yscale("log", base=10)
xlabel(L"N_x")
ylabel("Norms of the absolute error, \$ | u_{\\mathrm{sim}} - u_{\\mathrm{exact}} | \$")
title("Convergence for forced free slip")
removespines("top", "right")

filename = "forced_free_slip_convergence_$(typeof(arch)).png"
filepath = joinpath(@__DIR__, "figs", filename)
mkpath(dirname(filepath))
savefig(filepath, dpi=480)

p = sortperm(Nx)
for (name, error) in zip(names, errors)
    L₁ = map(e -> e.L₁, error)
    L∞ = map(e -> e.L∞, error)
    name = "Forced flow free slip " * strip(name.s, '$')
    test_rate_of_convergence(L₁[p], Nx[p], expected=-2.0, atol=0.001, name=name * " L₁")
    test_rate_of_convergence(L∞[p], Nx[p], expected=-2.0, atol=0.005, name=name * " L∞")
end
