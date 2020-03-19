using PyPlot, Glob, JLD2

defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

include("ConvergenceTests/ConvergenceTests.jl")

free_slip_simulation_files = glob("data/free_decay*.jld2")

#=
function max_u_timeseries(filename)
    iters = ConvergenceTests.iterations(filename)
    max_u = zeros(length(iters))
    max_v = zeros(length(iters))
    t = zeros(length(iters))

    file = jldopen(filename)
    for (i, iter) = enumerate(iters)
        t[i] = file["timeseries/t/$iter"]
        max_u[i] = maximum(file["timeseries/u/$iter"])
        max_v[i] = maximum(file["timeseries/v/$iter"])
    end
    close(file)

    return max_u, max_v, t
end

function collect_max_u(filenames...)
    max_u = []
    max_v = []
    t = []
    grids = []

    for filename in filenames
        timeseries = max_u_timeseries(filename)
        push!(max_u, timeseries[1])
        push!(max_v, timeseries[2])
        push!(t, timeseries[3])
        push!(grids, ConvergenceTests.RegularCartesianGrid(filename))
    end

    Nx = map(g -> g.Nx, grids)

    return max_u, max_v, t, Nx
end


max_u, max_v, t, Nx = collect_max_u(free_slip_simulation_files...)

close("all")
fig, axs = subplots()

for i = 1:length(max_u)
    Nxi = Nx[i]
    ti = t[i]

    simulated_max_u = max_u[i]
    simulated_max_v = max_v[i]

    analytical_max_u = @. 1 + exp(-2ti)
    analytical_max_v = @. exp(-2ti)

    plot(ti, (simulated_max_u .- analytical_max_u) ./ analytical_max_u, "-",  color=defaultcolors[i], label="\$ N_x = \$ $Nxi")
    plot(ti, (simulated_max_v .- analytical_max_v) ./ analytical_max_v, "--", color=defaultcolors[i], label="\$ N_x = \$ $Nxi")
end

legend()

#plot(t[end], 1 .+ exp.(-2t[end]), "k-", linewidth=2, alpha=0.6)
=#

errors = ConvergenceTests.compute_errors((x, y, z, t) -> ConvergenceTests.DoublyPeriodicFreeDecay.u(x, y, t), 
                                         free_slip_simulation_files...)

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
