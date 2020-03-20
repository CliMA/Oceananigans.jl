using PyPlot, Glob, JLD2

using Oceananigans, Oceananigans.Fields, Oceananigans.Grids

using Oceananigans.Fields: nodes

defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

include("ConvergenceTests/ConvergenceTests.jl")

function extract_final_c(filename)
    grid = RegularCartesianGrid(filename)
    iters = ConvergenceTests.iterations(filename)
    last_iter = iters[end]
    loc = location(:c)

    c_data = ConvergenceTests.field_data(filename, :c, iters[end])
    c_simulation = Field{loc[1], loc[2], loc[3]}(c_data, grid, FieldBoundaryConditions(grid, loc))

    c_3d(x, y, z, t) = ConvergenceTests.TwoDimensionalDiffusion.c(x, y, t)
    x, y, z, t = (nodes(c_simulation)..., ConvergenceTests.iteration_time(filename, iters[end]))
    c_analytical = c_3d.(x, y, z, t)

    return c_simulation, c_analytical
end

filenames = glob("data/Periodic_Bounded*.jld2")

errors = ConvergenceTests.compute_errors((x, y, z, t) -> ConvergenceTests.TwoDimensionalDiffusion.c(x, y, t), 
                                         filenames...; name=:c)

sizes = ConvergenceTests.extract_sizes(filenames...)

Nx = map(sz -> sz[1], sizes)
L₁ = map(err -> err.L₁, errors)
L∞ = map(err -> err.L∞, errors)

close("all")
fig, ax = subplots()

ax.tick_params(bottom=false, labelbottom=false)

ax.loglog(Nx, L₁, linestyle="None", marker="o", label="error, \$L_1\$-norm")
ax.loglog(Nx, L∞, linestyle="None", marker="^", label="error, \$L_\\infty\$-norm")

ax.loglog(Nx, L₁[1] * (Nx[1] ./ Nx).^2, "k--", linewidth=1, alpha=0.6, label=L"\sim N_x^{-2}")
ax.loglog(Nx, L₁[1] * Nx[1] ./ Nx,      "k-",  linewidth=1, alpha=0.6, label=L"\sim N_x^{-1}")

legend()

title("Convergence for two dimensional diffusion")
ylabel("Norms of the absolute error, \$ | u_\\mathrm{simulation} - u_\\mathrm{analytical} | \$")
xlabel(L"N_x")


fig, axs = subplots(nrows=2, ncols=length(filenames), figsize=(16, 6))

for (i, filename) in enumerate(filenames)
    c_sim, c_ana = extract_final_c(filename)

    sca(axs[1, i])
    imshow(interior(c_sim)[:, :, 1])
    title("\$ N_x = \$ $(Nx[i])")

    sca(axs[2, i])
    imshow(abs.(c_ana[:, :, 1] .- interior(c_sim)[:, :, 1]))
    title("error")

end

for ax in axs
    ax.tick_params(left=false, bottom=false, labelleft=false, labelbottom=false)
end
