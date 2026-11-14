# A hat advected into a cylinder in a doubly periodic box, with and without the bounds-preserving limiter.
#
# The obstacle is the point of the setup: next to an immersed boundary the WENO stencil falls back to its
# low-order buffer schemes, which is where a high-order reconstruction overshoots. In uniform flow the same hat
# stays within 1e-4 of its bounds and there is nothing to see. The pressure solve is the conjugate-gradient one
# so that the advecting velocity is divergence-free next to the boundary, without which no bounds-preserving
# scheme can hold the bound there.
#
# Δt comes from λˣ + λʸ ≤ 5/18, and `stop_time` is when the hat wraps the cylinder and the excursion is largest.
# The strict bound holds for a forward Euler or SSP update; the model's `SplitRungeKutta3` is neither, so what
# this shows is how far the limiter suppresses the excursions in practice.

using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Printf

N = 128
stop_time = 0.52
maximum_speed = 2

underlying_grid = RectilinearGrid(size=(N, N), x=(0, 1), y=(0, 1), halo=(6, 6), topology=(Periodic, Periodic, Flat))
obstacle(x, y) = hypot(x - 0.55, y - 0.55) < 0.1
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBoundary(obstacle))

hat(x, y) = max(0, 1 - hypot(x - 0.2, y - 0.2) / 0.12)

steps = ceil(Int, stop_time * maximum_speed / ((5//18) * minimum_xspacing(grid)))
Δt = stop_time / steps

function advect_hat(advection)
    model = NonhydrostaticModel(grid; advection, tracers = :c,
                                pressure_solver = ConjugateGradientPoissonSolver(grid))
    set!(model, u=1, v=1, c=hat)

    tracer_minimum = zeros(steps)
    tracer_maximum = zeros(steps)
    for step in 1:steps
        time_step!(model, Δt)
        tracer_minimum[step] = minimum(interior(model.tracers.c))
        tracer_maximum[step] = maximum(interior(model.tracers.c))
    end

    return model.tracers.c, tracer_minimum, tracer_maximum
end

unlimited, unlimited_minimum, unlimited_maximum = advect_hat(WENO(order=7))
limited, limited_minimum, limited_maximum = advect_hat(WENO(order=7, bounds=(0, 1)))

@printf("%-22s  min %+.4e   max %.6f\n", "WENO(7)", minimum(unlimited_minimum), maximum(unlimited_maximum))
@printf("%-22s  min %+.4e   max %.6f\n", "WENO(7, bounds=(0, 1))", minimum(limited_minimum), maximum(limited_maximum))

using CairoMakie

x = xnodes(grid, Center())
times = Δt .* (1:steps)
masked(c) = [obstacle(xi, yj) ? NaN : interior(c)[i, j, 1] for (i, xi) in enumerate(x), (j, yj) in enumerate(x)]

fig = Figure(size=(1150, 850))

# `:balance` is centred on zero over a symmetric range, so anything below the lower bound reads as blue.
for (column, (name, c)) in enumerate((("WENO(7)", unlimited), ("WENO(7, bounds=(0, 1))", limited)))
    ax = Axis(fig[1, column], title=name, xlabel="x", ylabel="y", aspect=1, limits=(0.3, 0.95, 0.3, 0.95))
    plot = heatmap!(ax, x, x, masked(c), colormap=:balance, colorrange=(-1, 1), nan_color=:gray)
    column == 2 && Colorbar(fig[1, 3], plot, label="c")
end

ax = Axis(fig[2, 1:3], title="Tracer minimum", xlabel="t", ylabel="min(c)")
lines!(ax, times, unlimited_minimum, color=:crimson, label="WENO(7)")
lines!(ax, times, limited_minimum, color=:seagreen, label="WENO(7, bounds=(0, 1))")
hlines!(ax, [0], color=:gray, linewidth=0.5)
axislegend(ax, position=:lb)

save("two_dimensional_hat_advection.png", fig)
