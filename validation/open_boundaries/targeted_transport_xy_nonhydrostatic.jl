# Nonhydrostatic 2D (xy) flows driven by `target_transport` on `NormalRadiation` open boundaries: the
# counterpart of the hydrostatic Flather version in `targeted_transport_xy_hydrostatic_flather.jl`.
#
# Every side is open, the exterior is at rest and the model starts from rest. A side given a number has
# that net transport pinned (per unit depth, positive in the positive coordinate direction); a side given
# `nothing` has no target and absorbs whatever net inflow the pinned sides leave over, since the pressure
# solve needs zero net transport. In the unbalanced case the south and north sides therefore each supply
# half of the missing inflow.

using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: NormalRadiation
using CairoMakie
using Printf
using Statistics: mean

Lx = Ly = 50kilometers
Nx = Ny = 64
U = 0.05                      # speed through a fully open side [m s⁻¹]
Q = U * Ly                    # transport per unit depth through a whole side [m² s⁻¹]
Δt = 2minutes
stop_time = 1day
ν = 10                        # viscosity [m² s⁻¹]

experiments = ("west → east"           => (west = Q, east = Q,       south = nothing, north = nothing),
               "west → north"          => (west = Q, east = nothing, south = nothing, north = Q),
               "diagonal"              => (west = Q, east = nothing, south = Q,       north = nothing),
               "targets don't balance" => (west = Q, east = 2Q,      south = nothing, north = nothing))

grid = RectilinearGrid(CPU(); size = (Nx, Ny), x = (0, Lx), y = (0, Ly), halo = (4, 4),
                       topology = (Bounded, Bounded, Flat))

open_boundary(target) = NormalFlowBoundaryCondition(0; scheme = NormalRadiation(; inflow_timescale = Inf, outflow_timescale = Inf,
                                                                                  target_transport = target))

function run_experiment(targets)
    boundary_conditions = (u = FieldBoundaryConditions(west = open_boundary(targets.west), east = open_boundary(targets.east)),
                           v = FieldBoundaryConditions(south = open_boundary(targets.south), north = open_boundary(targets.north)))
    # AB2 rather than RK3: on the uniform diagonal outflow RK3 triggers the `NormalRadiation` phase-speed artefact
    # shown in `targeted_transport_diagonal_flow.jl`, which has nothing to do with `target_transport`.
    model = NonhydrostaticModel(grid; boundary_conditions, advection = WENO(order = 5), closure = ScalarDiffusivity(; ν),
                                timestepper = :QuasiAdamsBashforth2)
    run!(Simulation(model; Δt, stop_time, verbose = false))
    return model
end

models = [run_experiment(targets) for (name, targets) in experiments]

Δx, Δy = Lx / Nx, Ly / Ny
g = Oceananigans.defaults.gravitational_acceleration
fmt(target) = isnothing(target) ? "free" : @sprintf("%+.1f Q", target / Q)

println("Final transports in units of Q:")
for ((name, targets), model) in zip(experiments, models)
    u, v, _ = model.velocities
    @printf("%-22s west %+.3f  east %+.3f  south %+.3f  north %+.3f\n", name,
            sum(interior(u, 1, :, 1)) * Δy / Q, sum(interior(u, Nx + 1, :, 1)) * Δy / Q,
            sum(interior(v, :, 1, 1)) * Δx / Q, sum(interior(v, :, Ny + 1, 1)) * Δx / Q)
end

xc = xnodes(grid, Center()) / kilometers
yc = ynodes(grid, Center()) / kilometers
stride = 4
fig = Figure(size = (1000, 340 * length(experiments)))
Label(fig[0, 1:4], "Nonhydrostatic, NormalRadiation with target_transport"; fontsize = 18, font = :bold)

for (row, ((name, targets), model)) in enumerate(zip(experiments, models))
    u, v, _ = model.velocities
    uᶜ = (interior(u, 1:Nx, :, 1) .+ interior(u, 2:Nx+1, :, 1)) ./ 2
    vᶜ = (interior(v, :, 1:Ny, 1) .+ interior(v, :, 2:Ny+1, 1)) ./ 2
    p = interior(model.pressures.pNHS, :, :, 1) ./ g
    p′ = p .- mean(p)

    title = @sprintf("%s:  W %s,  E %s,  S %s,  N %s", name, fmt(targets.west), fmt(targets.east), fmt(targets.south), fmt(targets.north))
    ax = Axis(fig[row, 1]; title, xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect())
    hm = heatmap!(ax, xc, yc, sqrt.(uᶜ.^2 .+ vᶜ.^2); colormap = :viridis, colorrange = (0, 2U))
    arrows2d!(ax, xc[1:stride:end], yc[1:stride:end], uᶜ[1:stride:end, 1:stride:end], vᶜ[1:stride:end, 1:stride:end];
              lengthscale = 0.5 * stride * Δx / kilometers / U, color = :white)
    Colorbar(fig[row, 2], hm; label = "speed (m s⁻¹)")

    ax = Axis(fig[row, 3]; title = "(p − ⟨p⟩) / g", xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect())
    hm = heatmap!(ax, xc, yc, p′; colormap = :balance, colorrange = max(maximum(abs, p′), 1e-6) .* (-1, 1))
    Colorbar(fig[row, 4], hm; label = "m")
end

save("targeted_transport_xy_nonhydrostatic.png", fig)
