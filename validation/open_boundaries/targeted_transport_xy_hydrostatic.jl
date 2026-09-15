# Hydrostatic 2D (xy) flows driven by `target_transport` on Flather (`GravityWaveRadiation`) open boundaries:
# the barotropic counterpart of `targeted_transport_xy_nonhydrostatic.jl`.
#
# Every side is open, with Flather on the barotropic transport towards an exterior at rest, Chapman on η and
# a radiating `NormalRadiation` on the baroclinic velocity, and the model starts from rest. A side given a
# number has that net transport pinned through `target_transport` on the Flather condition (positive in the
# positive coordinate direction); a side given `nothing` is left to the Flather condition alone, which lets
# any net imbalance in or out as the free surface adjusts. In the unbalanced case the south and north sides
# therefore each supply half of the missing inflow, as the pool sides do in the nonhydrostatic model.

using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: NormalRadiation, GravityWaveRadiationBoundaryCondition, SurfaceWaveRadiationBoundaryCondition
using CairoMakie
using Printf
using Statistics: mean

Lx = Ly = 50kilometers
H  = 20                       # depth [m]
Nx = Ny = 64
U = 0.05                      # speed through a fully open side [m s⁻¹]
Q = U * H * Ly                # transport through a whole side [m³ s⁻¹]
Δt = 2minutes
stop_time = 1day
substeps = 8
ν = 10                        # horizontal viscosity [m² s⁻¹]

experiments = ("west → east"           => (west = Q, east = Q,       south = nothing, north = nothing),
               "west → north"          => (west = Q, east = nothing, south = nothing, north = Q),
               "diagonal"              => (west = Q, east = nothing, south = Q,       north = nothing),
               "targets don't balance" => (west = Q, east = 2Q,      south = nothing, north = nothing))

grid = RectilinearGrid(CPU(); size = (Nx, Ny, 1), x = (0, Lx), y = (0, Ly), z = (-H, 0),
                       topology = (Bounded, Bounded, Bounded))

flather(target) = GravityWaveRadiationBoundaryCondition((0, 0); target_transport = target)
baroclinic()    = NormalFlowBoundaryCondition(0; scheme = NormalRadiation(; inflow_timescale = Inf, outflow_timescale = Inf))
chapman()       = SurfaceWaveRadiationBoundaryCondition()

function run_experiment(targets)
    boundary_conditions = (U = FieldBoundaryConditions(west = flather(targets.west), east = flather(targets.east)),
                           V = FieldBoundaryConditions(south = flather(targets.south), north = flather(targets.north)),
                           u = FieldBoundaryConditions(west = baroclinic(), east = baroclinic()),
                           v = FieldBoundaryConditions(south = baroclinic(), north = baroclinic()),
                           η = FieldBoundaryConditions(west = chapman(), east = chapman(), south = chapman(), north = chapman()))
    model = HydrostaticFreeSurfaceModel(grid; boundary_conditions, free_surface = SplitExplicitFreeSurface(grid; substeps),
                                        closure = HorizontalScalarDiffusivity(; ν), buoyancy = nothing, tracers = ())
    run!(Simulation(model; Δt, stop_time, verbose = false))
    return model
end

models = [run_experiment(targets) for (name, targets) in experiments]

Δx, Δy = Lx / Nx, Ly / Ny
fmt(target) = isnothing(target) ? "free" : @sprintf("%+.1f Q", target / Q)

println("Final barotropic transports in units of Q:")
for ((name, targets), model) in zip(experiments, models)
    Uᵇ, Vᵇ = model.free_surface.barotropic_velocities
    @printf("%-22s west %+.3f  east %+.3f  south %+.3f  north %+.3f\n", name,
            sum(interior(Uᵇ, 1, :, 1)) * Δy / Q, sum(interior(Uᵇ, Nx + 1, :, 1)) * Δy / Q,
            sum(interior(Vᵇ, :, 1, 1)) * Δx / Q, sum(interior(Vᵇ, :, Ny + 1, 1)) * Δx / Q)
end

xc = xnodes(grid, Center()) / kilometers
yc = ynodes(grid, Center()) / kilometers
stride = 4
fig = Figure(size = (1000, 340 * length(experiments)))
Label(fig[0, 1:4], "Hydrostatic, Flather with target_transport"; fontsize = 18, font = :bold)

for (row, ((name, targets), model)) in enumerate(zip(experiments, models))
    u, v, _ = model.velocities
    uᶜ = (interior(u, 1:Nx, :, 1) .+ interior(u, 2:Nx+1, :, 1)) ./ 2
    vᶜ = (interior(v, :, 1:Ny, 1) .+ interior(v, :, 2:Ny+1, 1)) ./ 2
    η = interior(model.free_surface.displacement, :, :, 1)
    η′ = η .- mean(η)

    title = @sprintf("%s:  W %s,  E %s,  S %s,  N %s", name, fmt(targets.west), fmt(targets.east), fmt(targets.south), fmt(targets.north))
    ax = Axis(fig[row, 1]; title, xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect())
    hm = heatmap!(ax, xc, yc, sqrt.(uᶜ.^2 .+ vᶜ.^2); colormap = :viridis, colorrange = (0, 2U))
    arrows2d!(ax, xc[1:stride:end], yc[1:stride:end], uᶜ[1:stride:end, 1:stride:end], vᶜ[1:stride:end, 1:stride:end];
              lengthscale = 0.5 * stride * Δx / kilometers / U, color = :white)
    Colorbar(fig[row, 2], hm; label = "speed (m s⁻¹)")

    ax = Axis(fig[row, 3]; title = @sprintf("η − ⟨η⟩,  ⟨η⟩ = %+.1f cm", 100 * mean(η)), xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect())
    hm = heatmap!(ax, xc, yc, η′; colormap = :balance, colorrange = max(maximum(abs, η′), 1e-6) .* (-1, 1))
    Colorbar(fig[row, 4], hm; label = "m")
end

save("targeted_transport_xy_hydrostatic.png", fig)
