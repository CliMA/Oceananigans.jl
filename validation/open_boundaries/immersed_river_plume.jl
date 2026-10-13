# A river plume: fresh water enters a rotating coastal ocean through a river mouth cut into an
# immersed coastline (a `NormalFlow` immersed boundary condition on `v` and a `Value` condition on
# salinity), and the free surface drains through radiating open boundaries on the sea side.

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: xnode, ynode, znode
using CUDA
using Oceananigans.BoundaryConditions: NormalRadiation, GravityWaveRadiationBoundaryCondition, SurfaceWaveRadiationBoundaryCondition
using CairoMakie
using Printf
using Statistics: mean

arch = GPU()
Lx, Ly, Lz = 60kilometers, 40kilometers, 30
Nx, Ny, Nz = 240, 160, 15

coast = 5kilometers      # land occupies y < coast
mouth = 4kilometers      # width of the river mouth, centered at x = Lx / 2
river_depth = 5          # [m]
V_river = 0.5            # river velocity [m s⁻¹]
S_ocean = 32             # ambient salinity [g kg⁻¹]
f = 1e-4                 # Coriolis parameter [s⁻¹]
stop_time = 5days

grid = RectilinearGrid(arch; size = (Nx, Ny, Nz), x = (0, Lx), y = (0, Ly), z = (-Lz, 0),
                       topology = (Bounded, Bounded, Bounded))
grid = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> ifelse(y < coast, 0, -Lz)))

@inline in_mouth(x, z) = abs(x - Lx / 2) < mouth / 2 && z > -river_depth
@inline river_velocity(i, j, k, grid, clock, fields) = ifelse(in_mouth(xnode(i, grid, Center()), znode(k, grid, Center())), V_river, zero(V_river))

flather(side) = GravityWaveRadiationBoundaryCondition((0, 0))
baroclinic() = NormalFlowBoundaryCondition(0; scheme = NormalRadiation(; inflow_timescale = Inf, outflow_timescale = Inf))
chapman() = SurfaceWaveRadiationBoundaryCondition()

v_bcs = FieldBoundaryConditions(immersed = ImmersedBoundaryCondition(south = NormalFlowBoundaryCondition(river_velocity, discrete_form = true)),
                                north = baroclinic())
u_bcs = FieldBoundaryConditions(west = baroclinic(), east = baroclinic())
U_bcs = FieldBoundaryConditions(west = flather(:west), east = flather(:east))
V_bcs = FieldBoundaryConditions(north = flather(:north))
η_bcs = FieldBoundaryConditions(west = chapman(), east = chapman(), north = chapman())
S_bcs = FieldBoundaryConditions(immersed = ImmersedBoundaryCondition(south = ValueBoundaryCondition(0)))

closure = (VerticalScalarDiffusivity(ν = 1e-3, κ = 1e-4), HorizontalScalarDiffusivity(ν = 10, κ = 10))
free_surface = SplitExplicitFreeSurface(grid; substeps = 30)
buoyancy = SeawaterBuoyancy(constant_temperature = 10)

model = HydrostaticFreeSurfaceModel(grid; free_surface, closure, buoyancy, coriolis = FPlane(; f),
                                    momentum_advection = WENO(order = 5), tracer_advection = WENO(order = 5),
                                    tracers = :S,
                                    boundary_conditions = (; u = u_bcs, v = v_bcs, U = U_bcs, V = V_bcs, η = η_bcs, S = S_bcs))

set!(model, S = S_ocean)

simulation = Simulation(model; Δt = 60, stop_time)

u, v, w = model.velocities
S = model.tracers.S
η = model.free_surface.displacement
freshwater = Field(Integral((S_ocean - S) / S_ocean))
Q_river = V_river * mouth * river_depth

wall_clock = Ref(time_ns())
function progress(sim)
    compute!(freshwater)
    elapsed = (time_ns() - wall_clock[]) / 1e9
    @info @sprintf("iter %d, t = %s, max|u| = %.2f, max|v| = %.2f, min S = %.2f, freshwater / (Q t) = %.4f, mean η = %.3f m, wall %.1f s",
                   iteration(sim), prettytime(sim), maximum(abs, u), maximum(abs, v), minimum(S),
                   first(freshwater) / (Q_river * max(time(sim), 1e-9)), mean(η), elapsed)
    wall_clock[] = time_ns()
    return nothing
end
add_callback!(simulation, progress, TimeInterval(3hours))

filename = "immersed_river_plume"
simulation.output_writers[:surface] = JLD2Writer(model, (; u, v, S, η);
                                                 filename = filename * "_surface",
                                                 indices = (:, :, Nz), schedule = TimeInterval(1hour),
                                                 overwrite_existing = true)
simulation.output_writers[:section] = JLD2Writer(model, (; v, w, S);
                                                 filename = filename * "_section",
                                                 indices = (Nx ÷ 2, :, :), schedule = TimeInterval(1hour),
                                                 overwrite_existing = true)
run!(simulation)

# Plot
St = FieldTimeSeries(filename * "_surface.jld2", "S")
ηt = FieldTimeSeries(filename * "_surface.jld2", "η")
Ss = FieldTimeSeries(filename * "_section.jld2", "S")
times = St.times
Nt = length(times)

x, y, z = nodes(St)
xs, ys, zs = nodes(Ss)

fig = Figure(size = (1200, 900))
n = Observable(Nt)
title = @lift @sprintf("River plume through an immersed coastline, t = %s", prettytime(times[$n]))
Label(fig[0, 1:2], title; fontsize = 20)

axS = Axis(fig[1, 1]; xlabel = "x (km)", ylabel = "y (km)", title = "surface salinity", aspect = DataAspect())
Sn = @lift interior(St[$n], :, :, 1)
hm = heatmap!(axS, x / 1e3, y / 1e3, Sn; colormap = :haline, colorrange = (20, S_ocean))
Colorbar(fig[1, 2], hm)

axη = Axis(fig[2, 1]; xlabel = "x (km)", ylabel = "y (km)", title = "free surface (cm)", aspect = DataAspect())
ηn = @lift 100 .* interior(ηt[$n], :, :, 1)
hm = heatmap!(axη, x / 1e3, y / 1e3, ηn; colormap = :balance, colorrange = (-5, 5))
Colorbar(fig[2, 2], hm)

axs = Axis(fig[3, 1]; xlabel = "y (km)", ylabel = "z (m)", title = "salinity across the river mouth")
Ssn = @lift interior(Ss[$n], 1, :, :)
hm = heatmap!(axs, ys / 1e3, zs, Ssn; colormap = :haline, colorrange = (20, S_ocean))
Colorbar(fig[3, 2], hm)

save(filename * ".png", fig)
record(fig, filename * ".mp4", 1:Nt; framerate = 12) do i
    n[] = i
end
