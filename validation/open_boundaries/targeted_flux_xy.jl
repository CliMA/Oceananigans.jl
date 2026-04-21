# Validation script for `target_volume_flux` in `PerturbationAdvection`.
#
# Configuration: 2D (xy) flow with inflow at east and west, outflow at north and south.
# All four boundaries use PerturbationAdvection open boundary conditions.
# The east and north boundaries carry a prescribed target_volume_flux that forces exactly
# 60 % of the "full" natural flux through each, independent of the radiation scheme.
# The west and south boundaries have no target and participate in the global pool
# correction that restores volume conservation.
#
# With U_in = 1 m/s, Lx = Ly = 10 m (unit depth in z):
#
#   east  target  = -6 m²/s  (u < 0 → westward inflow, 60 % of natural 10)
#   north target  = +6 m²/s  (v > 0 → northward outflow, 60 % of natural 10)
#   west  (pool)  ≈ +10 m²/s (full natural inflow from west)
#   south (pool)  ≈ -10 m²/s (full natural outflow through south)
#
# The asymmetry (more inflow from west, more outflow through south) produces a
# SW-directed mean interior flow that is visible in u, v, and the vorticity field.

using Oceananigans, CairoMakie, Printf
using Oceananigans.BoundaryConditions: PerturbationAdvection

# ── Parameters ────────────────────────────────────────────────────────────────
arch = CPU()
Nx   = Ny  = 64
Lx   = Ly  = 10.0
U_in = 1.0          # natural inflow / outflow speed [m/s]
Re   = 200.0        # Reynolds number  (ν = U_in * Lx / Re)
ν    = U_in * Lx / Re

# target_volume_flux = ∫(normal velocity) dA  in the positive coordinate direction.
#   east face:  u < 0 (westward = inflow from east)  → negative target
#   north face: v > 0 (northward = outflow to north) → positive target
Q_east  = -0.6 * U_in * Ly   # [m²/s]
Q_north = +0.6 * U_in * Lx   # [m²/s]

inflow_timescale  = 0.3
outflow_timescale = Inf
prefix            = "targeted_flux_xy"

# ── Grid ──────────────────────────────────────────────────────────────────────
grid = RectilinearGrid(arch;
                       topology = (Bounded, Bounded, Flat),
                       size = (Nx, Ny),
                       x = (0, Lx),
                       y = (0, Ly),
                       halo = (4, 4))

@info grid

# ── Boundary conditions ───────────────────────────────────────────────────────
# Pool boundaries (west, south): participate in the global volume-flux correction.
# Targeted boundaries (east, north): independently corrected to Q_east / Q_north.
u_bcs = FieldBoundaryConditions(
    west = OpenBoundaryCondition(+U_in;
               scheme = PerturbationAdvection(; inflow_timescale, outflow_timescale)),
    east = OpenBoundaryCondition(-U_in;
               scheme = PerturbationAdvection(; inflow_timescale, outflow_timescale,
                                               target_volume_flux = Q_east)))

v_bcs = FieldBoundaryConditions(
    south = OpenBoundaryCondition(-U_in;
                scheme = PerturbationAdvection(; inflow_timescale, outflow_timescale)),
    north = OpenBoundaryCondition(+U_in;
                scheme = PerturbationAdvection(; inflow_timescale, outflow_timescale,
                                               target_volume_flux = Q_north)))

# ── Model ─────────────────────────────────────────────────────────────────────
model = NonhydrostaticModel(grid;
                            advection = UpwindBiased(order = 5),
                            closure = ScalarDiffusivity(; ν),
                            boundary_conditions = (; u = u_bcs, v = v_bcs))
@info model

# ── Initial conditions ────────────────────────────────────────────────────────
# Small noise: the open BCs drive the flow from rest.
set!(model, u = (x, y) -> 1e-2 * randn(), v = (x, y) -> 1e-2 * randn())

# ── Simulation ────────────────────────────────────────────────────────────────
Δx        = minimum_xspacing(grid)
Δt₀       = 0.2 * Δx / U_in
stop_time = 20.0

simulation = Simulation(model; Δt = Δt₀, stop_time)

wizard = TimeStepWizard(cfl = 0.3, max_Δt = Δt₀)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(20))

wall_time = Ref(time_ns())
function progress(sim)
    u, v, _ = sim.model.velocities
    elapsed  = 1e-9 * (time_ns() - wall_time[])
    @printf("t = %5.2f  Δt = %.4f  max|u| = %.3f  max|v| = %.3f  wall = %s\n",
            time(sim), sim.Δt, maximum(abs, u), maximum(abs, v), prettytime(elapsed))
    wall_time[] = time_ns()
end
simulation.callbacks[:progress] = Callback(progress, IterationInterval(200))

# ── Output writers ────────────────────────────────────────────────────────────
u, v, _ = model.velocities
ζ = @at (Center, Center, Center) ∂x(v) - ∂y(u)

simulation.output_writers[:fields] =
    JLD2Writer(model, (; u, v, ζ);
               schedule           = TimeInterval(0.5),
               filename           = prefix * "_fields.jld2",
               overwrite_existing = true,
               with_halos         = true)

@info "Running simulation …"
run!(simulation)
@info "Simulation finished in $(prettytime(simulation.run_wall_time))."

# ── Diagnostic: print final boundary volume fluxes ────────────────────────────
bvf = model.boundary_volume_fluxes
map(compute!, bvf)

west_Q  = Array(interior(bvf.west_volume_flux))[1, 1, 1]
east_Q  = Array(interior(bvf.east_volume_flux))[1, 1, 1]
south_Q = Array(interior(bvf.south_volume_flux))[1, 1, 1]
north_Q = Array(interior(bvf.north_volume_flux))[1, 1, 1]

@info @sprintf("""
Final boundary volume fluxes:
  west  (pool)     = %+.4f m²/s  (target: none)
  east  (targeted) = %+.4f m²/s  (target: %.4f)
  south (pool)     = %+.4f m²/s  (target: none)
  north (targeted) = %+.4f m²/s  (target: %.4f)
  net inflow = %+.2e m²/s (should be ≈ 0)
""",
west_Q, east_Q, Q_east, south_Q, north_Q, Q_north,
west_Q - east_Q + south_Q - north_Q)

# ── Animation ─────────────────────────────────────────────────────────────────
u_ts = FieldTimeSeries(prefix * "_fields.jld2", "u")
v_ts = FieldTimeSeries(prefix * "_fields.jld2", "v")
ζ_ts = FieldTimeSeries(prefix * "_fields.jld2", "ζ")

times = u_ts.times
Nt    = length(times)
@info "Loaded $Nt snapshots for animation."

# Coordinate arrays for interior data
xu, yu = xnodes(u_ts), ynodes(u_ts)
xv, yv = xnodes(v_ts), ynodes(v_ts)
xζ, yζ = xnodes(ζ_ts), ynodes(ζ_ts)

# Colour limits: maximum over all snapshots, saturated at 1.5 × U_in for u and v.
clim_u = 1.5 * U_in
clim_v = 1.5 * U_in
clim_ζ = max(1e-6, maximum(
    maximum(abs, interior(ζ_ts[n], :, :, 1)) for n in 1:Nt))

n     = Observable(1)
title = @lift @sprintf("Targeted flux xy — t = %.2f s", times[$n])

u_plt = @lift interior(u_ts[$n], :, :, 1)
v_plt = @lift interior(v_ts[$n], :, :, 1)
ζ_plt = @lift interior(ζ_ts[$n], :, :, 1)

fig = Figure(size = (1300, 480))
Label(fig[0, 1:6], title; fontsize = 16, font = :bold)

ax_u = Axis(fig[1, 1]; title = "u  (m/s)",  xlabel = "x (m)", ylabel = "y (m)", aspect = DataAspect())
ax_v = Axis(fig[1, 3]; title = "v  (m/s)",  xlabel = "x (m)", ylabel = "y (m)", aspect = DataAspect())
ax_ζ = Axis(fig[1, 5]; title = "ζ  (1/s)",  xlabel = "x (m)", ylabel = "y (m)", aspect = DataAspect())

hm_u = heatmap!(ax_u, xu, yu, u_plt; colorrange = (-clim_u, clim_u), colormap = :balance)
hm_v = heatmap!(ax_v, xv, yv, v_plt; colorrange = (-clim_v, clim_v), colormap = :balance)
hm_ζ = heatmap!(ax_ζ, xζ, yζ, ζ_plt; colorrange = (-clim_ζ, clim_ζ), colormap = :curl)

Colorbar(fig[1, 2], hm_u; label = "m/s")
Colorbar(fig[1, 4], hm_v; label = "m/s")
Colorbar(fig[1, 6], hm_ζ; label = "1/s")

resize_to_layout!(fig)

animation_file = prefix * ".mp4"
CairoMakie.record(fig, animation_file, 1:Nt; framerate = 16) do i
    i % 10 == 0 && @info "Frame $i / $Nt"
    n[] = i
end

@info "Animation saved to $animation_file."
