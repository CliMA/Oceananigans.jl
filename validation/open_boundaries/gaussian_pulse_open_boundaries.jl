"""
One-dimensional Gaussian pulse entering through a Flather open boundary and
exiting through the opposite one.  Validates that the Flather characteristic
condition correctly transmits an incoming shallow-water signal and that the
outgoing boundary absorbs it without significant reflection.

The analytical solution for a rightward-propagating linear shallow-water pulse is

    η(x, t) = A exp(−(x − x₀ − c t)² / 2σ²)
    U(x, t) = c η(x, t)

where c = √(gH) is the barotropic phase speed and U = Hu is the depth-integrated
transport.  At both boundaries we prescribe (U_ext, η_ext) from the analytical
solution via Flather conditions.
"""

using Oceananigans
using Oceananigans.Grids: xnode
using Oceananigans.BoundaryConditions: FlatherBoundaryCondition, RadiationBoundaryCondition
using CairoMakie
using Printf

g  = Oceananigans.defaults.gravitational_acceleration
H  = 100.0          # depth [m]
c  = sqrt(g * H)    # barotropic phase speed ≈ 31.3 m/s
A  = 0.01           # pulse amplitude [m]  (small → linear regime)
σ  = 500.0          # pulse width [m]
Lx = 10_000.0       # domain length [m]
x₀ = -3σ            # initial pulse center (3σ to the left of the domain)

## Analytical solution

@inline η_analytical(x, t) = A * exp(-(x - x₀ - c * t)^2 / (2 * σ^2))
@inline U_analytical(x, t) = c * η_analytical(x, t)

## Grid

Nx = 200
grid = RectilinearGrid(size = (Nx, 1, 1),
                        x = (0, Lx),
                        y = (0, 100.0),
                        z = (-H, 0),
                        topology = (Bounded, Periodic, Bounded))

## Boundary conditions
#
# West (inflow): prescribe the analytical (U, η) through a discrete Flather condition.
# East (outflow): prescribe the analytical solution as well for a clean comparison.

@inline function west_flather_bc(i, j, grid, clock, model_fields)
    t = clock.time
    return (U_analytical(0, t), η_analytical(0, t))
end

@inline function east_flather_bc(i, j, grid, clock, model_fields)
    t = clock.time
    U = U_analytical(xnode(grid.Nx+1, grid, Face()), t)
    η = η_analytical(xnode(grid.Nx+1, grid, Face()), t)
    return (U, η)
end

U_bcs = FieldBoundaryConditions(grid, (Face(), Center(), nothing);
                                west = FlatherBoundaryCondition(west_flather_bc; discrete_form = true),
                                east = FlatherBoundaryCondition(east_flather_bc; discrete_form = true))

u_bcs = FieldBoundaryConditions(west = RadiationBoundaryCondition(0),
                                east = RadiationBoundaryCondition(0))

## Model

free_surface = SplitExplicitFreeSurface(grid; substeps = 30, extend_halos = false)

model = HydrostaticFreeSurfaceModel(grid;
                                    free_surface        = free_surface,
                                    boundary_conditions = (u = u_bcs, U = U_bcs),
                                    buoyancy            = nothing,
                                    tracers             = ())

## Analytical fields via KernelFunctionOperation
#
# These are evaluated on the fly at each saved time step so that
# the JLD2 file contains both numerical and analytical solutions.

@inline η_analytical_kernel(i, j, k, grid, clock) = η_analytical(xnode(i, grid, Center()), clock.time)
@inline U_analytical_kernel(i, j, k, grid, clock) = U_analytical(xnode(i, grid, Face()),   clock.time)

η_exact = KernelFunctionOperation{Center, Center, Face}(η_analytical_kernel, grid, model.clock)
U_exact = KernelFunctionOperation{Face, Center, Nothing}(U_analytical_kernel, grid, model.clock)

## Simulation with JLD2 output

Δx = Lx / Nx
Δt = 0.4 * Δx / c
T_total = (Lx - x₀ + 4σ) / c

simulation = Simulation(model; Δt, stop_time = 2T_total)

function progress(sim)
    η = sim.model.free_surface.displacement
    @printf("  iter %05d | t = %6.1f s | max|η| = %.2e\n",
            iteration(sim), time(sim), maximum(abs, η))
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(200))

outdir = "output"
mkpath(outdir)
outfile = joinpath(outdir, "gaussian_pulse.jld2")

η = model.free_surface.displacement
U = model.free_surface.barotropic_velocities.U

simulation.output_writers[:fields] = JLD2Writer(model,
    (; η, U, η_exact, U_exact);
    schedule           = IterationInterval(10),
    filename           = outfile,
    overwrite_existing = true)

run!(simulation)

@info "Simulation complete. Output saved to $outfile"

## Load results with FieldTimeSeries

η_ts = FieldTimeSeries(outfile, "η")
U_ts = FieldTimeSeries(outfile, "U")
η_exact_ts = FieldTimeSeries(outfile, "η_exact")
U_exact_ts = FieldTimeSeries(outfile, "U_exact")

times = η_ts.times
Nt    = length(times)
xc    = xnodes(grid, Center())
xf    = xnodes(grid, Face())

@info "Loaded $Nt snapshots from $outfile"

## Compute L₂ error when the pulse center is near the domain midpoint

t_mid   = (Lx / 2 - x₀) / c
mid_idx = argmin(abs.(times .- t_mid))
η_num   = interior(η_ts[mid_idx], :, 1, 1)
η_ana   = interior(η_exact_ts[mid_idx], :, 1, 1)
L₂ = sqrt(sum((η_num .- η_ana).^2) / Nx) / A

@info @sprintf("L₂ / A = %.4f at t = %.1f s (pulse near midpoint)", L₂, times[mid_idx])

## Animate: two-panel video (η on top, U on bottom)

fig = Figure(size = (900, 650))

ax_η = Axis(fig[1, 1]; ylabel = "η [m]",
            title = "Gaussian pulse through Flather open boundaries")
ax_U = Axis(fig[2, 1]; xlabel = "x [m]", ylabel = "U [m²/s]")

ylims!(ax_η, -0.3A, 1.3A)
ylims!(ax_U, -0.3A * c, 1.3A * c)

η_obs  = Observable(interior(η_ts[1], :, 1, 1))
ηa_obs = Observable(interior(η_exact_ts[1], :, 1, 1))
U_obs  = Observable(interior(U_ts[1], :, 1, 1))
Ua_obs = Observable(interior(U_exact_ts[1], :, 1, 1))
tlabel = Observable(@sprintf("t = %.1f s", times[1]))

lines!(ax_η, xc, ηa_obs; color = :grey60, linestyle = :dash, linewidth = 2, label = "analytical")
lines!(ax_η, xc, η_obs;  color = :dodgerblue, linewidth = 2, label = "numerical")
axislegend(ax_η; position = :rt)
text!(ax_η, 0.02, 0.92; text = tlabel, space = :relative, fontsize = 14, align = (:left, :top))

lines!(ax_U, xf, Ua_obs; color = :grey60, linestyle = :dash, linewidth = 2, label = "analytical")
lines!(ax_U, xf, U_obs;  color = :firebrick, linewidth = 2, label = "numerical")
axislegend(ax_U; position = :rt)

frames_per_second = 30

record(fig, "gaussian_pulse_open_boundaries.mp4", 1:Nt; framerate = frames_per_second) do idx
    η_num_i = interior(η_ts[idx], :, 1, 1)
    η_ana_i = interior(η_exact_ts[idx], :, 1, 1)

    η_obs[]  = η_num_i
    ηa_obs[] = η_ana_i
    U_obs[]  = interior(U_ts[idx], :, 1, 1)
    Ua_obs[] = interior(U_exact_ts[idx], :, 1, 1)

    L₂_i = sqrt(sum((η_num_i .- η_ana_i).^2) / Nx) / A
    tlabel[] = @sprintf("t = %.1f s  (L₂/A = %.4f)", times[idx], L₂_i)
end

@info "Saved gaussian_pulse_open_boundaries.mp4"
