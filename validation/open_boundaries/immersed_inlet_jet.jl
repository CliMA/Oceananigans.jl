# A tidal inlet jet: water enters a stratified, rotating basin through a gap in an immersed
# coastline (a `NormalFlow` immersed boundary condition on `u`) and leaves through a radiating
# open boundary on the east side of the domain.

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: xnode, ynode, znode
using CUDA
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using CairoMakie
using Printf
using Statistics: mean

arch = GPU()
const Lx, Ly, Lz = 2kilometers, 1kilometer, 20
Nx, Ny, Nz = 256, 128, 32

const coast = 200           # the coastline sits at x = 200 m
const inlet_width = 100     # [m]
const inlet_depth = 10      # [m]
const U = 0.5               # inlet velocity [m s⁻¹]
const N² = 1e-4             # ambient stratification [s⁻²]
const f = 1e-4              # Coriolis parameter [s⁻¹]
stop_time = 2hours

grid = RectilinearGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 5), x = (0, Lx), y = (0, Ly), z = (-Lz, 0),
                       topology = (Bounded, Bounded, Bounded))
grid = ImmersedBoundaryGrid(grid, GridFittedBoundary((x, y, z) -> x < coast))

@inline in_inlet(y, z) = abs(y - Ly / 2) < inlet_width / 2 && z > -inlet_depth
@inline inlet_velocity(i, j, k, grid, clock, fields) = ifelse(in_inlet(ynode(j, grid, Center()), znode(k, grid, Center())), U, zero(U))

radiation = PerturbationAdvection(inflow_timescale = 10minutes, outflow_timescale = 1hour)
u_bcs = FieldBoundaryConditions(immersed = NormalFlowBoundaryCondition(inlet_velocity, discrete_form = true),
                                east = NormalFlowBoundaryCondition(0; scheme = radiation))
c_bcs = FieldBoundaryConditions(immersed = ValueBoundaryCondition(1))

model = NonhydrostaticModel(grid; advection = WENO(order = 7), coriolis = FPlane(; f),
                            buoyancy = BuoyancyTracer(), tracers = (:b, :c),
                            boundary_conditions = (; u = u_bcs, c = c_bcs),
                            pressure_solver = ConjugateGradientPoissonSolver(grid; reltol = 1e-7, maxiter = 200))

set!(model, b = (x, y, z) -> N² * z)

simulation = Simulation(model; Δt = 2, stop_time)
conjure_time_step_wizard!(simulation, cfl = 0.6, max_Δt = 10)

u, v, w = model.velocities
c = model.tracers.c
ζ = ∂x(v) - ∂y(u)
∫c = Field(Integral(c))
∫u_east = Field(Integral(view(u, Nx + 1, :, :), dims = (2, 3)))
Q_inlet = U * inlet_width * inlet_depth

wall_clock = Ref(time_ns())
function progress(sim)
    compute!(∫c)
    compute!(∫u_east)
    elapsed = (time_ns() - wall_clock[]) / 1e9
    @info @sprintf("iter %d, t = %s, Δt = %s, max|u| = %.2f, ∫c dV / (Q t) = %.4f, outflow / Q = %.4f, wall %.1f s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), maximum(abs, u),
                   first(∫c) / (Q_inlet * max(time(sim), 1e-9)), first(∫u_east) / Q_inlet, elapsed)
    wall_clock[] = time_ns()
    return nothing
end
add_callback!(simulation, progress, TimeInterval(5minutes))

filename = "immersed_inlet_jet"
simulation.output_writers[:surface] = JLD2Writer(model, (; u, v, c, ζ, b = model.tracers.b);
                                                 filename = filename * "_surface",
                                                 indices = (:, :, Nz), schedule = TimeInterval(2minutes),
                                                 overwrite_existing = true)
simulation.output_writers[:section] = JLD2Writer(model, (; u, w, c, b = model.tracers.b);
                                                 filename = filename * "_section",
                                                 indices = (:, Ny ÷ 2, :), schedule = TimeInterval(2minutes),
                                                 overwrite_existing = true)
run!(simulation)

# Plot
ct = FieldTimeSeries(filename * "_surface.jld2", "c")
ζt = FieldTimeSeries(filename * "_surface.jld2", "ζ")
cs = FieldTimeSeries(filename * "_section.jld2", "c")
ws = FieldTimeSeries(filename * "_section.jld2", "w")
times = ct.times
Nt = length(times)

x, y, z = nodes(ct)
xs, ys, zs = nodes(cs)
xζ, yζ, _ = nodes(ζt)

fig = Figure(size = (1200, 1000))
n = Observable(Nt)
title = @lift @sprintf("Inlet jet through an immersed coastline, t = %s", prettytime(times[$n]))
Label(fig[0, 1:2], title; fontsize = 20)

axc = Axis(fig[1, 1]; xlabel = "x (m)", ylabel = "y (m)", title = "surface dye", aspect = DataAspect())
cn = @lift interior(ct[$n], :, :, 1)
hm = heatmap!(axc, x, y, cn; colormap = :tempo, colorrange = (0, 1))
Colorbar(fig[1, 2], hm)

axζ = Axis(fig[2, 1]; xlabel = "x (m)", ylabel = "y (m)", title = "surface vorticity / f", aspect = DataAspect())
ζn = @lift interior(ζt[$n], :, :, 1) ./ f
hm = heatmap!(axζ, xζ, yζ, ζn; colormap = :balance, colorrange = (-20, 20))
Colorbar(fig[2, 2], hm)

axs = Axis(fig[3, 1]; xlabel = "x (m)", ylabel = "z (m)", title = "dye along the inlet axis")
csn = @lift interior(cs[$n], :, 1, :)
hm = heatmap!(axs, xs, zs, csn; colormap = :tempo, colorrange = (0, 1))
Colorbar(fig[3, 2], hm)

save(filename * ".png", fig)
record(fig, filename * ".mp4", 1:Nt; framerate = 12) do i
    n[] = i
end
