# A buoyant plume from a hydrothermal vent: warm water is injected through a patch of the immersed
# sea floor (a `NormalFlow` immersed boundary condition on `w` with `Value` conditions on buoyancy
# and a dye), rises through a stratified water column, and the volume leaves through radiating
# open boundaries on the four sides of the domain.

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: xnode, ynode, znode
using CUDA
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using CairoMakie
using Printf

arch = GPU()
const L = 200                   # domain width and height [m]
const N = 128
const seafloor = -L + 10        # the sea floor sits 10 m above the bottom of the domain
const vent_radius = 8           # [m]
const W = 0.2                   # vent velocity [m s⁻¹]
const b_vent = 0.03             # vent buoyancy anomaly [m s⁻²]
const N² = 1e-4                 # ambient stratification [s⁻²]
stop_time = 30minutes

grid = RectilinearGrid(arch; size = (N, N, N), halo = (5, 5, 5), x = (-L/2, L/2), y = (-L/2, L/2), z = (-L, 0),
                       topology = (Bounded, Bounded, Bounded))
grid = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> seafloor))

@inline in_vent(x, y) = x^2 + y^2 < vent_radius^2
@inline vent_velocity(i, j, k, grid, clock, fields) = ifelse(in_vent(xnode(i, grid, Center()), ynode(j, grid, Center())), W, zero(W))
@inline vent_buoyancy(i, j, k, grid, clock, fields) = N² * znode(k, grid, Center()) + ifelse(in_vent(xnode(i, grid, Center()), ynode(j, grid, Center())), b_vent, zero(b_vent))

radiation = PerturbationAdvection(inflow_timescale = 1minute, outflow_timescale = 10minutes)
open = NormalFlowBoundaryCondition(0; scheme = radiation)

w_bcs = FieldBoundaryConditions(immersed = ImmersedBoundaryCondition(bottom = NormalFlowBoundaryCondition(vent_velocity, discrete_form = true)))
u_bcs = FieldBoundaryConditions(west = open, east = open)
v_bcs = FieldBoundaryConditions(south = open, north = open)
b_bcs = FieldBoundaryConditions(immersed = ImmersedBoundaryCondition(bottom = ValueBoundaryCondition(vent_buoyancy, discrete_form = true)))
c_bcs = FieldBoundaryConditions(immersed = ImmersedBoundaryCondition(bottom = ValueBoundaryCondition(1)))

model = NonhydrostaticModel(grid; advection = WENO(order = 7), buoyancy = BuoyancyTracer(), tracers = (:b, :c),
                            boundary_conditions = (; u = u_bcs, v = v_bcs, w = w_bcs, b = b_bcs, c = c_bcs),
                            pressure_solver = ConjugateGradientPoissonSolver(grid; reltol = 1e-7, maxiter = 200))

set!(model, b = (x, y, z) -> N² * z)

simulation = Simulation(model; Δt = 1, stop_time)
conjure_time_step_wizard!(simulation, cfl = 0.6, max_Δt = 5)

u, v, w = model.velocities
b, c = model.tracers
∫c = Field(Integral(c))
Δx = L / N
Q_vent = W * Δx^2 * count(in_vent(x, y) for x in xnodes(grid, Center()), y in ynodes(grid, Center()))

wall_clock = Ref(time_ns())
function progress(sim)
    compute!(∫c)
    elapsed = (time_ns() - wall_clock[]) / 1e9
    @info @sprintf("iter %d, t = %s, Δt = %s, max w = %.2f, ∫c dV / (Q t) = %.4f, wall %.1f s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), maximum(w),
                   Array(interior(∫c))[1] / (Q_vent * max(time(sim), 1e-9)), elapsed)
    wall_clock[] = time_ns()
    return nothing
end
add_callback!(simulation, progress, TimeInterval(2minutes))

filename = "immersed_vent_plume"
simulation.output_writers[:section] = JLD2Writer(model, (; w, b, c);
                                                 filename = filename * "_section",
                                                 indices = (:, N ÷ 2, :), schedule = TimeInterval(30),
                                                 overwrite_files = true)
run!(simulation)

# Plot
ct = FieldTimeSeries(filename * "_section.jld2", "c")
wt = FieldTimeSeries(filename * "_section.jld2", "w")
bt = FieldTimeSeries(filename * "_section.jld2", "b")
times = ct.times
Nt = length(times)
x, y, z = nodes(ct)
xw, yw, zw = nodes(wt)

fig = Figure(size = (1400, 500))
n = Observable(Nt)
title = @lift @sprintf("Vent plume through an immersed sea floor, t = %s", prettytime(times[$n]))
Label(fig[0, 1:6], title; fontsize = 20)

axc = Axis(fig[1, 1]; xlabel = "x (m)", ylabel = "z (m)", title = "dye", aspect = DataAspect())
cn = @lift interior(ct[$n], :, 1, :)
hm = heatmap!(axc, x, z, cn; colormap = :tempo, colorrange = (0, 0.5))
Colorbar(fig[1, 2], hm)

axw = Axis(fig[1, 3]; xlabel = "x (m)", ylabel = "z (m)", title = "w (m s⁻¹)", aspect = DataAspect())
wn = @lift interior(wt[$n], :, 1, :)
hm = heatmap!(axw, xw, zw, wn; colormap = :balance, colorrange = (-0.3, 0.3))
Colorbar(fig[1, 4], hm)

axb = Axis(fig[1, 5]; xlabel = "x (m)", ylabel = "z (m)", title = "buoyancy anomaly (m s⁻²)", aspect = DataAspect())
bn = @lift interior(bt[$n], :, 1, :) .- N² .* reshape(z, 1, :)
hm = heatmap!(axb, x, z, bn; colormap = :thermal, colorrange = (0, 0.01))
Colorbar(fig[1, 6], hm)

save(filename * ".png", fig)
CairoMakie.record(fig, filename * ".mp4", 1:Nt; framerate = 12) do i
    n[] = i
end
