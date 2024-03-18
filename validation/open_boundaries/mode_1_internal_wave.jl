using Oceananigans
using Oceananigans.Fields: FunctionField

# TODO: clean up all the global variables in this script

h = 10
k = π/h

λ = 2π / k

grid = RectilinearGrid(topology = (Bounded, Flat, Bounded), size = (256, 128), extent = (3*λ, h))

xc, yc, zf = nodes(grid, Center(), Center(), Face())
xf, yc, zc = nodes(grid, Face(), Center(), Center())

N² = 0.2
w₀ = .03

ω = @show sqrt(k^2 * N² / (k^2 + (π/h)^2))

u(x, z, t) = w₀ * π / (k * h) * cos(π * z / h) * cos(k * x - ω * t)

w(x, z, t) = w₀ * sin(π * z / h) * sin(k * x - ω * t)

b(x, z, t) = - N² * w₀ / ω * sin(π * z / h) * cos(k * x - ω * t) + N² * z

@inline u_east(z, t) = u(xf[grid.Nx], z, t)
@inline u_west(z, t) = u(0, z, t)

@inline w_east(z, t) = w(xc[grid.Nx], z, t)
@inline w_west(z, t) = w(0, z, t)

@inline b_east(z, t) = b(xc[grid.Nx], z, t)
@inline b_west(z, t) = b(0, z, t)

u_boundaries = FieldBoundaryConditions(east = OpenBoundaryCondition(u_east),
                                       west = OpenBoundaryCondition(u_west))

w_boundaries = FieldBoundaryConditions(east = OpenBoundaryCondition(w_east), 
                                       west = OpenBoundaryCondition(w_west)) 

b_boundaries = FieldBoundaryConditions(east = OpenBoundaryCondition(b_east),
                                       west = OpenBoundaryCondition(b_west))

sponge_thickness = λ/2

mask(x, z) = ((sponge_thickness - x) / sponge_thickness) * (x < sponge_thickness) + ((x - grid.Lx + sponge_thickness) / sponge_thickness) * (x > grid.Lx - sponge_thickness)

cₚ = ω/k

cᵣ = 4 * (w₀ * π / (k * h) + cₚ)

u_forcing = Relaxation(; rate = cᵣ / sponge_thickness, mask, target = u)
w_forcing = Relaxation(; rate = cᵣ / sponge_thickness, mask, target = w)
b_forcing = Relaxation(; rate = cᵣ / sponge_thickness, mask, target = b)

model = NonhydrostaticModel(; grid, 
                              forcing = (; u = u_forcing, w = w_forcing, b = b_forcing),
                              boundary_conditions = (; u = u_boundaries, w = w_boundaries, b = b_boundaries),
                              advection = WENO(; order = 3),
                              buoyancy = BuoyancyTracer(),
                              tracers = :b)

set!(model, u = (x, z) -> u(x, z, 0), w = (x, z) -> w(x, z, 0), b = (x, z) -> b(x, z, 0))

simulation = Simulation(model, Δt = 0.1, stop_time = 400)

simulation.output_writers[:velocities] = JLD2OutputWriter(model, 
                                                          merge(model.velocities, model.tracers),
                                                          filename = "internal_wave.jld2", 
                                                          schedule = TimeInterval(0.5), 
                                                          overwrite_existing=true)

prog(sim) = @info prettytime(time(sim))*" with Δt = "*prettytime(sim.Δt)

simulation.callbacks[:progress] = Callback(prog, TimeInterval(5))

conjure_time_step_wizard!(simulation, cfl=0.5)

run!(simulation)

using CairoMakie

fig = Figure();

n = Observable(1)

u_ts = FieldTimeSeries("internal_wave.jld2", "u");
w_ts = FieldTimeSeries("internal_wave.jld2", "w");
b_ts = FieldTimeSeries("internal_wave.jld2", "b");

u_plt = @lift interior(u_ts[$n], :, 1, :)
w_plt = @lift interior(w_ts[$n], :, 1, :)
b_plt = @lift interior(b_ts[$n], :, 1, :) .- [N² * z for x in xc, z in zc]

title = @lift "t = $(u_ts.times[$n])"

ax = Axis(fig[1, 1], aspect = DataAspect(); title)
ax2 = Axis(fig[2, 1], aspect = DataAspect())
ax3 = Axis(fig[3, 1], aspect = DataAspect())

hm = heatmap!(ax, xf./h, zc./h, u_plt, colorrange = w₀ * π / (k * h).*(-1, 1), colormap = Reverse(:roma))
hm2 = heatmap!(ax2, xc./h, zf./h, w_plt, colorrange = w₀.*(-1, 1), colormap = Reverse(:roma))
hm3 = heatmap!(ax3, xc./h, zc./h, b_plt, colorrange = N² * w₀ / ω.*(-1, 1), colormap = Reverse(:roma))

record(fig, "internal_wave.mp4", 1:length(u_ts.times); framerate = 20) do i; 
    n[] = i
end