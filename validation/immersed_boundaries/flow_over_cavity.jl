using Oceananigans
using Oceananigans.Utils: prettysummary
using GLMakie

underlying_grid = RectilinearGrid(size = (256, 128), halo = (3, 3),
                                  x = (-3, 3),
                                  z = (-0.1, 1),
                                  topology = (Periodic, Flat, Bounded))

cavity(x, y) = ifelse(abs(x) < 1, 0.0, 0.1)
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(cavity))
closure = ScalarDiffusivity(ν=1e-3)
advection = WENO5()
model = NonhydrostaticModel(; grid, closure, advection)

uᵢ(x, y, z) = z > 0.1 ? 1 : 0
set!(model, u=uᵢ)

simulation = Simulation(model, Δt=1e-2, stop_iteration=100)

wall_clock = Ref(time_ns())
function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    @info string("Iteration: ", iteration(sim), ", time: ", prettysummary(time(sim)), ", wall time: ", prettytime(elapsed))
    wall_clock[] = time_ns()
end

simulation.callbacks[:p] = Callback(progress, IterationInterval(10))

u, v, w = model.velocities
ζ = ∂x(v) - ∂y(u)

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; ζ),
                                                      filename = "flow_over_cavity",
                                                      schedule = IterationInterval(10),
                                                      overwrite_existing = true)

run!(simulation)

ζt = FieldTimeSeries("flow_over_cavity.jld2", "ζ")
Nt = length(ζt.times)

fig = Figure()
ax = Axis(fig[1, 1], aspect=1)
slider = Slider(fig[2, 1], range=1:Nt, startvalue=1)
n = slider.value

ζn = @lift interior(ζt[$n], :, :, 1)
heatmap!(ax, ζn)

display(fig)


