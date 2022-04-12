using Oceananigans
using Oceananigans.Fields: ConstantField
using GLMakie

grid = RectilinearGrid(size=(64, 64, 1), halo=(3, 3, 3), x=(-5, 5), y=(-5, 5), z=(0, 1), topology=(Periodic, Periodic, Bounded))
velocities = PrescribedVelocityFields(u=ConstantField(sqrt(1/2)), v=ConstantField(sqrt(1/2)))
tracer_advection = WENO5()
model = HydrostaticFreeSurfaceModel(; grid, tracer_advection, velocities, tracers=:c, buoyancy=nothing)
set!(model, c=(x, y, z) -> exp(-x^2 - y^2))
simulation = Simulation(model, Δt=0.1/grid.Nx, stop_iteration=10000)
progress(sim) = @info string("Iteration: ", iteration(sim), ", time: ", prettytime(sim))
simulation.callbacks[:p] = Callback(progress, IterationInterval(100))

simulation.output_writers[:c] = JLD2OutputWriter(model, model.tracers,
                                                 schedule = IterationInterval(10),
                                                 prefix = "diagonal_advection",
                                                 force = true)

run!(simulation)

ct = FieldTimeSeries("diagonal_advection.jld2", "c")
Nt = length(ct.times)

fig = Figure()
ax = Axis(fig[1, 1], aspect=1)
slider = Slider(fig[2, 1], range=1:Nt, startvalue=1)
n = slider.value
c = @lift interior(ct[$n], :, :, 1)
heatmap!(ax, c)

display(fig)
