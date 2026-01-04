using Oceananigans
using Dates

arch = CPU()
grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))

start_time = DateTime(2020, 1, 1)
stop_time = start_time + Hour(6)
times = start_time:Hour(1):stop_time
c_forcing = FieldTimeSeries{Center, Center, Center}(grid, times)
set!(c_forcing[3], 1)
set!(c_forcing[4], -1)

clock = Clock(time=start_time)
model = HydrostaticFreeSurfaceModel(grid;
                                    clock,
                                    tracers = :c,
                                    forcing = (; c=c_forcing))
simulation = Simulation(model, Δt=60, stop_time=stop_time)

ow = JLD2Writer(model, model.tracers,
                filename = "datetime_clock_timestepping.jld2",
                overwrite_existing = true,
                schedule = IterationInterval(1))

simulation.output_writers[:jld2] = ow

run!(simulation)

ct = FieldTimeSeries("datetime_clock_timestepping.jld2", "c")
t = ct.times

using GLMakie
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Time", ylabel="c")
lines!(ax, t, interior(ct, 1, 1, 1, :))
display(fig)
