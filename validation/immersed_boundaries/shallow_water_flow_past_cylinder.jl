using Oceananigans
using Oceananigans.OutputReaders: FieldTimeSeries 
using Oceananigans.Grids: min_Δx, min_Δy
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary

using GLMakie
using Printf

experiment_name = "shallow_water_flow_past_cylinder"

underlying_grid = RectilinearGrid(size=(128, 64), x=(-5, 10), y=(-3, 3),
                                  topology=(Periodic, Bounded, Flat),
                                  halo = (4, 4))

cylinder(x, y, z)  = (x^2 + y^2) < 1

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBoundary(cylinder))

model = ShallowWaterModel(; grid, gravitational_acceleration = 1)

set!(model, h = 1, uh = 1)

wall_clock = [time_ns()]

function progress(sim)
    @info(@sprintf("iteration: %d, time: %.2e, Δt: %.2e, wall time: %s, max|uh|: %.2f, max|vh|: %.2f, max|h|: %.2f",
                   sim.model.clock.iteration,
                   sim.model.clock.time,
                   sim.Δt,
                   prettytime(1e-9 * (time_ns() - wall_clock[1])),
                   maximum(abs, sim.model.solution.uh),
                   maximum(abs, sim.model.solution.vh),
                   maximum(abs, sim.model.solution.h)))

    wall_clock[1] = time_ns()

    return nothing
end

gravity_wave_speed = sqrt(model.gravitational_acceleration * 1)
Δmin = minimum((min_Δx(grid), min_Δy(grid)))
wave_propagation_time_scale = Δmin / gravity_wave_speed

wizard = TimeStepWizard(cfl=0.5, max_change=1.1, max_Δt=0.05Δmin)

simulation = Simulation(model, Δt=0.001Δmin, stop_time=2)

simulation.callbacks[:progress] = Callback(progress, IterationInterval(20))
simulation.callbacks[:wizard]   = Callback(wizard,   IterationInterval(5))

uh, vh, h = model.solution

ζ = Field(∂x(vh / h) - ∂y(uh / h))

outputs = merge(model.solution, (ζ=ζ,))

simulation.output_writers[:fields] =
    JLD2OutputWriter(model, outputs,
                     schedule = TimeInterval(0.1),
                     filename = experiment_name,
                     overwrite_existing = true)

run!(simulation)


# animate saved output

filepath = experiment_name * ".jld2"

ζ_timeseries = FieldTimeSeries(filepath, "ζ")
uh_timeseries = FieldTimeSeries(filepath, "uh")

times = ζ_timeseries.times
grid = ζ_timeseries.grid

xζ, yζ, zζ = nodes(ζ_timeseries)
xu, yu, zu = nodes(uh_timeseries)

n = Observable(1)

title = @lift @sprintf("t=%1.2f", times[$n])

ζₙ  = @lift interior(ζ_timeseries[$n], :, :, 1)
uhₙ = @lift interior(uh_timeseries[$n], :, :, 1)

axis_kwargs = (xlabel = L"x",
               ylabel = L"y",
               limits = (extrema(xnodes(grid, Face())), extrema(ynodes(grid, Face()))),
               aspect = grid.Lx/grid.Ly)

fig = Figure(resolution = (800, 800), fontsize=20)
axζ  = Axis(fig[2, 1]; title="relative vorticity", axis_kwargs...)
axuh = Axis(fig[3, 1]; title="zonal transport", axis_kwargs...)

fig[1, :] = Label(fig, title, fontsize=24, tellwidth=false)

hmζ  = GLMakie.heatmap!(axζ,  xζ, yζ, ζₙ, colormap=:balance)
Colorbar(fig[2, 2], hmζ)

hmuh = GLMakie.heatmap!(axuh, xu, yu, uhₙ, colormap=:balance)
Colorbar(fig[3, 2], hmuh)

frames = 1:length(times)

record(fig, experiment_name * ".mp4", frames, framerate=8) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    n[] = i
end
