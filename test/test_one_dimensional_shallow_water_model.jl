# # One dimensional shallow water example

using Oceananigans
using Oceananigans.Models: ShallowWaterModel
using Oceananigans.Grids: Periodic, Bounded

grid = RegularCartesianGrid(size=(64, 1, 1), extent=(2π, 2π, 2π))

model = ShallowWaterModel(        grid = grid,
                          architecture = CPU(),
                             advection = nothing, 
                              coriolis = nothing
                                  )

width = 0.3
 h(x, y, z)  = 1.0 + 0.1 * exp(-(x - π)^2 / (2width^2));  
uh(x, y, z) = 0.0
vh(x, y, z) = 0.0 

set!(model, uh = uh, vh = vh, h = h)

simulation = Simulation(model, Δt = 0.1, stop_iteration = 10)

using Plots
using Oceananigans.Grids: xnodes 

x = xnodes(model.solution.h);

h_plot = plot(x, interior(model.solution.h)[:, 1, 1],
              linewidth = 2,
              label = "t = 0",
              xlabel = "x",
              ylabel = "height")

run!(simulation)

using Printf

plot!(h_plot, x, interior(model.solution.h)[:, 1, 1], linewidth=2,
      label=@sprintf("t = %.3f", model.clock.time))

using Oceananigans.OutputWriters: JLD2OutputWriter, IterationInterval

simulation.output_writers[:height] =
    JLD2OutputWriter(model, model.solution, prefix = "one_dimensional_wave_equation",
                     schedule=IterationInterval(1), force = true)


simulation.stop_iteration += 10

run!(simulation)

using JLD2

file = jldopen(simulation.output_writers[:height].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

anim = @animate for (i, iter) in enumerate(iterations)

    h = file["timeseries/h/$iter"][:, 1, 1]
    t = file["timeseries/t/$iter"]

    plot(x, h, linewidth=2, title=@sprintf("t = %.3f", t),
         label="", xlabel="x", ylabel="height", xlims=(0, 2π))
end
