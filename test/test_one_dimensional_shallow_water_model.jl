# # One dimensional shallow water example

using Oceananigans
using Oceananigans.Models: ShallowWaterModel
using Oceananigans.Grids: Periodic, Bounded

grid = RegularCartesianGrid(size=(64, 1, 1), extent=(10, 1, 1), topology=(Periodic, Periodic, Bounded))

model = ShallowWaterModel(        grid = grid,
            gravitational_acceleration = 1,
                          architecture = CPU(),
                             advection = nothing,
                              coriolis = FPlane(f=1.0)
                                  )

width = 0.3
 h(x, y, z)  = 1.0 + 0.1 * exp(-(x - 5)^2 / (2width^2));
uh(x, y, z) = 0.0
vh(x, y, z) = 0.0

set!(model, uh = uh, vh = vh, h = h)

simulation = Simulation(model, Δt = 0.01, stop_iteration = 500)



using Plots
using Oceananigans.Grids: xnodes

x = xnodes(model.solution.h);

h_plot = plot(x, interior(model.solution.h)[:, 1, 1],
              linewidth = 2,
              label = "t = 0",
              xlabel = "x",
              ylabel = "height")


using Oceananigans.OutputWriters: NetCDFOutputWriter, IterationInterval

simulation.output_writers[:height] =
    NetCDFOutputWriter(model, model.solution, filepath = "one_dimensional_wave_equation.nc",
                       schedule=IterationInterval(1), mode="c")

run!(simulation)


using Printf

plt = plot!(h_plot, x, interior(model.solution.h)[:, 1, 1], linewidth=2,
            label=@sprintf("t = %.3f", model.clock.time))

savefig("slice")
println("Saving plot of initial and final conditions.")

using NCDatasets

NCDataset("one_dimensional_wave_equation.nc") do ds
    @info "Saving Hovmoller plots of the solution."

    contourf(ds["time"], ds["xC"], ds["h"][:, 1, 1, :], linewidth=0)
    savefig("Hovmolleer_h.png")

    contourf(ds["time"], ds["xF"], ds["uh"][:, 1, 1, :], linewidth=0)
    savefig("Hovmolleer_uh.png")

    contourf(ds["time"], ds["xC"], ds["vh"][:, 1, 1, :], linewidth=0)
    savefig("Hovmolleer_vh.png")
end
