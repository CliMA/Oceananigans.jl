using Oceananigans
using Oceananigans.Models: ShallowWaterModel
using Oceananigans.Grids: Periodic, Bounded
using Oceananigans.Advection

using Plots
using Oceananigans.Grids: xnodes,ynodes

using Oceananigans.OutputWriters: IterationInterval, NetCDFOutputWriter

using Printf

L = 10
N = 64

grid = RegularCartesianGrid(size=(N, 1, 1), x=(-L/2,L/2), y=(0,1), z=(0,1), topology=(Periodic, Periodic, Bounded))

model = ShallowWaterModel(        grid = grid,
            gravitational_acceleration = 1,
                          architecture = CPU(),
                             advection = WENO5(), 
                              coriolis = FPlane(f=0.0)
                                  )

amp = 0.0001

width = 0.3
 h(x, y, z) = 1.0 + amp * exp(- x^2 / (2width^2));  
uh(x, y, z) = 0.0
vh(x, y, z) = 0.0 

set!(model, uh = uh, vh = vh, h = h)


simulation = Simulation(model, Δt = 0.01, stop_iteration = 300)

simulation.output_writers[:height] =
    NetCDFOutputWriter(model, model.solution, filepath = "1D_shallow_water.nc",
                       mode = "c", schedule=IterationInterval(1))

run!(simulation)

xn = xnodes(model.solution.h);

hexact = 1.0 .+ 0.5 .* amp .* exp.(-(xn .+ 3).^2/2width^2) .+ 0.5 .* amp .* exp.(-(xn .- 3).^2/2width^2)

plt1 = plot(xn, interior(model.solution.h)[:, 1, 1], linewidth=2,
            label="Numerical", title=@sprintf("t = %.3f", model.clock.time))

plt2 = plot!(xn, hexact, linewidth=2, label="Exact", reuse = false)

#display(plt2)
savefig("numerical_vs_exactd")
println("Saving plot of initial and final conditions.")

