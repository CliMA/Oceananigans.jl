### Set up the model with the initial conditions

using Oceananigans
using Oceananigans.Models: ShallowWaterModel
using Oceananigans.Grids: Periodic, Bounded

Lx = 10
Ly = 10
grid = RegularCartesianGrid(size=(64, 64, 1), extent=(Lx, Ly, 1) , topology=(Periodic, Periodic, Bounded))

model = ShallowWaterModel(        grid = grid,
            gravitational_acceleration = 1,
                          architecture = CPU(),
                             advection = nothing, 
                              coriolis = FPlane(f=1.0)
                                  )

width = 0.3
 h(x, y, z)  = 1.0 + 0.1 * exp( - (x - Lx/2.)^2 / (2width^2) - (y - Ly/2.)^2 / (2width^2) );  
uh(x, y, z) = 0.0
vh(x, y, z) = 0.0 

set!(model, uh = uh, vh = vh, h = h)

simulation = Simulation(model, Δt = 0.01, stop_iteration = 500)

### Set up the plots and save initial height field

using Plots
using Oceananigans.Grids: xnodes, ynodes

x = xnodes(model.solution.h);
y = ynodes(model.solution.h);

h_plot = contourf(x, y, interior(model.solution.h)[:, :, 1],
                  c = :balance,
                  linewidth=0,
                  clim = (0.9, 1.1),
                  label = "height",
                  xlabel = "x",
                  ylabel = "y")

savefig("initial_height_2d")

@time time_step!(model, 1)

### Set up the OutputWriter using NetCDF

using Oceananigans.OutputWriters: JLD2OutputWriter, IterationInterval, NetCDFOutputWriter

simulation.output_writers[:height] =
    NetCDFOutputWriter(model, model.solution, filepath = "two_dimensional_wave_equation.nc",
                       mode = "c", schedule=IterationInterval(1))

### Find the solution

run!(simulation)

println("Done!")

### Plot the initial and final solution

using Printf
using NCDatasets

### Create an animation of the solution
anim = NCDataset(simulation.output_writers[:height].filepath) do ds
    @info "Saving animation of the solution."
    
    anim = @animate for (n, t) in enumerate(ds["time"])
        contourf(ds["xC"], ds["yC"], ds["h"][:, :, 1, n], title=@sprintf("t = %.3f", t),
                 c = :balance,
                 linewidth=0,
                 clim = (0.9, 1.1),
                 label="Height",
                 xlabel="x",
                 ylabel="y",
                 xlims=(0, 10),
                 ylims=(0, 10))
    end

    return anim
end

gif(anim, "two_dimensional_shallow_water_nc.gif", fps = 15) # hide

