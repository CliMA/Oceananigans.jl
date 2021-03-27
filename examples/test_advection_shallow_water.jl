using Oceananigans
using Oceananigans.Models
using Plots
using Revise

closure = IsotropicDiffusivity(κ=1.0)

 grid = RegularRectilinearGrid(size=64, x=(-π, π), topology=(Periodic, Flat, Flat))
model = ShallowWaterModel(grid=grid, gravitational_acceleration=1, tracers=(:c), closure=closure)

c(x, y, z) = exp(-x^2*10)

set!(model, uh=0, h=1, c=c)

using Oceananigans.OutputWriters: JLD2OutputWriter, IterationInterval

simulation = Simulation(model, Δt = 0.01, stop_iteration=100)

simulation.output_writers[:c] =
    JLD2OutputWriter(model, model.tracers, prefix = "tracer_diffusion",
                     schedule=IterationInterval(10), force = true)

run!(simulation)
                  
using JLD2, Printf

x = xnodes(model.tracers.c)

file = jldopen(simulation.output_writers[:c].filepath)
                     
iterations = parse.(Int, keys(file["timeseries/t"]))
                     
anim = @animate for (i, iter) in enumerate(iterations)
                     
    c = file["timeseries/c/$iter"][:, 1, 1]
    t = file["timeseries/t/$iter"]
    
    print("max c = ", maximum(c), "\n")
    plot(x, c, linewidth=2, title=@sprintf("t = %.3f", t),
            label="", xlabel="Tracer", ylabel="x", xlims=(-π, π))
end

mp4(anim, "tracer_diffusion.mp4", fps = 8) # hide