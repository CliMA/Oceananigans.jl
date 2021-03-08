using Statistics
using Oceananigans
#using Oceananigans.Distributed

using Oceananigans.Architectures: CPU, architecture
using Oceananigans.Models: ShallowWaterModel
using Oceananigans.Grids: Periodic, Bounded, RegularRectilinearGrid
using Oceananigans.Grids: xnodes, ynodes, interior
using Oceananigans.Simulations: Simulation, set!, run!, TimeStepWizard
using Oceananigans.Coriolis: FPlane
using Oceananigans.Advection: WENO5
using Oceananigans.OutputWriters: NetCDFOutputWriter, IterationInterval, TimeInterval
using Oceananigans.Fields, Oceananigans.AbstractOperations
using Oceananigans.TurbulenceClosures: AnisotropicBiharmonicDiffusivity

topo = (Periodic, Periodic, Bounded)
grid = RegularRectilinearGrid(topology=topo, size=(8, 1, 1), extent=(4π, 4π, 1), halo=(3, 3, 3))
arch = CPU()

model = ShallowWaterModel(
    architecture = arch,
            grid = grid,
     timestepper = :QuasiAdamsBashforth2,
#     timestepper = :RungeKutta3,
    advection = WENO5(),
#    advection = nothing,
#         closure = IsotropicDiffusivity(ν=1e-5),
gravitational_acceleration = 1.0
)

using Random

Random.seed!(123)
uh₀ = rand(size(model.grid)...);
uh₀ .-= mean(uh₀);
set!(model, uh=uh₀, vh=uh₀, h=model.grid.Lz)

progress(sim) = @info "Iteration: $(sim.model.clock.iteration), time: $(sim.model.clock.time)"
simulation = Simulation(model, Δt=0.01, stop_time=0.02, iteration_interval=1, progress=progress)

uh, vh, h = model.solution
outputs = (ζ=ComputedField(∂x(vh/h) - ∂y(uh/h)), uh, vh, h)
filepath = "serial_shallow_water_turbulence_x.nc"
simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, outputs, filepath=filepath, schedule=TimeInterval(0.01), mode="c")

run!(simulation)

using Printf
using NCDatasets
#using CairoMakie
using Plots

xc = xnodes(model.solution.h)
yc = ynodes(model.solution.h)

kwargs = (
         xlabel = "x",
         ylabel = "y",
           fill = true,
         levels = 20,
      linewidth = 0,
          color = :balance,
       colorbar = true
)

ds = NCDataset("serial_shallow_water_turbulence_x.nc", "r")

iterations = keys(ds["time"])

print("uh = ", uh[:,1,1,1], "\n")
print("vh = ", vh[:,1,1,1], "\n")
print(" h = ",  h[:,1,1,1], "\n")
          
print("uh = ", uh[1:8,1,1,1], "\n")
print("vh = ", vh[1:8,1,1,1], "\n")
print(" h = ",  h[1:8,1,1,1], "\n")
          
anim = @animate for (iter, t) in enumerate(ds["time"])
    ζ = ds["ζ"][:,1,1,iter]

    plot_ζ = plot(xc, ζ,  title=@sprintf("Total ζ at t = %.3f", t))
    
    print("At t = ", t, " maximum of ζ = ", maximum(ζ), " and minimum of ζ = ", minimum(ζ), "\n")
    print("               maximums: uh = ", maximum(uh), ",  vh = ", maximum(vh), ",  h = ", maximum(h), "\n")
    print("               minimums: uh = ", minimum(uh), ", vh = ", minimum(vh), ", h = ", minimum(h), "\n")
end

#=
anim = @animate for (iter, t) in enumerate(ds["time"])
    ζ = ds["ζ"][:,1,1,iter]

    plot_ζ = contour(xc, yc, ζ',  title=@sprintf("Total ζ at t = %.3f", t); kwargs...)

    display(plot_ζ)
    
    print("At t = ", t, " maximum of ζ = ", maximum(abs, ζ), " and minimum of h = ", minimum(h), "\n")
end
=#

close(ds)

mp4(anim, "serial_shallow_water_turbulence_x.mp4", fps=15)


