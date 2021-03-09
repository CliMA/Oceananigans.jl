using Statistics
using Oceananigans

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
grid = RegularRectilinearGrid(topology=topo, size=(128, 128, 1), extent=(4π, 4π, 1), halo=(3, 3, 3))
arch = CPU()

model = ShallowWaterModel(
    architecture = arch,
            grid = grid,
     timestepper = :QuasiAdamsBashforth2,
#     timestepper = :RungeKutta3,
    advection = WENO5(),
#         closure = IsotropicDiffusivity(ν=1e-5),
gravitational_acceleration = 1.0
)

uh₀ = rand(size(model.grid)...);
uh₀ .-= mean(uh₀);
set!(model, uh=uh₀, vh=uh₀, h=model.grid.Lz)

progress(sim) = @info "Iteration: $(sim.model.clock.iteration), time: $(sim.model.clock.time)"
simulation = Simulation(model, Δt=0.001, stop_time=100.0, iteration_interval=1, progress=progress)

uh, vh, h = model.solution
outputs = (ζ=ComputedField(∂x(vh/h) - ∂y(uh/h)), h)
filepath = "serial_shallow_water_turbulence.nc"
simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, outputs, filepath=filepath, schedule=TimeInterval(1.0), mode="c")

run!(simulation)

using Printf
using NCDatasets
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

ds = NCDataset("serial_shallow_water_turbulence.nc", "r")

iterations = keys(ds["time"])

anim = @animate for (iter, t) in enumerate(ds["time"])
    ζ = ds["ζ"][:,:,1,iter]

    plot_ζ = contour(xc, yc, ζ',  title=@sprintf("Total ζ at t = %.3f", t); kwargs...)

    display(plot_ζ)
    
    print("At t = ", t, " maximum of ζ = ", maximum(abs, ζ), " and minimum of h = ", minimum(h), "\n")
end


close(ds)

mp4(anim, "serial_shallow_water_turbulence.mp4", fps=15)


