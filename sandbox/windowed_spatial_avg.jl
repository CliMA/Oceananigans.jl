using Printf
using Revise
using Oceananigans
using Oceananigans.Fields
using Oceananigans.OutputWriters
using Oceananigans.Advection
using Oceananigans.Utils

#++++ Model set-up
Nx = Ny = Nz = 8
Lx = Ly = Lz = 1
N² = 1e-4 # s⁻²

grid = RegularRectilinearGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz), topology=(Periodic, Periodic, Periodic))

model = IncompressibleModel(
                   grid = grid,
              advection = UpwindBiasedFifthOrder(),
            timestepper = :RungeKutta3,
                closure = IsotropicDiffusivity(ν=1e-4, κ=1e-4),
               coriolis = FPlane(f=1e-4),
                tracers = (:b,), # P for Plankton
               buoyancy = BuoyancyTracer(),
)
println()
println(model)
println()
#-----

#++++ Initial conditions
u0(x, y, z) = y<Ly/2 ? 0.5 : 0.7
v0(x, y, z) = 0.2
w0(x, y, z) = 0.2*y

set!(model, u=u0, v=v0, w=w0)
#-----


#++++
#----


#++++
using Oceananigans.Grids
using Oceananigans.Diagnostics: WindowedSpatialAverage
u, v, w = model.velocities
slicer = FieldSlicer(j=Ny÷2+1:Ny)

Uw = WindowedSpatialAverage(u; dims=2, field_slicer=slicer)
U2w = WindowedSpatialAverage(ComputedField(u^2); dims=(1, 2), field_slicer=slicer)
#----



progress(sim) = @printf("Iteration: %d, time: %s, Δt: %s\n",
                        sim.model.clock.iteration,
                        prettytime(sim.model.clock.time),
                        prettytime(sim.Δt))

simulation = Simulation(model, Δt=1second, iteration_interval=5, progress=progress, stop_iteration=20,)

wout = (; Uw, U2w)
simulation.output_writers[:simple_output] = NetCDFOutputWriter(model, wout, 
                                                               schedule = AveragedTimeInterval(10seconds),
                                                               filepath = "windowed_avg.nc", mode = "c")

run!(simulation)
