using MPI
using Statistics: mean

MPI.Init()
using Printf
using Oceananigans
using Oceananigans.Utils: prettytime

rank = MPI.Comm_rank(MPI.COMM_WORLD)
size = MPI.Comm_size(MPI.COMM_WORLD)

using Oceananigans.Distributed

N     = (128, 128, 128)
ranks = (2, 2, 1)
topo  = (Periodic, Periodic, Periodic)
arch  = DistributedArch(CPU(); ranks, topology = topo)
grid  = RectilinearGrid(arch, size = N ./ ranks, extent = (2π, 2π, 2π), topology = topo, halo = (7, 7, 7))
model = NonhydrostaticModel(; grid, advection = WENO(order = 7), tracers = :b, timestepper = :RungeKutta3)

set!(model, u = (x, y, z) -> rand(), v = (x, y, z) -> rand())

simulation = Simulation(model, Δt = 1e-2, stop_iteration = 10000)

wtime = Ref(time_ns())

u, v, w = model.veloc

function progress(sim) 
   @info @sprintf("iteration: %d, wall time: %s \n", sim.model.clock.iteration, prettytime((time_ns() - wtime[])*1e-9))
   wtime[] = time_ns()
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, (; ζ)),
                                                      filename = "test_$(rank)",
                                                      schedule = TimeInterval(0.1))

run!(simulation)

MPI.Finalize()
