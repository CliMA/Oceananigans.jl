using MPI
using Statistics: mean

MPI.Init()
using Printf
using Oceananigans
using Oceananigans.Utils: prettytime

rank = MPI.Comm_rank(MPI.COMM_WORLD)
size = MPI.Comm_size(MPI.COMM_WORLD)

#=
local_comm = MPI.Comm_split_type(MPI.COMM_WORLD, MPI.COMM_TYPE_SHARED, rank)
node_rank  = MPI.Comm_rank(local_comm)

buff = (send = Array(zeros(5, 5, 5)), recv = Array(zeros(5, 5, 5)))
buff.send .= rank + 10

rank_recv_from = rank - 1 < 0 ? size - 1 : rank - 1
rank_send_to   = rank + 1 > size - 1 ? 0 : rank + 1

recvreq = MPI.Irecv!(buff.recv, rank_recv_from, 0, MPI.COMM_WORLD)
sendreq =  MPI.Isend(buff.send, rank_send_to,   0, MPI.COMM_WORLD)

MPI.Waitall([sendreq, recvreq])

@info rank mean(buff.send) mean(buff.recv) rank_recv_from rank_send_to
MPI.Barrier(MPI.COMM_WORLD)
=#

using Oceananigans.Distributed

N = 128 ÷ size

topo  = (Periodic, Periodic, Periodic)
arch  = DistributedArch(GPU(); ranks = (size, 1, 1), topology = topo)
grid  = RectilinearGrid(arch, size = (N, N, N), extent = (1, 1, 1), topology = topo, halo = (7, 7, 7))
# model = HydrostaticFreeSurfaceModel(; grid, momentum_advection = WENO(order = 9), tracer_advection = WENO(order = 9),
#                                       tracers = :b, buoyancy = nothing, coriolis = nothing, 
#                                       free_surface = SplitExplicitFreeSurface(; substeps = 30))
model = NonhydrostaticModel(; grid, advection = WENO(order = 9), tracers = :b, timestepper = :RungeKutta3)

simulation = Simulation(model, Δt = 1e-4, stop_iteration = 40)

wtime = Ref(time_ns())

function progress(sim) 
   @info @sprintf("iteration: %d, wall time: %s \n", sim.model.clock.iteration, prettytime((time_ns() - wtime[])*1e-9))
   wtime[] = time_ns()
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

run!(simulation)

MPI.Finalize()
