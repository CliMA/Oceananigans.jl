# Distributed output writing validation
#
# Run with:
#
#   mpiexec -n 2 julia --project distributed_output_writing.jl
#

using MPI
using Oceananigans
using Oceananigans.DistributedComputations

topology = (Periodic, Periodic, Flat)
arch = Distributed(CPU())

Nranks = MPI.Comm_size(arch.communicator)
rank = MPI.Comm_rank(arch.communicator)

grid = RectilinearGrid(arch; topology, size = (16 ÷ Nranks, 16), halo = (3, 3), extent = (2π, 2π))

model = NonhydrostaticModel(; grid)

uᵢ = rand(size(grid)...)
vᵢ = rand(size(grid)...)
set!(model, u = uᵢ, v = vᵢ)

u, v, w = model.velocities
ζ = ∂x(v) - ∂y(u)

simulation = Simulation(model, Δt = 0.01, stop_iteration = 10)

simulation.output_writers[:fields] = JLD2Writer(model, merge(model.velocities, (; ζ)),
                                                schedule = IterationInterval(1),
                                                with_halos = true,
                                                filename = "test_output_writing_rank$rank",
                                                overwrite_existing = true)

run!(simulation)

@info "Output writing completed on rank $rank"
