using MPI
using Oceananigans
using Oceananigans.DistributedComputations

topology = (Periodic, Periodic, Flat)
arch = Distributed(CPU(); topology)

Nranks = MPI.Comm_size(comm)
grid = RectilinearGrid(arch; topology, size=(16 ÷ Nranks, 16), halo=(3, 3), extent=(2π, 2π))

model = NonhydrostaticModel(; grid)

uᵢ = rand(size(grid)...)
vᵢ = rand(size(grid)...)
set!(model, u=uᵢ, v=vᵢ)

u, v, w = model.velocities
ζ = ∂x(v) - ∂y(u)

simulation = Simulation(model, Δt=0.01, stop_iteration=3)

rank = MPI.Comm_rank(comm)
simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, (; ζ)),
                                                      schedule = IterationInterval(1),
                                                      with_halos = true,
                                                      filename = "test_output_writing_rank$rank",
                                                      overwrite_existing = true)

run!(simulation)

