using MPI
using Oceananigans
using Oceananigans.Distributed

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
Nranks = MPI.Comm_size(comm)

topology = (Periodic, Periodic, Flat)
arch = MultiArch(CPU(); topology, ranks=(1, Nranks, 1))
grid = RectilinearGrid(arch; topology, size=(16, 16), halo=(3, 3), extent=(2π, 2π))

model = NonhydrostaticModel(; grid)

uᵢ = rand(size(grid)...)
vᵢ = rand(size(grid)...)
set!(model, u=uᵢ, v=vᵢ)

u, v, w = model.velocities
ζ = ∂x(v) - ∂y(u)

simulation = Simulation(model, Δt=0.01, stop_iteration=3)

simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, (; ζ)),
                                                      schedule = IterationInterval(1),
                                                      with_halos = true,
                                                      prefix = "test_output_writing_rank$rank",
                                                      force = true)

run!(simulation)

