using MPI
using Oceananigans
using Oceananigans.Distributed

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
Nranks = MPI.Comm_size(comm)

@info "Running on rank $rank of $Nranks"

Nx = Ny = 256
Lx = Ly = 2π
topology = (Periodic, Periodic, Flat)
arch = MultiArch(CPU(); topology, ranks=(1, 2, 1))
grid = RectilinearGrid(; topology, size=(Nx, Ny), halo=(3, 3), x=(0, 2π), y=(0, 2π))
model = NonhydrostaticModel(arch; grid, advection=WENO5(), closure=ScalarDiffusivity(ν=1e-4, κ=1e-4))

ϵ(x, y, z) = 2rand() - 1 # ∈ (-1, 1)
set!(model, u=ϵ, v=ϵ)

u, v, w = model.velocities
e_op = @at (Center, Center, Center) 1/2 * (u^2 + v^2)
e = Field(e_op)
compute!(e)
@show e

simulation = Simulation(model, Δt=0.01, stop_time=10)

run!(simulation)

compute!(e)
@show e

