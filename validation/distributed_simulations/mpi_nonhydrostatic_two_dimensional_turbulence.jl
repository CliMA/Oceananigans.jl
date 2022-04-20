# Run this script with
#
# $ mpiexec -n 2 julia --project mpi_nonhydrostatic_two_dimensional_turbulence.jl
#
# for example.
#
# You also probably should set
#
# $ export JULIA_NUM_THREADS=1

using MPI
using Oceananigans
using Oceananigans.Distributed
using Statistics
using Printf
using Logging

Logging.global_logger(OceananigansLogger())

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
Nranks = MPI.Comm_size(comm)

@info "Running on rank $rank of $Nranks..."

Nx = Ny = 256
Lx = Ly = 2π
topology = (Periodic, Periodic, Flat)
arch = MultiArch(CPU(); topology, ranks=(1, Nranks, 1))
grid = RectilinearGrid(arch; topology, size=(Nx, Ny), halo=(3, 3), x=(0, 2π), y=(0, 2π))

@info "Built $Nranks grids:"
@show grid

model = NonhydrostaticModel(; grid, advection=WENO5(), closure=ScalarDiffusivity(ν=1e-4, κ=1e-4))

# This doesn't work?
# ϵ(x, y, z) = 2rand() - 1 # ∈ (-1, 1)
# set!(model, u=ϵ, v=ϵ)

uᵢ = rand(size(grid)...)
vᵢ = rand(size(grid)...)
uᵢ .-= mean(uᵢ)
vᵢ .-= mean(vᵢ)
set!(model, u=uᵢ, v=vᵢ)

u, v, w = model.velocities
e_op = @at (Center, Center, Center) 1/2 * (u^2 + v^2)
e = Field(e_op)
ζ = Field(∂x(v) - ∂y(u))
compute!(e)
compute!(ζ)

simulation = Simulation(model, Δt=0.01, stop_iteration=1000)

function progress(sim)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    compute!(ζ)
    compute!(e)

    rank == 0 && @info(string("Iteration: ", iteration(sim), ", time: ", prettytime(sim)))

    @info @sprintf("Rank %d: max|ζ|: %.2e, max(e): %.2e",
                   MPI.Comm_rank(comm), maximum(abs, ζ), maximum(abs, e))

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

rank = MPI.Comm_rank(MPI.COMM_WORLD)
outputs = merge(model.velocities, (; e, ζ))
simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                      schedule = TimeInterval(0.1),
                                                      with_halos = true,
                                                      prefix = "two_dimensional_turbulence_rank$rank",
                                                      force = true)

run!(simulation)

