# Run this script with
#
# $ mpirun -n 2 julia --project distributed_nonhydrostatic_turbulence.jl
#
# for example.
#
# You also probably should set
#
# $ export JULIA_NUM_THREADS=1
#
# See MPI.jl documentation for more information on how to setup the MPI environment.
# If you have a local installation of MPI, you can use it by setting
#
# julia> MPIPreferences.use_system_binaries()
#
# before running the script.

using MPI
using Oceananigans
using Oceananigans.DistributedComputations
using Statistics
using Printf
using Random

Nx = Ny = 256
Lx = Ly = 2π
topology = (Periodic, Periodic, Flat)
arch = Distributed(CPU())
grid = RectilinearGrid(arch; topology, size=(Nx, Ny), halo=(3, 3), x=(0, 2π), y=(0, 2π))

@show grid

model = NonhydrostaticModel(; grid, advection=WENO(), closure=ScalarDiffusivity(ν=1e-4, κ=1e-4))

# Make sure we use different seeds for different cores.
rank = arch.local_rank
Random.seed!((rank+ 1) * 1234)

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
    rank = sim.model.grid.architecture.local_rank
    compute!(ζ)
    compute!(e)

    rank == 0 && @info(string("Iteration: ", iteration(sim), ", time: ", prettytime(sim)))

    @info @sprintf("Rank %d: max|ζ|: %.2e, max(e): %.2e",
                   rank, maximum(abs, ζ), maximum(abs, e))

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

outputs = merge(model.velocities, (; e, ζ))

simulation.output_writers[:fields] = JLD2Writer(model, outputs,
                                                schedule = TimeInterval(0.1),
                                                with_halos = true,
                                                filename = "two_dimensional_turbulence_rank$rank",
                                                overwrite_existing = true)

run!(simulation)

