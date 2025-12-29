# Distributed nonhydrostatic 2D turbulence validation
#
# Run with:
#
#   mpiexec -n 2 julia --project distributed_nonhydrostatic_turbulence.jl
#

using MPI
using Oceananigans
using Oceananigans.DistributedComputations
using Statistics
using Printf
using Random

Nx = Ny = 64
Lx = Ly = 2π
topology = (Periodic, Periodic, Flat)

arch = Distributed(CPU())
grid = RectilinearGrid(arch; topology, size = (Nx, Ny), halo = (3, 3), x = (0, 2π), y = (0, 2π))

@show grid

model = NonhydrostaticModel(grid; advection = WENO(), closure = ScalarDiffusivity(ν = 1e-4, κ = 1e-4))

# Use different seeds for different ranks
rank = arch.local_rank
Random.seed!((rank + 1) * 1234)

uᵢ = rand(size(grid)...)
vᵢ = rand(size(grid)...)
uᵢ .-= mean(uᵢ)
vᵢ .-= mean(vᵢ)
set!(model, u = uᵢ, v = vᵢ)

u, v, w = model.velocities
e_op = @at (Center, Center, Center) 1/2 * (u^2 + v^2)
e = Field(e_op)
ζ = Field(∂x(v) - ∂y(u))
compute!(e)
compute!(ζ)

simulation = Simulation(model, Δt = 0.01, stop_iteration = 100)

function progress(sim)
    rank = sim.model.grid.architecture.local_rank
    compute!(ζ)
    compute!(e)

    rank == 0 && @info string("Iteration: ", iteration(sim), ", time: ", prettytime(sim))

    @info @sprintf("Rank %d: max|ζ|: %.2e, max(e): %.2e", rank, maximum(abs, ζ), maximum(abs, e))

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

run!(simulation)

@info "Simulation completed on rank $rank"
