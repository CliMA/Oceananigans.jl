using Oceananigans
using Oceananigans.Models.NonhydrostaticModels: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver
using Oceananigans.DistributedComputations: reconstruct_global_field, @handshake
using Oceananigans.Models.NonhydrostaticModels: nonhydrostatic_pressure_solver
using Oceananigans.Solvers: compute_laplacian!
# using CairoMakie
using Printf

const Nx = 32
const Ny = 32
const Nz = 32

const Lx = 1
const Ly = 1
const Lz = 1

const Δx = Lz / Nz
const Δy = Lz / Ny
const Δz = Lz / Nz

const Δt = 1e-3

@inline initial_u(x, y, z) = rand()

# arch = Distributed(CPU(); synchronized_communication=true)
# arch = Distributed(CPU())
arch = Distributed(GPU())
# arch = CPU()
# arch = GPU()

@info "Create underlying_grid"
underlying_grid = RectilinearGrid(
    arch,
    size = (Nx, Ny, Nz),
    x = (0.0, Lx),
    y = (0.0, Ly),
    z = (0.0, Lz),
    topology = (Bounded, Bounded, Bounded),
    halo = (4, 4, 4),
)

bottom_height(x, y) = 0.1

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))
# grid = underlying_grid

u_forcing = Forcing((x, y, z, t) -> rand())

@info "Create pressure solver"
preconditioner = nonhydrostatic_pressure_solver(underlying_grid)
# preconditioner = nothing
pressure_solver = ConjugateGradientPoissonSolver(
    grid, maxiter=10000, preconditioner=preconditioner)
# pressure_solver = nonhydrostatic_pressure_solver(underlying_grid)
# pressure_solver = nothing

@info "Create model"
model = NonhydrostaticModel(;
    grid,
    advection = WENO(),
    forcing = (; u = u_forcing),
    pressure_solver = pressure_solver,
)

@info "Set initial values"
set!(model, u = initial_u)

simulation = Oceananigans.Simulation(model; Δt = Δt, stop_iteration = 100)

u, v, w = model.velocities
d = Field(∂x(u) + ∂y(v) + ∂z(w))

function progress(sim)
    if pressure_solver isa ConjugateGradientPoissonSolver
        pressure_iters = iteration(pressure_solver)
    else
        pressure_iters = 0
    end

    msg = @sprintf("rank %d, Iter: %d, time: %6.3e, Δt: %6.3e, Poisson iters: %d",
                    arch.local_rank, iteration(sim), time(sim), sim.Δt, pressure_iters)

    compute!(d)

    msg *= @sprintf(", max u: %6.3e, max v: %6.3e, max w: %6.3e, max d: %6.3e, max pressure: %6.3e",
                    maximum(abs, sim.model.velocities.u),
                    maximum(abs, sim.model.velocities.v),
                    maximum(abs, sim.model.velocities.w),
                    maximum(abs, d),
                    maximum(abs, sim.model.pressures.pNHS),
    )

    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(
    progress,
    IterationInterval(1),
)

run!(simulation)