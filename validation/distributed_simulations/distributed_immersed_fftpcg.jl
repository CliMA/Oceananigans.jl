# Distributed immersed boundary FFT-PCG solver validation
#
# Run with:
#
#   mpiexec -n 2 julia --project distributed_immersed_fftpcg.jl
#

using MPI
using Oceananigans
using Oceananigans.Models.NonhydrostaticModels: ConjugateGradientPoissonSolver
using Oceananigans.Models.NonhydrostaticModels: nonhydrostatic_pressure_solver
using Oceananigans.DistributedComputations
using Printf

function initial_conditions!(model)
    h = 0.05
    x₀ = 0.5
    y₀ = 0.5
    z₀ = 0.55
    bᵢ(x, y, z) = -exp(-((x - x₀)^2 + (y - y₀)^2 + (z - z₀)^2) / 2h^2)
    set!(model, b = bᵢ)
end

function setup_grid(N, arch)
    grid = RectilinearGrid(arch, Float64,
                           size = (N, N, N),
                           halo = (4, 4, 4),
                           x = (0, 1),
                           y = (0, 1),
                           z = (0, 1),
                           topology = (Bounded, Bounded, Bounded))

    slope(x, y) = (5 + tanh(40*(x - 1/6)) + tanh(40*(x - 2/6)) + tanh(40*(x - 3/6)) +
                       tanh(40*(x - 4/6)) + tanh(40*(x - 5/6))) / 20 +
                  (5 + tanh(40*(y - 1/6)) + tanh(40*(y - 2/6)) + tanh(40*(y - 3/6)) +
                       tanh(40*(y - 4/6)) + tanh(40*(y - 5/6))) / 20

    grid = ImmersedBoundaryGrid(grid, GridFittedBottom(slope))
    return grid
end

function setup_model(grid, pressure_solver)
    model = NonhydrostaticModel(; grid, pressure_solver,
                                advection = WENO(),
                                coriolis = FPlane(f = 0.1),
                                tracers = :b,
                                buoyancy = BuoyancyTracer())

    initial_conditions!(model)
    return model
end

function setup_simulation(model)
    Δt = 2e-2
    stop_iteration = 100
    simulation = Simulation(model; Δt, stop_iteration, minimum_relative_step = 1e-10)

    wall_time = Ref(time_ns())

    function progress(sim)
        pressure_solver = sim.model.pressure_solver

        if pressure_solver isa ConjugateGradientPoissonSolver
            pressure_iters = iteration(pressure_solver)
        else
            pressure_iters = 0
        end

        if sim.model.architecture isa Distributed
            local_rank = sim.model.architecture.local_rank
        else
            local_rank = 0
        end

        elapsed = 1e-9 * (time_ns() - wall_time[])

        u, v, w = sim.model.velocities
        d = Field(∂x(u) + ∂y(v) + ∂z(w))
        compute!(d)

        msg = @sprintf("rank %d, iter: %d, time: %s, Δt: %.4f, Poisson iters: %d",
                       local_rank, iteration(sim), prettytime(time(sim)), sim.Δt, pressure_iters)

        msg *= @sprintf(", max|u|: %.2e, max|w|: %.2e, max|b|: %.2e, max|div|: %.2e, max|p|: %.2e, wall time: %s",
                        maximum(abs, sim.model.velocities.u),
                        maximum(abs, sim.model.velocities.w),
                        maximum(abs, sim.model.tracers.b),
                        maximum(abs, d),
                        maximum(abs, sim.model.pressures.pNHS),
                        prettytime(elapsed))

        @info msg
        wall_time[] = time_ns()

        return nothing
    end

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

    return simulation
end

N = 32
reltol = abstol = 1e-7

arch = Distributed(CPU())

grid = setup_grid(N, arch)

@info "Creating pressure solver"
preconditioner = nonhydrostatic_pressure_solver(grid.underlying_grid)
pressure_solver = ConjugateGradientPoissonSolver(grid, maxiter = 10000, preconditioner = preconditioner)

@info "Creating model"
model = setup_model(grid, pressure_solver)

simulation = setup_simulation(model)

run!(simulation)

@info "Simulation completed"
