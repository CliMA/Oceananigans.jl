using Oceananigans
using Oceananigans.Models.NonhydrostaticModels: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver
using Oceananigans.DistributedComputations: reconstruct_global_field, @handshake
using Oceananigans.Models.NonhydrostaticModels: nonhydrostatic_pressure_solver
using Oceananigans.Solvers: compute_laplacian!
# using CairoMakie
using Printf

function initial_conditions!(model)
    h = 0.05
    x₀ = 0.5
    y₀ = 0.5
    z₀ = 0.55
    bᵢ(x, y, z) = - exp(-((x - x₀)^2 + (y - y₀)^2 + (z - z₀)^2) / 2h^2)

    set!(model, b=bᵢ)
end

function setup_grid(N, arch)
    grid = RectilinearGrid(arch, Float64,
                        size = (N, N, N), 
                        halo = (4, 4, 4),
                        x = (0, 1),
                        y = (0, 1),
                        z = (0, 1),
                        topology = (Bounded, Bounded, Bounded))

    slope(x, y) = (5 + tanh(40*(x - 1/6)) + tanh(40*(x - 2/6)) + tanh(40*(x - 3/6)) + tanh(40*(x - 4/6)) + tanh(40*(x - 5/6))) / 20 + 
                  (5 + tanh(40*(y - 1/6)) + tanh(40*(y - 2/6)) + tanh(40*(y - 3/6)) + tanh(40*(y - 4/6)) + tanh(40*(y - 5/6))) / 20

    grid = ImmersedBoundaryGrid(grid, GridFittedBottom(slope))
    return grid
end

function setup_model(grid, pressure_solver)
    model = NonhydrostaticModel(; grid, pressure_solver,
                                  advection = WENO(),
                                  coriolis = FPlane(f=0.1),
                                  tracers = :b,
                                  buoyancy = BuoyancyTracer())

    initial_conditions!(model)
    return model
end

reltol = abstol = 1e-7

function setup_simulation(model, Δt, stop_iteration)
    return Simulation(model, Δt=Δt, stop_iteration=stop_iteration)
end

function setup_simulation(model)
    Δt = 2e-2
    stop_iteration = 100
    simulation = Simulation(model; Δt = Δt, stop_iteration = stop_iteration, minimum_relative_step = 1e-10)
    
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
    
        msg = @sprintf("rank %d, iter: %d, time: %s, Δt: %.4f, Poisson iters: %d",
                        local_rank, iteration(sim), prettytime(time(sim)), sim.Δt, pressure_iters)
    
        elapsed = 1e-9 * (time_ns() - wall_time[])
    
        u, v, w = sim.model.velocities
        d = Field(∂x(u) + ∂y(v) + ∂z(w))
        compute!(d)
    
        msg *= @sprintf(", max u: %6.3e, max w: %6.3e, max b: %6.3e, max d: %6.3e, max pressure: %6.3e, wall time: %s",
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

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(1))
    
    u, v, w = model.velocities
    d = Field(∂x(u) + ∂y(v) + ∂z(w))

    b = model.tracers.b
    p = model.pressures.pNHS
    
    prefix = "2D_staircase_convection"

    if model.architecture isa Distributed
        prefix *= "_distributed"
    else
        prefix *= "_nondistributed"
    end

    # outputs = (; u, v, w, b, d, p)

    # OUTPUT_PATH = "./"
    # simulation.output_writers[:jld2] = JLD2Writer(model, outputs,
    #                                                     schedule = TimeInterval(0.1),
    #                                                     filename = "$(OUTPUT_PATH)/$(prefix)_fields",
    #                                                     overwrite_existing = true,
    #                                                     with_halos = true)

    return simulation
end

const N = 32

# arch = Distributed(CPU())
arch = Distributed(GPU())

grid = setup_grid(N, arch)

@info "Create pressure solver"
preconditioner = nonhydrostatic_pressure_solver(grid.underlying_grid)
# preconditioner = nothing
pressure_solver = ConjugateGradientPoissonSolver(
    grid, maxiter=10000, preconditioner=preconditioner)
# pressure_solver = nonhydrostatic_pressure_solver(underlying_grid)
# pressure_solver = nothing

@info "Create model"
model = setup_model(grid, pressure_solver)

simulation = setup_simulation(model)

run!(simulation)