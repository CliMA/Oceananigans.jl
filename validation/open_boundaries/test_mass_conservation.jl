using Oceananigans
using Oceananigans.BoundaryConditions: PerturbationAdvectionOpenBoundaryCondition
using Oceananigans.Diagnostics: AdvectiveCFL
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver
using Printf
using Random: seed!
seed!(156)

function create_mass_conservation_simulation(; 
    use_open_boundary_condition = true,
    immersed_bottom = nothing,
    arch = CPU(),
    N = 32,
    L = 1.0,
    U₀ = 1.0,
    stop_time = 1,
    inflow_timescale = 1e-1,
    outflow_timescale = Inf,
    add_progress_messenger = false,
    poisson_solver = nothing,
    timestepper = :QuasiAdamsBashforth2,)
    # Create underlying grid
    underlying_grid = RectilinearGrid(arch, topology = (Bounded, Flat, Bounded),
                                      size = (N, N),
                                      extent = (L, L),
                                      halo = (4, 4))
    
    # Choose grid type based on immersed_bottom parameter
    grid = immersed_bottom isa Nothing ? underlying_grid : ImmersedBoundaryGrid(underlying_grid, immersed_bottom)

    if poisson_solver isa Nothing
        pressure_solver = grid isa ImmersedBoundaryGrid ? ConjugateGradientPoissonSolver(grid) : nothing
    elseif (poisson_solver == FFTBasedPoissonSolver) && (grid isa ImmersedBoundaryGrid)
        pressure_solver = poisson_solver(grid.underlying_grid)
    else
        pressure_solver = poisson_solver(grid)
    end

    # Set boundary conditions based on boolean flag
    if use_open_boundary_condition
        u_boundary_conditions = FieldBoundaryConditions(
            west = OpenBoundaryCondition(U₀),
            east = PerturbationAdvectionOpenBoundaryCondition(U₀; inflow_timescale, outflow_timescale)
        )
        boundary_conditions = (; u = u_boundary_conditions)
    else
        boundary_conditions = NamedTuple()
    end

    model = NonhydrostaticModel(; grid, boundary_conditions, pressure_solver, timestepper)
    uᵢ(x, z) = U₀ + 1e-2 * rand()
    set!(model, u=uᵢ)

    # Calculate time step
    Δt = 0.1 * minimum_zspacing(grid) / abs(U₀)
    simulation = Simulation(model; Δt, stop_time, verbose=false)

    if add_progress_messenger
        # Set up progress monitoring
        u, v, w = model.velocities
        ∫∇u = Field(Integral(∂x(u) + ∂z(w)))
        cfl_calculator = AdvectiveCFL(simulation.Δt)

        function progress(sim)
            u, v, w = model.velocities
            cfl_value = cfl_calculator(model)
            compute!(∫∇u)
            @info @sprintf("time: %.3f, max|u|: %.3f, CFL: %.2f, Net flux: %.4e",
                           time(sim), maximum(abs, u), cfl_value, ∫∇u[])
        end
        add_callback!(simulation, progress, IterationInterval(50))
    end
    
    return simulation
end

simulation = create_mass_conservation_simulation(; immersed_bottom = nothing);
run!(simulation)
