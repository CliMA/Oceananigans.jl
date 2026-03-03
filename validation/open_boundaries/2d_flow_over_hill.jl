using Oceananigans
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, fft_poisson_solver

function run_2d_flow_over_hill(; scheme = PerturbationAdvection(),
                                 arch = CPU(),
                                 model_type = NonhydrostaticModel,
                                 Nz = 16,
                                 H = 1,
                                 Lx = 10,
                                 U = 1,
                                 pressure_solver_constructor = nothing,
                                 stop_time = 50,
                                 cfl = 0.8,
                                 debug = false,
                                 simname = "2d_flow_over_hill",
                                 output = false)

    # Grid definition
    Lz = 2 * H
    Nx = (Nz * Lx) ÷ Lz
    x₀ = Lx / 3

    if model_type == NonhydrostaticModel
        z = (-Lz, 0)
    else
        z = MutableVerticalDiscretization(-Lz:(Lz/Nz):0)
    end
    grid_base = RectilinearGrid(arch; topology = (Bounded, Flat, Bounded), size = (Nx, Nz), x = (0, Lx), z, halo = (8, 8))
    @show grid_base
    hill(x) = H * exp(-((x - x₀)/2)^2) - Lz
    grid = ImmersedBoundaryGrid(grid_base, GridFittedBottom(hill))

    # Set default pressure solver if not provided
    if isnothing(pressure_solver_constructor)
        pressure_solver = ConjugateGradientPoissonSolver(grid, maxiter=5, preconditioner=fft_poisson_solver(grid.underlying_grid))
    else
        pressure_solver = pressure_solver_constructor(grid)
    end

    # Create boundary conditions
    u_boundaries = FieldBoundaryConditions(
        west = OpenBoundaryCondition(U; scheme),
        east = OpenBoundaryCondition(U; scheme)
    )
    boundary_conditions = (u = u_boundaries,)

    # Model kwargs
    kwargs = (; boundary_conditions,)
    advection = WENO(; order=5, minimum_buffer_upwind_order=1)
    if model_type == NonhydrostaticModel
        kwargs = merge(kwargs, (; pressure_solver, advection))
    else
        kwargs = merge(kwargs, (; free_surface = ImplicitFreeSurface(), momentum_advection = advection, tracer_advection = advection, vertical_coordinate = ZStarCoordinate()))
    end
    model = model_type(grid; kwargs...)
    set!(model, u=U)

    Δt = 0.1 * minimum_xspacing(grid) / abs(U)
    simulation = Simulation(model; Δt = Δt, stop_time, verbose = debug)

    conjure_time_step_wizard!(simulation, IterationInterval(1); cfl)

    if debug
        progress = ProgressMessengers.TimedMessenger()
        add_callback!(simulation, progress, IterationInterval(100))
    end

    if output
        output_dir = "output"
        mkpath(output_dir)
        u, v, w = model.velocities
        ω = ∂z(u) - ∂x(w)
        outputs = (; ω, model.velocities...)

        if model_type == HydrostaticFreeSurfaceModel
            outputs = merge(outputs, (; η = model.free_surface.displacement))
        end
        simulation.output_writers[:snaps] = JLD2Writer(model, outputs,
                                                       schedule = TimeInterval(0.5),
                                                       filename = joinpath(output_dir, simname),
                                                       overwrite_existing = true,
                                                       with_halos = true)
    end

    run!(simulation)

    if model_type == NonhydrostaticModel
        return !any(isnan, interior(model.pressures.pNHS))
    else
        return !any(isnan, interior(model.free_surface.displacement))
    end
end
