using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: PerturbationAdvectionOpenBoundaryCondition, OpenBoundaryCondition
using Oceananigans.Diagnostics: AdvectiveCFL
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver, DiagonallyDominantPreconditioner, fft_poisson_solver

using Printf
using Test
using BenchmarkTools
using Random: seed!
seed!(156)

function create_mass_conservation_simulation(; 
    use_open_boundary_condition = true,
    immersed_bottom = nothing,
    arch = CPU(),
    N = 32,
    Lx = 1.0,
    Lz = 1.0,
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
                                      extent = (Lx, Lz),
                                      halo = (4, 4))
    
    # Choose grid type based on immersed_bottom parameter
    grid = immersed_bottom isa Nothing ? underlying_grid : ImmersedBoundaryGrid(underlying_grid, immersed_bottom)

    if poisson_solver isa Nothing
        pressure_solver = grid isa ImmersedBoundaryGrid ? ConjugateGradientPoissonSolver(grid) : nothing
    elseif poisson_solver == :fft
        if grid isa ImmersedBoundaryGrid
            pressure_solver = FFTBasedPoissonSolver(grid.underlying_grid)
        else
            pressure_solver = FFTBasedPoissonSolver(grid)
        end
    elseif poisson_solver == :conjugate_gradient_with_diagonally_dominant_preconditioner
        pressure_solver = ConjugateGradientPoissonSolver(grid, preconditioner=DiagonallyDominantPreconditioner())
    elseif poisson_solver == :conjugate_gradient_with_fft_preconditioner
        pressure_solver = ConjugateGradientPoissonSolver(grid, preconditioner=fft_poisson_solver(grid.underlying_grid))
    else
        error("Unknown poisson_solver option: $poisson_solver. Valid options are: nothing, :fft, :conjugate_gradient_with_diagonally_dominant_preconditioner, :conjugate_gradient_with_fft_preconditioner")
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
    Δt = 0.1 * minimum_zspacing(grid) / abs(maximum(model.velocities.u))
    simulation = Simulation(model; Δt, stop_time, verbose=false)
    conjure_time_step_wizard!(simulation, IterationInterval(1), cfl=0.1)

    #+++ Progress Messenger
    if add_progress_messenger
        # Set up progress monitoring
        u, v, w = model.velocities
        ∫∇u = Field(Average(Field(∂x(u) + ∂z(w))))

        function progress(sim)
            u, v, w = model.velocities
            compute!(∫∇u)
            max_u = maximum(abs, u)
            @info @sprintf("time: %s, max|u|: %.3f, Net flux: %.4e",
                           prettytime(time(sim)), max_u, maximum(∫∇u))
            u_critical = 1e2
            if max_u > u_critical
                @warn "max|u| > $u_critical, stopping simulation"
                stop_time(sim) = time(sim)
                sim.running = false
            end
        end
        add_callback!(simulation, progress, IterationInterval(5))
    end
    #---
    
    return simulation
end

bottom(x) = -400meters + 100meters * sin(2π * x / 1e3meters)
common_kwargs = (; arch=GPU(),
                   immersed_bottom = GridFittedBottom(bottom),
                   Lx = 2700meters,
                   Lz = 600meters,
                   stop_time = 1day,
                   U₀ = 0.1,
                   add_progress_messenger = true,
                   timestepper = :RungeKutta3)

simulation = create_mass_conservation_simulation(; use_open_boundary_condition = true, common_kwargs...);
u, v, w = simulation.model.velocities
∇u = Field(∂x(u) + ∂z(w))
@test maximum(abs, Field(Average(∇u))) < 1e-10
# fig = Figure()
# ax = Axis(fig[1, 1])
# heatmap!(ax, u, colorrange=(-1, 1), colormap=:balance)
# fig
b1 = @benchmark time_step!(simulation)

simulation = create_mass_conservation_simulation(; use_open_boundary_condition = false, common_kwargs...);
u, v, w = simulation.model.velocities
∇u = Field(∂x(u) + ∂z(w))
@test maximum(abs, Field(Average(∇u))) < 1e-10
b2 = @benchmark time_step!(simulation)
