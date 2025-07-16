using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: PerturbationAdvectionOpenBoundaryCondition
using Oceananigans.Diagnostics: AdvectiveCFL
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver

using Printf
using Random: seed!
using Test
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
    Δt = 0.1 * minimum_zspacing(grid) / abs(maximum(model.velocities.u))
    simulation = Simulation(model; Δt, stop_time, verbose=false)

    if add_progress_messenger
        # Set up progress monitoring
        u, v, w = model.velocities
        ∫∇u = Field(Average(Field(∂x(u) + ∂z(w))))

        function progress(sim)
            u, v, w = model.velocities
            compute!(∫∇u)
            @info @sprintf("time: %.3f, max|u|: %.3f, Net flux: %.4e",
                           time(sim), maximum(abs, u), maximum(∫∇u))
        end
        add_callback!(simulation, progress, IterationInterval(1))
    end
    
    return simulation
end


common_kwargs = (; add_progress_messenger = true, Lx = 2700meters, Lz = 600meters, stop_time = 1hour, U₀ = 0.1)

bottom(x) = -400meters + 100meters * sin(2π * x / 1e3meters)
simulation = create_mass_conservation_simulation(; immersed_bottom = GridFittedBottom(bottom), common_kwargs...);
u, v, w = simulation.model.velocities
∇u = Field(∂x(u) + ∂z(w))

using GLMakie
fig = Figure()  
ax = Axis(fig[1,1])
hm = heatmap!(ax, ∇u, colormap=:balance)
Colorbar(fig[1,2], hm)

@test maximum(abs, Field(Average(∇u))) < 1e-10
