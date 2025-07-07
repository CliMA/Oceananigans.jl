using Oceananigans
using Oceananigans.BoundaryConditions: PerturbationAdvectionOpenBoundaryCondition
using Oceananigans.Diagnostics: AdvectiveCFL
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Printf
using Random: seed!
seed!(156)

function create_open_domain_simulation(; immersed_bottom = nothing,
                                         N = 10,
                                         L = 1.0,
                                         U₀ = 1.0,
                                         inflow_timescale = 1e-1,
                                         outflow_timescale = Inf,
                                         timestepper = :RungeKutta3,
                                         pressure_solver = nothing)
    # Create underlying grid
    underlying_grid = RectilinearGrid(topology = (Bounded, Flat, Bounded),
                                      size = (N, N), 
                                      extent = (L, L),
                                      halo = (4, 4))
    
    if immersed_bottom !== nothing
        grid = ImmersedBoundaryGrid(underlying_grid, immersed_bottom)
        pressure_solver = pressure_solver isa Nothing ? ConjugateGradientPoissonSolver(grid) : pressure_solver
    else
        grid = underlying_grid
    end
    
    # Set up boundary conditions
    u_boundary_conditions = FieldBoundaryConditions(
        west = OpenBoundaryCondition(U₀),
        east = PerturbationAdvectionOpenBoundaryCondition(U₀; inflow_timescale, outflow_timescale)
    )
    
    boundary_conditions = (; u = u_boundary_conditions)
    
    # Create model with appropriate parameters
    model_kwargs = (; grid, boundary_conditions, timestepper)
    
    if grid isa ImmersedBoundaryGrid
        model_kwargs = merge(model_kwargs, (; pressure_solver))
    end
    
    model = NonhydrostaticModel(; model_kwargs...)
    
    # Set initial velocity field with small perturbations
    uᵢ(x, z) = U₀ + 1e-2 * rand()
    fill!(model.velocities.u, U₀)
    set!(model, u=uᵢ)
    
    # Calculate time step
    Δt = 0.1 * minimum_zspacing(grid) / abs(U₀)
    stop_time = 5
    
    simulation = Simulation(model; Δt, stop_time, verbose=false)
    
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
    
    return simulation
end

simulation = create_open_domain_simulation(immersed_bottom=nothing); # No ImmersedBoundary
run!(simulation)
