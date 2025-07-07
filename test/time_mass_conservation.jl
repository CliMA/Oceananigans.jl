using Oceananigans
using Oceananigans.BoundaryConditions: PerturbationAdvectionOpenBoundaryCondition
using BenchmarkTools

function create_mass_conservation_simulation(; use_open_boundary_condition,
                                               callback = nothing,
                                               Δt = 1.0,
                                               stop_time = 1e3,
                                               verbose = false)
    Δx = 0.05
    Nx = round(Int, 2 / Δx)
    grid = RectilinearGrid(size = (Nx, Nx), halo = (4, 4), extent = (1, 1),
                           topology = (Bounded, Flat, Bounded))

    U₀ = 1.0
    inflow_timescale = 1e-1
    outflow_timescale = Inf

    # Set boundary conditions based on boolean flag
    if use_open_boundary_condition
        u_boundary_conditions = FieldBoundaryConditions(west = OpenBoundaryCondition(U₀),
                                                        east = PerturbationAdvectionOpenBoundaryCondition(U₀; inflow_timescale, outflow_timescale))
        boundary_conditions = (; u = u_boundary_conditions)
    else
        boundary_conditions = NamedTuple()
    end

    model = NonhydrostaticModel(; grid,
                                  timestepper=:QuasiAdamsBashforth2,
                                  boundary_conditions
                                  )

    simulation = Simulation(model; Δt=Δt, stop_time=stop_time, verbose=verbose)

    # Add callback if provided
    if callback !== nothing
        add_callback!(simulation, callback, IterationInterval(1))
    end

    return simulation
end

# Example usage with default settings
simulation = create_mass_conservation_simulation(; use_open_boundary_condition = true);

# Benchmark the time stepping
@btime time_step!(simulation)