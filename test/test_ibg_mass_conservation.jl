using Oceananigans
using Oceananigans.Diagnostics: AdvectiveCFL
using Printf
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Oceananigans.BoundaryConditions: PerturbationAdvectionOpenBoundaryCondition


# Create regular underlying grid
underlying_grid = RectilinearGrid(topology = (Bounded, Periodic, Bounded),
                                  size = (4, 4, 4), extent = (1, 1, 1))

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(-0.6))
cg_solver = ConjugateGradientPoissonSolver(grid)

# Set up open boundary conditions with perturbation advection
U = 1.0  # Background velocity
inflow_timescale = 1e-4
outflow_timescale = Inf

u_boundaries = FieldBoundaryConditions(
    west = PerturbationAdvectionOpenBoundaryCondition(U; inflow_timescale, outflow_timescale),
    east = PerturbationAdvectionOpenBoundaryCondition(U; inflow_timescale, outflow_timescale)
)

# Create nonhydrostatic model with immersed boundary grid and open boundaries
model = NonhydrostaticModel(; grid,
                              boundary_conditions = (u = u_boundaries,),
                              pressure_solver = cg_solver,
                              advection = WENO(order=5))

# Set initial velocity field with small perturbations
u₀(x, y, z) = U + 1e-2 * rand()
set!(model, u=u₀)

# Test pressure correction with immersed boundaries
Δt = 0.1 * minimum_zspacing(grid) / abs(U)
simulation = Simulation(model; Δt, stop_time=50, verbose=false)

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
add_callback!(simulation, progress, IterationInterval(20))
run!(simulation)