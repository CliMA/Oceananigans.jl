using Oceananigans
using Oceananigans.BoundaryConditions: PerturbationAdvectionOpenBoundaryCondition

U(y, z, t) = 1#(1 + tanh(3t))/2

u_east = PerturbationAdvectionOpenBoundaryCondition(U, inflow_timescale = Inf, outflow_timescale = Inf)
u_west = PerturbationAdvectionOpenBoundaryCondition(U, inflow_timescale = 1.0, outflow_timescale = Inf)#OpenBoundaryCondition(U)

u_bcs = FieldBoundaryConditions(east = u_east, west = u_west)
v_bcs = FieldBoundaryConditions(east = GradientBoundaryCondition(0), west = GradientBoundaryCondition(0))
w_bcs = FieldBoundaryConditions(east = GradientBoundaryCondition(0), west = GradientBoundaryCondition(0))

grid = RectilinearGrid(topology = (Bounded, Periodic, Bounded), size = (32, 16, 1), x = (-1, 1), y = (-0.5, 0.5), z = (-1, 0))

circle(x, y) = ifelse((x^2 + y^2) < 0.1^2, 0, -1)

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(circle))

model = NonhydrostaticModel(; grid, 
                              boundary_conditions = (u = u_bcs, v = v_bcs, w = w_bcs),
                              advection = WENO(order = 5))

Δt = 0.3 * minimum_xspacing(grid) / 2

simulation = Simulation(model; Δt, stop_time = 20)

simulation.output_writers[:fields] = JLD2OutputWriter(model, model.velocities; filename = "cylinder_$(grid.underlying_grid.Nz).jld2", schedule = TimeInterval(0.1), overwrite_existing = true)

run!(simulation)