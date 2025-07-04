using Oceananigans
using Oceananigans.BoundaryConditions: PerturbationAdvectionOpenBoundaryCondition
using BenchmarkTools

Δx = Δz = 0.05
Nx = Nz = round(Int, 2 / Δx)
grid = RectilinearGrid(size = (Nx, Nz), halo = (4, 4), extent = (1, 1),
                       topology = (Bounded, Flat, Bounded))

U₀ = 1.0
inflow_timescale = 1e-1
outflow_timescale = Inf
u_boundary_conditions = FieldBoundaryConditions(west = OpenBoundaryCondition(U₀),
                                                east = PerturbationAdvectionOpenBoundaryCondition(U₀; inflow_timescale, outflow_timescale))
boundary_conditions = (; u = u_boundary_conditions)

model = NonhydrostaticModel(; grid,
                              timestepper=:QuasiAdamsBashforth2,
                              boundary_conditions
                              )

simulation = Simulation(model; Δt=1, stop_time=1e3, verbose=false)
ū = Average(view(model.velocities.u, 1, :, :), dims=(2, 3)) |> Field
ū2 = interior(model.velocities.u, 1, :, :)

@inline function west_avg(i, j, k, grid, u)
    i = 1
    U = 0.0
    for j in 1:grid.Ny, k in 1:grid.Nz
        U += u[i, j, k]
    end
    return U / (grid.Ny * grid.Nz)
end
ū3_op = KernelFunctionOperation{Nothing, Nothing, Nothing}(west_avg, grid, model.velocities.u)
ū3 = Field(ū3_op)

compute_ū(sim) = compute!(ū)
compute_ū2(sim) = mean(ū2)
compute_ū3(sim) = compute!(ū3)
#add_callback!(simulation, compute_ū, IterationInterval(1))
#add_callback!(simulation, compute_ū2, IterationInterval(1))
@btime time_step!(simulation)