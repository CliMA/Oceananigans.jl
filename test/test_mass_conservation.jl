using Oceananigans
using Oceananigans.BoundaryConditions: PerturbationAdvectionOpenBoundaryCondition
using Oceananigans.Diagnostics: AdvectiveCFL
using Printf
using Random: seed!
seed!(156)

Δx = Δz = 0.05
Nx = Nz = round(Int, 2 / Δx)
grid = RectilinearGrid(size = (Nx, Nz), halo = (4, 4), extent = (1, 1),
                       topology = (Bounded, Flat, Bounded))

U₀ = 1.0
inflow_timescale = 1e-1
outflow_timescale = Inf
u_boundary_conditions = FieldBoundaryConditions(west = PerturbationAdvectionOpenBoundaryCondition(U₀; inflow_timescale, outflow_timescale),
                                                east = PerturbationAdvectionOpenBoundaryCondition(U₀; inflow_timescale, outflow_timescale))
boundary_conditions = (; u = u_boundary_conditions,)

model = NonhydrostaticModel(; grid, boundary_conditions,
                            timestepper = :RungeKutta3,)

uᵢ(x, z) = U₀ + 1e-4 * rand()
fill!(model.velocities.u, U₀)
set!(model, u=uᵢ)
simulation = Simulation(model; Δt=0.1Δx/U₀, stop_time=1, verbose=false)

cfl_calculator = AdvectiveCFL(simulation.Δt)
function progress(sim)
    u, v, w = model.velocities
    cfl_value = cfl_calculator(model)
    west_mass_flux = Field(Average(view(u, 1, :, :)))[]
    east_mass_flux = Field(Average(view(u, grid.Nx+1, :, :)))[]
    net_mass_flux = east_mass_flux - west_mass_flux
    @info @sprintf("time: %.3f, max|u|: %.3f, CFL: %.2f, west mass flux: %.6e, east mass flux: %.6e, Net mass flux: %.4e",
                   time(sim), maximum(abs, u), cfl_value, west_mass_flux, east_mass_flux, net_mass_flux)
end
add_callback!(simulation, progress, IterationInterval(20))
run!(simulation)
