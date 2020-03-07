"""
Potential flow around a circular cylinder using a continuous forcing immersed boundary method
See: https://en.wikipedia.org/wiki/Potential_flow_around_a_circular_cylinder
"""

using Printf

using Oceananigans
using Oceananigans.Diagnostics
using Oceananigans.OutputWriters

const U = 1.0  # Mean flow
const V = 0.0  # Mean flow
const R = 1.0  # Radius of the cylinder

topology = (Periodic, Flat, Bounded)
domain = (x=(-3R, 3R), y=(-1, 1), z=(-3R, 3R))

N = Nx = Nz = 256
grid = RegularCartesianGrid(topology=topology, size=(Nx, 1, Nz); domain...)

@inline radius(x, z) = x^2 + z^2
@inline boundary(x, y, z) = radius(x, z) <= R ? 1.0 : 0.0

# Continuous forcing immersed boundary method
@inline u_immersed_boundary(i, j, k, grid, t, U, C, p) = @inbounds - boundary(grid.xF[i], grid.yC[j], grid.zC[k]) * p.K * U.u[i, j, k]
@inline u_far_field(i, j, k, grid, t, U, C, p) = @inbounds ifelse(abs(grid.xF[i]) >= 2R, - p.K * (U.u[i, j, k] - p.U∞), 0)
@inline u_forcing(args...) = u_immersed_boundary(args...) + u_far_field(args...)

@inline w_immersed_boundary(i, j, k, grid, t, U, C, p) = @inbounds - boundary(grid.xC[i], grid.yC[j], grid.zF[k]) * p.K * U.w[i, j, k]

K = 1.0 # "Spring constant" for immersed boundary method
parameters = (K=K, U∞=U)
forcing = ModelForcing(u=u_forcing, w=w_immersed_boundary)

model = IncompressibleModel(
    grid = grid,
    buoyancy = nothing,
    tracers = nothing,
    forcing = forcing,
    parameters = parameters,
    closure = ConstantIsotropicDiffusivity(ν=0)
)

Δt = 0.1 * model.grid.Δx / U
cfl = AdvectiveCFL(Δt)

function print_progress(simulation)
    model = simulation.model

    # Calculate simulation progress in %.
    progress = 100 * (model.clock.time / simulation.stop_time)

    # Find maximum velocities.
    umax = maximum(abs, model.velocities.u.data.parent)
    wmax = maximum(abs, model.velocities.w.data.parent)

    i, t = model.clock.iteration, model.clock.time
    @printf("[%06.2f%%] i: %d, t: %.3f, U_max: (%.2e, %.2e), CFL: %.2e, next Δt: %.2e s\n",
            progress, i, t, umax, wmax, cfl(model), simulation.Δt)
end

simulation = Simulation(model, Δt=Δt, stop_time=100, progress=print_progress, progress_frequency=10)

fields = Dict("u" => model.velocities.u, "w" => model.velocities.w)
simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, fields, interval=0.1, filename="flow_around_cylinder.nc")

print_progress(simulation)
run!(simulation)

