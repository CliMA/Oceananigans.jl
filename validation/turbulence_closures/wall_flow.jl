using Oceananigans
using Oceananigans.Units
using Printf
simname = "wall_flow"

const κ = 0.4
H = 1
L = 2π*H
z₀ = 1e-4*H
N = 32
u★ = 1

grid = RectilinearGrid(arch=CPU(), size=(N, N, N), topology=(Periodic, Periodic, Bounded),
                       x=(0, L), y=(0, L), z=(0, H))

z₁ = first(znodes(grid, Center()))
cᴰ = (κ / log(z₁ / z₀))^2

@inline drag_u(x, y, t, u, v, p) = - p.cᴰ * √(u^2 + v^2) * u
@inline drag_v(x, y, t, u, v, p) = - p.cᴰ * √(u^2 + v^2) * v

u_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(drag_u, field_dependencies = (:u, :v), parameters = (; cᴰ)))
v_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(drag_v, field_dependencies = (:u, :v), parameters = (; cᴰ)))

@inline x_pressure_gradient(x, y, z, t, p) = p.u★^2 / p.H
u_forcing = Forcing(x_pressure_gradient, parameters=(; u★, H))

closure = ScaleInvariantSmagorinsky(averaging=(1,2))
#closure = SmagorinskyLilly()
model = NonhydrostaticModel(; grid, timestepper = :RungeKutta3,
                            advection = CenteredFourthOrder(),
                            boundary_conditions = (; u=u_bcs, v=v_bcs),
                            forcing = (; u = u_forcing),
                            closure = closure)
@show model

noise(x, y, z) = 1e0 * u★ * randn()
u₀(x, y, z) = (u★/κ) * log(z/z₀)
set!(model, u=u₀, v=noise, w=noise)

Δt₀ = 1e-4 * (H / u★) / N
simulation = Simulation(model, Δt = Δt₀, stop_time = 20)

wizard = TimeStepWizard(max_change=1.1, cfl=0.9)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(2))

start_time = time_ns() # so we can print the total elapsed wall time
progress_message(sim) = @printf("Iteration: %04d,  time: %s,  Δt: %s,  max|u|: %.1e m/s,  wall time: %s\n",
                                iteration(sim), prettytime(time(sim)), prettytime(sim.Δt), maximum(abs, sim.model.velocities.u), prettytime((time_ns() - start_time) * 1e-9))
add_callback!(simulation, Callback(progress_message, IterationInterval(100)))

closure_name = string(nameof(typeof(closure)))
u, v, w = model.velocities

U = Average(u, dims=(1,2))
V = Average(v, dims=(1,2))
w² = w*w
uw = u*w
vw = v*w
τ = √(uw^2 + vw^2)

#using Oceananigans.Grids: znode
#z = KernelFunctionOperation{Center, Center, Center}(znode, grid, Center(), Center(), Center())

outputs = (; u, v, w, U, V, w², τ)

simulation.output_writers[:fields] = NetCDFOutputWriter(model, outputs;
                                                        filename = joinpath(@__DIR__, simname *"_"* closure_name*".nc"),
                                                        schedule = TimeInterval(1),
                                                        overwrite_existing = true)

run!(simulation)
