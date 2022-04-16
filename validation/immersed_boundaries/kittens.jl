using Oceananigans
using Oceananigans.Utils: prettysummary
using GLMakie

underlying_grid = RectilinearGrid(size = (256, 128), halo = (3, 3),
                                  x = (-2, 4),
                                  y = (-2, 2),
                                  topology = (Periodic, Periodic, Flat))

@inline annulus(x, y, z) = ((x^2 + y^2) < 1) & ((x^2 + y^2) > 0.7)
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBoundary(annulus); precompute_mask=false)
closure = ScalarDiffusivity(ν=1e-3)
advection = WENO5()
no_slip = ValueBoundaryCondition(0)
boundary_conditions = (; u = FieldBoundaryConditions(immersed=no_slip))
model = NonhydrostaticModel(; grid, closure, advection, timestepper=:RungeKutta3)

# Initial condition = steady flow outside annulus + random perturbations
ϵ(x, y, z) = 2rand() - 1
uᵢ(x, y, z) = ifelse((x^2 + y^2) > 1, 0.5, 0.0) + ϵ(x, y, z)
set!(model, u=uᵢ, v=ϵ)

simulation = Simulation(model, Δt=5e-2, stop_time=4)

wizard = TimeStepWizard(cfl=0.8)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

wall_clock = Ref(time_ns())
function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    @info string("Iteration: ", iteration(sim),
                 ", time: ", prettysummary(time(sim)),
                 ", wall time: ", prettytime(elapsed))
    wall_clock[] = time_ns()
end

simulation.callbacks[:p] = Callback(progress, IterationInterval(10))

u, v, w = model.velocities
ζ = ∂x(v) - ∂y(u)

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; ζ),
                                                      filename = "kittens",
                                                      schedule = TimeInterval(0.1),
                                                      overwrite_existing = true)

run!(simulation)

ζt = FieldTimeSeries("kittens.jld2", "ζ")
Nt = length(ζt.times)

fig = Figure()
ax = Axis(fig[1, 1], aspect=3/2)
slider = Slider(fig[2, 1], range=1:Nt, startvalue=1)
n = slider.value

ζn = @lift interior(ζt[$n], :, :, 1)
heatmap!(ax, ζn)

display(fig)

record(fig, "kittens.mp4", 1:Nt, framerate=24) do nn
    n[] = nn
end
