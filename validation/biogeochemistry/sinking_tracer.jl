using Oceananigans
using Oceananigans.Units
using GLMakie

grid = RectilinearGrid(size=(128, 128), x=(0, 128), z=(-64, 0), topology=(Periodic, Flat, Bounded))

b_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(1e-8))

@inline growth_func(x, y, z, t, p) = 1 / p.τ * exp(z / p.h)
growth = Forcing(growth_func, parameters=(τ=1hour, h=4.0))
sinking = AdvectiveForcing(WENO5(), w=-1)

model = NonhydrostaticModel(; grid,
                            tracers = (:b, :P),
                            buoyancy = BuoyancyTracer(),
                            boundary_conditions = (; b=b_bcs),
                            forcing = (; P = (growth, sinking)))

bᵢ(x, y, z) = 1e-5 * z + 1e-9 * rand()
set!(model, b=bᵢ)

Δz = grid.Δzᵃᵃᶜ
Δt = 0.1 * Δz # for a sinking velocity w=1
simulation = Simulation(model; Δt, stop_iteration = 1000)


fig = Figure()
ax = Axis(fig[1, 1])

P = model.tracers.P
hm = heatmap!(ax, interior(P, :, 1, :))

function update!(sim)
    hm.input_args[1][] = interior(P, :, 1, :)
    return nothing
end

simulation.callbacks[:plot] = Callback(update!, IterationInterval(100))

display(fig)

run!(simulation)

