using Oceananigans
using Oceananigans.TurbulenceClosures: DirectionallyAveragedCoefficient

N = 16
arch = CPU()
grid = RectilinearGrid(arch,
                       size=(N, N, N),
                       extent=(2π, 2π, 2π),
                       topology=(Periodic, Periodic, Periodic))

#closure = SmagorinskyLilly(coefficient=0.16)
closure = SmagorinskyLilly(coefficient=DirectionallyAveragedCoefficient())
model = NonhydrostaticModel(; grid, closure)

ϵ(x, y, z) = 2rand() - 1
set!(model, u=ϵ, v=ϵ, w=ϵ)

simulation = Simulation(model, Δt=0.2, stop_iteration=10)
wizard = TimeStepWizard(cfl=0.7, max_change=1.1, max_Δt=0.5)
add_callback!(simulation, wizard, IterationInterval(10))

run!(simulation)

simulation.stop_iteration += 100
@time run!(simulation)

using GLMakie
u, v, w = model.velocities
k = round(Int, size(grid, 3)/2)
heatmap(view(u, :, :, k))
