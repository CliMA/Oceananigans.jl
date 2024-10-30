using Oceananigans
using Oceananigans.TurbulenceClosures: LagrangianAveraging

N = 16
arch = CPU()
grid = RectilinearGrid(arch,
                       size=(N, N, N),
                       extent=(N, N, N),
                       topology=(Periodic, Periodic, Periodic))

advection = Centered(order=2)
#closure = nothing
closure = Smagorinsky(coefficient=0.16)
closure = Smagorinsky(coefficient=LillyCoefficient())
#coefficient = DynamicCoefficient(averaging=(1, 2))
coefficient = DynamicCoefficient(averaging=LagrangianAveraging())
closure = Smagorinsky(; coefficient)
@time model = NonhydrostaticModel(; grid, closure, advection)

ϵ(x, y, z) = 2rand() - 1
set!(model, u=ϵ, v=ϵ, w=ϵ)

simulation = Simulation(model, Δt=0.1, stop_iteration=100)
wizard = TimeStepWizard(cfl=0.7, max_change=1.1, max_Δt=0.5)
add_callback!(simulation, wizard, IterationInterval(10))

@time time_step!(model, 1)
pause
run!(simulation)

simulation.stop_iteration += 100
@time run!(simulation)

using GLMakie
u, v, w = model.velocities
k = round(Int, size(grid, 3)/2)
heatmap(view(u, :, :, k))

