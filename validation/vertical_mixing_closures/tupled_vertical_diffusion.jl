using GLMakie
using Oceananigans
using Oceananigans.Units

arch = CPU()
grid = RectilinearGrid(arch, size=128, z=(-5, 5), topology=(Flat, Flat, Bounded))

single_closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), κ=1)

tuple_closure = (VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), κ=1/2), 
                 VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), κ=1/2))

closure = tuple_closure

model = HydrostaticFreeSurfaceModel(; grid, closure, tracers=:c, buoyancy=nothing, coriolis=nothing)
set!(model, c = (x, y, z) -> exp(-z^2))
simulation = Simulation(model, Δt=1e-4, stop_iteration=100)

c_snapshots = []
c = model.tracers.c
getc(sim) = push!(c_snapshots, Array(interior(c, 1, 1, :)))
simulation.callbacks[:c] = Callback(getc, IterationInterval(10))

run!(simulation)

fig = Figure()
ax = Axis(fig[1, 1])
Nt = length(c_snapshots)
slider = Slider(fig[2, 1], range=1:Nt, startvalue=1)
n = slider.value
z = znodes(c)
c = @lift c_snapshots[$n]
lines!(ax, c, z)
display(fig)
