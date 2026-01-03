using Oceananigans
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ImplicitFreeSurface
using GLMakie
using Printf

Nz = 128
Lz = 10
Δz = Lz / Nz
z = 0:Δz:Lz
g = 10
grid = RectilinearGrid(size=(128, Nz); halo=(4, 4), x=(-10, 10), z, topology=(Bounded, Flat, Bounded))
free_surface = ImplicitFreeSurface(gravitational_acceleration=g)
model = NonhydrostaticModel(; grid, free_surface)

ηᵢ(x, z) = 0.1 * exp(-x^2 / 2)
set!(model, η=ηᵢ)

fig = Figure()
ax = Axis(fig[1, 1])
heatmap!(ax, interior(model.pressures.pNHS, :, 1, :))
fig

# set!(model.free_surface.displacement, ηᵢ)

#=
Δx = 20 / grid.Nx
c = sqrt(g)
Δt = 0.1 * Δx / c
simulation = Simulation(model; Δt, stop_iteration=10) #stop_time=5/c)

ηt = []
function progress(sim) 
    @info @sprintf("Time: %s, iteration: %d", prettytime(sim), iteration(sim))
    push!(ηt, deepcopy(interior(model.free_surface.displacement, :, 1, 1)))
    return nothing
end

add_callback!(simulation, progress, IterationInterval(1))

run!(simulation)
=#

#=
fig = Figure()
ax = Axis(fig[1, 1], xlabel="x", ylabel="η")
slider = Slider(fig[2, 1], range=1:length(ηt), startvalue=1)
n = slider.value
ηn = @lift ηt[$n]
lines!(ax, interior(model.free_surface.displacement, :, 1, 1))
fig
=#

# mountain(x) = (x - 3) / 2
# grid = ImmersedBoundaryGrid(grid, GridFittedBottom(mountain))
# Fu(x, z, t) = sin(t)
# model = NonhydrostaticModel(; grid, free_surface, advection=WENO(order=5), forcing=(; u=Fu))
