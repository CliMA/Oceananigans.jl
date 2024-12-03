using Oceananigans
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ImplicitFreeSurface
using GLMakie

grid = RectilinearGrid(size=(128, 32), halo=(4, 4), x=(-5, 5), z=(0, 1), topology=(Bounded, Flat, Bounded))

mountain(x) = (x - 3) / 2
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(mountain))

Fu(x, z, t) = sin(t)
free_surface = ImplicitFreeSurface(gravitational_acceleration=10)
model = NonhydrostaticModel(; grid, free_surface, advection=WENO(order=5), forcing=(; u=Fu))

simulation = Simulation(model, Δt=0.1, stop_time=20*2π)
conjure_time_step_wizard!(simulation, cfl=0.7)

progress(sim) = @info string(iteration(sim), ": ", time(sim))
add_callback!(simulation, progress, IterationInterval(100))

ow = JLD2OutputWriter(model, merge(model.velocities, (; η=model.free_surface.η)),
                      filename = "nonhydrostatic_internal_tide.jld2",  
                      schedule = TimeInterval(0.1),
                      overwrite_existing = true)

simulation.output_writers[:jld2] = ow

run!(simulation)

fig = Figure()

axη = Axis(fig[1, 1], xlabel="x", ylabel="Free surface \n displacement")
axw = Axis(fig[2, 1], xlabel="x", ylabel="Surface vertical velocity")
axu = Axis(fig[3, 1], xlabel="x", ylabel="z")
                      
ut = FieldTimeSeries("nonhydrostatic_internal_tide.jld2", "u") 
wt = FieldTimeSeries("nonhydrostatic_internal_tide.jld2", "w") 
ηt = FieldTimeSeries("nonhydrostatic_internal_tide.jld2", "η") 
Nt = length(wt)

slider = Slider(fig[4, 1], range=1:Nt, startvalue=1)
n = slider.value
Nz = size(ut.grid, 3)

u = @lift ut[$n]
η = @lift interior(ηt[$n], :, 1, 1)
w = @lift interior(wt[$n], :, 1, Nz+1)
x = xnodes(wt)

ulim = maximum(abs, ut) * 3/4

lines!(axη, x, η)
lines!(axw, x, w)
heatmap!(axu, u)

ylims!(axη, -0.1, 0.1)
ylims!(axw, -0.01, 0.01)

record(fig, "nonhydrostatic_internal_tide.mp4", 1:Nt) do nn
    @info "Drawing frame $nn of $Nt..."
    n[] = nn
end
