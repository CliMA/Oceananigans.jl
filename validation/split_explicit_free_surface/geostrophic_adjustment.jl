using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: SplitExplicitFreeSurface

grid = RegularRectilinearGrid(size = (128, 1),
                              x = (0, 1000kilometers), z = (-400meters, 0),
                              topology = (Bounded, Flat, Bounded))

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    coriolis = FPlane(f=1e-4),
                                    free_surface = SplitExplicitFreeSurface())

gaussian(x, L) = exp(-x^2 / 2L^2)

U = 0.1 # geostrophic velocity
L = grid.Lx / 40 # Gaussian width
x₀ = grid.Lx / 4 # Gaussian center

vᵍ(x, y, z) = - U * (x - x₀) / L * gaussian(x - x₀, L)

g = model.free_surface.parameters.g
η₀ = model.coriolis.f * U * L / g # geostrohpic free surface amplitude
ηᵍ(x) = η₀ * gaussian(x - x₀, L)
ηⁱ(x, y) = 2 * ηᵍ(x)
ηⁱ(x, y, z) = ηⁱ(x, y)

set!(model, v=vᵍ)

using Oceananigans.Fields: FunctionField

model.free_surface.state.η .= FunctionField{Center, Center, Nothing}(ηⁱ, grid)

gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed
wave_propagation_time_scale = model.grid.Δx / gravity_wave_speed

simulation = Simulation(model, Δt = 0.1wave_propagation_time_scale, stop_iteration = 1000)

output_fields = merge(model.velocities, (η=model.free_surface.η,))

simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                      schedule = IterationInterval(10),
                                                      prefix = "geostrophic_adjustment",
                                                      field_slicer = nothing,
                                                      force = true)

run!(simulation)

#=
# ## Visualizing the results

using Oceananigans.OutputReaders: FieldTimeSeries
using Plots, Printf

u_timeseries = FieldTimeSeries("geostrophic_adjustment.jld2", "u")
v_timeseries = FieldTimeSeries("geostrophic_adjustment.jld2", "v")
η_timeseries = FieldTimeSeries("geostrophic_adjustment.jld2", "η")

xη = xw = xv = xnodes(v_timeseries)
xu = xnodes(u_timeseries)

t = u_timeseries.times

anim = @animate for i = 1:length(t)

    u = interior(u_timeseries[i])[:, 1, 1]
    v = interior(v_timeseries[i])[:, 1, 1]
    η = interior(η_timeseries[i])[:, 1, 1]

    titlestr = @sprintf("Geostrophic adjustment at t = %.1f hours", t[i] / hours)

    u_plot = plot(xu / kilometers, u, linewidth = 2,
                  label = "", xlabel = "x (km)", ylabel = "u (m s⁻¹)", ylims = (-2e-3, 2e-3))

    v_plot = plot(xv / kilometers, v, linewidth = 2, title = titlestr,
                  label = "", xlabel = "x (km)", ylabel = "v (m s⁻¹)", ylims = (-U, U))

    η_plot = plot(xη / kilometers, η, linewidth = 2,
                  label = "", xlabel = "x (km)", ylabel = "η (m)", ylims = (-η₀/10, 2η₀))

    plot(v_plot, u_plot, η_plot, layout = (3, 1), size = (800, 600))
end

mp4(anim, "geostrophic_adjustment.mp4", fps = 15) # hide
=#