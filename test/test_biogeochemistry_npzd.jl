using Oceananigans, Printf, LinearAlgebra
using Oceananigans.Units: minutes, hour, hours, day, days, years
using Oceananigans.Biogeochemistry: AbstractBiogeochemistry
using Oceananigans.Grids: znode

import Oceananigans.Biogeochemistry: required_biogeochemical_tracers, update_biogeochemical_state!

struct NutrientPhytoplanktonDetritus{FT} <: AbstractBiogeochemistry
    max_growth_rate :: FT
    light_e_folding_distance :: FT
    mortality :: FT
    remineralisation_rate :: FT
    nutrient_limitation_half_saturation :: FT
end

@inline NutrientPhytoplanktonDetritus() = NutrientPhytoplanktonDetritus(1/day, 5.0, 0.1/day, 0.09/day, 2.5)

required_biogeochemical_tracers(::NutrientPhytoplanktonDetritus) = (:N, :P, :D)

@inline nutrient_limitation(N, K) = N/(N+K)

@inline function (bgc::NutrientPhytoplanktonDetritus)(i, j, k, grid, ::Val{:N}, clock, fields)
    z = znode(Center(), k, grid)

    Nⁿ, Pⁿ, Dⁿ = @inbounds fields.N[i, j, k], fields.P[i, j, k], fields.D[i, j, k]
    
    phytoplankton_growth = bgc.max_growth_rate*exp(z/bgc.light_e_folding_distance)*nutrient_limitation(Nⁿ, bgc.nutrient_limitation_half_saturation)*Pⁿ
    detritus_remineralisation = bgc.remineralisation_rate*Dⁿ

    return detritus_remineralisation - phytoplankton_growth
end

@inline function (bgc::NutrientPhytoplanktonDetritus)(i, j, k, grid, ::Val{:P}, clock, fields)
    z = znode(Center(), k, grid)

    Nⁿ, Pⁿ = @inbounds fields.N[i, j, k], fields.P[i, j, k]

    phytoplankton_growth = bgc.max_growth_rate*exp(z/bgc.light_e_folding_distance)*nutrient_limitation(Nⁿ, bgc.nutrient_limitation_half_saturation)*Pⁿ
    phytoplankton_death = bgc.mortality*Pⁿ

    return phytoplankton_growth - phytoplankton_death
end


@inline function (bgc::NutrientPhytoplanktonDetritus)(i, j, k, grid, ::Val{:D}, clock, fields)
    Pⁿ, Dⁿ = @inbounds fields.P[i, j, k], fields.D[i, j, k]

    phytoplankton_death = bgc.mortality*Pⁿ
    detritus_remineralisation = bgc.remineralisation_rate*Dⁿ

    return phytoplankton_death - detritus_remineralisation
end

required_biogeochemical_tracers(::NutrientPhytoplanktonDetritus) = (:N, :P, :D)
grid = RectilinearGrid(size=(20, 100), extent=(10, 20), topology=(Periodic, Flat, Bounded)) 

buoyancy_flux(x, y, t, params) = params.initial_buoyancy_flux * max(0.0, sin(t*π/(2years))) + params.initial_buoyancy_flux/10

buoyancy_flux_parameters = (initial_buoyancy_flux = 1e-8, ) # m² s⁻³)

buoyancy_flux_bc = FluxBoundaryCondition(buoyancy_flux, parameters = buoyancy_flux_parameters)

N² = 1e-4 # s⁻²

buoyancy_gradient_bc = GradientBoundaryCondition(N²)
buoyancy_bcs = FieldBoundaryConditions(top = buoyancy_flux_bc, bottom = buoyancy_gradient_bc)

model = NonhydrostaticModel(; grid,
                            timestepper = :RungeKutta3,
                            closure = ScalarDiffusivity(ν=1e-4, κ=1e-4),
                            coriolis = FPlane(f=1e-4),
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            boundary_conditions = (; b=buoyancy_bcs),
                            biogeochemistry = NutrientPhytoplanktonDetritus())


mixed_layer_depth = 32 # m

stratification(z) = z < -mixed_layer_depth ? N² * z : - N² * mixed_layer_depth
noise(z) = 1e-4 * N² * grid.Lz * randn() * exp(z / 4)
initial_buoyancy(x, y, z) = stratification(z) + noise(z)

set!(model, b=initial_buoyancy, N = 10.0, P = 1.0, D = 0.0)
 
simulation = Simulation(model, Δt = 1minutes, stop_time=3years)
tsw = TimeStepWizard(cfl=0.5, diffusive_cfl=0.5, max_change=1.2)
simulation.callbacks[:timestep] = Callback(tsw, IterationInterval(1))

progress(sim) = @printf("Iteration: %d, time: %s, Δt: %s\n",
                        iteration(sim), prettytime(sim), prettytime(sim.Δt))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

# and a basic `JLD2OutputWriter` that writes velocities and both
# the two-dimensional and horizontally-averaged plankton concentration,

outputs = (N = model.tracers.N, P = model.tracers.P, D = model.tracers.D, w=model.tracers.w)

simulation.output_writers[:simple_output] =
    JLD2OutputWriter(model, outputs,
                     schedule = TimeInterval(1days),
                     filename = "npzd.jld2",
                     overwrite_existing = true)

run!(simulation)


# Notice how the time-step is reduced at early times, when turbulence is strong,
# and increases again towards the end of the simulation when turbulence fades.

# ## Visualizing the solution
#
# We'd like to a make a plankton movie. First we load the output file
# and build a time-series of the buoyancy flux,

filepath = simulation.output_writers[:simple_output].filepath
w_timeseries = FieldTimeSeries(filepath, "w")

N_timeseries = FieldTimeSeries(filepath, "N")
P_timeseries = FieldTimeSeries(filepath, "P")
D_timeseries = FieldTimeSeries(filepath, "D")


times = P_timeseries.times
xp, yp, zp = nodes(P_timeseries)
xw, yw, zw = nodes(w_timeseries)
# Finally, we animate plankton mixing and blooming,

using GLMakie

@info "Making a movie about plankton..."

n = Observable(1)

title = @lift @sprintf("t = %s", prettytime(times[$n]))
wₙ = @lift interior(w_timeseries[$n], :, 1, :)
Nₙ = @lift interior(N_timeseries[$n], :, 1, :)
Pₙ = @lift interior(P_timeseries[$n], :, 1, :)
Dₙ = @lift interior(D_timeseries[$n], :, 1, :)

w_lims = (-maximum(abs, w_timeseries), maximum(abs, w_timeseries))

N_lims = (minimum(N_timeseries), maximum(N_timeseries))
P_lims = (minimum(P_timeseries), maximum(P_timeseries))
D_lims = (minimum(D_timeseries), maximum(D_timeseries))

fig = Figure(resolution = (1200, 1000))

ax_w = Axis(fig[2, 2]; xlabel = "x (m)", ylabel = "z (m)", aspect = 1)
ax_N = Axis(fig[2, 4]; xlabel = "x (m)", ylabel = "z (m)", aspect = 1)
ax_P = Axis(fig[3, 2]; xlabel = "x (m)", ylabel = "z (m)", aspect = 1)
ax_D = Axis(fig[3, 4]; xlabel = "x (m)", ylabel = "z (m)", aspect = 1)

fig[1, 1:2] = Label(fig, title, tellwidth=false)

hm_w = heatmap!(ax_w, xw, zw, wₙ; colormap = :matter, colorrange = w_lims)
Colorbar(fig[2, 1], hm_w; label = "Vertical velocity (m/s)", flipaxis = false)

hm_N = heatmap!(ax_N, xp, zp, Nₙ; colormap = :matter, colorrange = N_lims)
Colorbar(fig[2, 3], hm_N; label = "Nutrient concentration (mmol N/m³)", flipaxis = false)

hm_P = heatmap!(ax_P, xp, zp, Pₙ; colormap = :matter, colorrange = P_lims)
Colorbar(fig[3, 1], hm_P; label = "Phytoplankton concentration (mmol N/m³)", flipaxis = false)

hm_D = heatmap!(ax_D, xp, zp, Dₙ; colormap = :matter, colorrange = D_lims)
Colorbar(fig[3, 3], hm_D; label = "Detritus concentration (mmol N/m³)", flipaxis = false)

# And, finally, we record a movie.

frames = 1:length(times)

@info "Making an animation of convecting plankton..."

record(fig, "npd_example.mp4", frames, framerate=8) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    n[] = i
end