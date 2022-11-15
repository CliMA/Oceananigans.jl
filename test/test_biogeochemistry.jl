using Oceananigans, Printf
using Oceananigans.Units: minutes, hour, hours, day
using Oceananigans.Biogeochemistry: AbstractContinuousFormBiogeochemistry
using Oceananigans.Grids: znode

import Oceananigans.Biogeochemistry:
    required_biogeochemical_tracers
    biogeochemical_drift_velocity,
    biogeochemical_advection_schemes

struct SimplePlanktonGrowthDeath{FT, W, A} <: AbstractContinuousFormBiogeochemistry
     growth_rate :: FT
     light_penetration_depth :: FT
     mortality_rate :: FT
     sinking_velocity :: W     
     advection_scheme :: A
end

function SimplePlanktonGrowthDeath(; growth_rate,
                                     light_penetration_depth,
                                     mortality_rate,
                                     sinking_velocity = 0,
                                     advection_scheme = nothing)

    return SimplePlanktonGrowthDeath(growth_rate,
                                     light_penetration_depth,
                                     mortality_rate,
                                     sinking_velocity,
                                     advection_scheme)
end

######
###### Functions we have to define
######

@inline required_biogeochemical_tracers(::SimplePlanktonGrowthDeath) = (:P,)
@inline biogeochemical_drift_velocity(bgc::SimplePlanktonGrowthDeath, ::Val{P}) = (0.0, 0.0, bgc.w)
@inline biogeochemical_advection_scheme(bgc::SimplePlanktonGrowthDeath, ::Val{P}) = bgc.advection

@inline (bgc::SimplePlanktonGrowthDeath)(::Val{P}, x, y, z, t, P, bgc) = (bgc.μ₀ * exp(z / bgc.λ) - bgc.m) * P

#=
# Note, if we subtypted AbstractBiogeochemistry we would write
@inline function (bgc::SimplePlanktonGrowthDeath)(i, j, k, grid, ::Val{:P}, clock, fields)
    z = znode(Center(), k, grid)
    P = @inbounds fields.P[i, j, k]
    return (bgc.μ₀ * exp(z / bgc.λ) - bgc.m) * P
end
=#

grid = RectilinearGrid(size = (64, 64),
                       extent = (64, 64),
                       halo = (3, 3),
                       topology = (Periodic, Flat, Bounded))

buoyancy_flux_bc = FluxBoundaryCondition(1e-8)

N² = 1e-4 # s⁻²
buoyancy_gradient_bc = GradientBoundaryCondition(N²)
buoyancy_bcs = FieldBoundaryConditions(top = buoyancy_flux_bc, bottom = buoyancy_gradient_bc)

biogeochemistry = SimplePlanktonGrowthDeath(growth_rate = 1/day,
                                            light_penetration_depth = 5.0,
                                            mortality_rate = 0.1/day)

model = NonhydrostaticModel(; grid, biogeochemistry,
                            advection = WENO(; grid),
                            timestepper = :RungeKutta3,
                            closure = ScalarDiffusivity(ν=1e-4, κ=1e-4),
                            coriolis = FPlane(f=1e-4),
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            boundary_conditions = (; b=buoyancy_bcs))

mixed_layer_depth = 32 # m
stratification(z) = z < -mixed_layer_depth ? N² * z : - N² * mixed_layer_depth
noise(z) = 1e-4 * N² * grid.Lz * randn() * exp(z / 4)
initial_buoyancy(x, y, z) = stratification(z) + noise(z)

set!(model, b=initial_buoyancy, P = 1.0)

simulation = Simulation(model, Δt=2minutes, stop_time=24hours)

wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=2minutes)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

progress(sim) = @printf("Iteration: %d, time: %s, Δt: %s\n",
                        iteration(sim), prettytime(sim), prettytime(sim.Δt))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

outputs = (w = model.velocities.w,
           P = model.tracers.P,
           avg_P = Average(model.tracers.P, dims=(1, 2)))

simulation.output_writers[:simple_output] =
    JLD2OutputWriter(model, outputs,
                     schedule = TimeInterval(20minutes),
                     filename = "convecting_NPD.jld2",
                     overwrite_existing = true)

run!(simulation)

#=
# Notice how the time-step is reduced at early times, when turbulence is strong,
# and increases again towards the end of the simulation when turbulence fades.

# ## Visualizing the solution
#
# We'd like to a make a plankton movie. First we load the output file
# and build a time-series of the buoyancy flux,

filepath = simulation.output_writers[:simple_output].filepath

w_timeseries = FieldTimeSeries(filepath, "w")
P_timeseries = FieldTimeSeries(filepath, "P")
avg_P_timeseries = FieldTimeSeries(filepath, "avg_P")

times = w_timeseries.times
buoyancy_flux_time_series = [buoyancy_flux(0, 0, t, buoyancy_flux_parameters) for t in times]
nothing # hide

# and then we construct the ``x, z`` grid,

xw, yw, zw = nodes(w_timeseries)
xp, yp, zp = nodes(P_timeseries)
nothing # hide

# Finally, we animate plankton mixing and blooming,

using CairoMakie

@info "Making a movie about plankton..."

n = Observable(1)

title = @lift @sprintf("t = %s", prettytime(times[$n]))

wₙ = @lift interior(w_timeseries[$n], :, 1, :)
Pₙ = @lift interior(P_timeseries[$n], :, 1, :)
avg_Pₙ = @lift interior(avg_P_timeseries[$n], 1, 1, :)

w_lim = maximum(abs, interior(w_timeseries))
w_lims = (-w_lim, w_lim)

P_lims = (0.95, 1.1)

fig = Figure(resolution = (1200, 1000))

ax_w = Axis(fig[2, 2]; xlabel = "x (m)", ylabel = "z (m)", aspect = 1)
ax_P = Axis(fig[3, 2]; xlabel = "x (m)", ylabel = "z (m)", aspect = 1)
ax_b = Axis(fig[2, 3]; xlabel = "Time (hours)", ylabel = "Buoyancy flux (m² s⁻³)", yaxisposition = :right)

ax_avg_P = Axis(fig[3, 3]; xlabel = "Plankton concentration (μM)", ylabel = "z (m)", yaxisposition = :right)
xlims!(ax_avg_P, 0.85, 1.3)

fig[1, 1:3] = Label(fig, title, tellwidth=false)

hm_w = heatmap!(ax_w, xw, zw, wₙ; colormap = :balance, colorrange = w_lims)
Colorbar(fig[2, 1], hm_w; label = "Vertical velocity (m s⁻¹)", flipaxis = false)

hm_P = heatmap!(ax_P, xp, zp, Pₙ; colormap = :matter, colorrange = P_lims)
Colorbar(fig[3, 1], hm_P; label = "Plankton 'concentration'", flipaxis = false)

lines!(ax_b, times ./ hour, buoyancy_flux_time_series; linewidth = 1, color = :black, alpha = 0.4)

b_flux_point = @lift Point2(times[$n] / hour, buoyancy_flux_time_series[$n])
scatter!(ax_b, b_flux_point; marker = :circle, markersize = 16, color = :black)
lines!(ax_avg_P, avg_Pₙ, zp)

# And, finally, we record a movie.

frames = 1:length(times)

@info "Making an animation of convecting plankton..."

record(fig, "convecting_plankton.mp4", frames, framerate=8) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    n[] = i
end
nothing #hide

# ![](convecting_plankton.mp4)
# =#
