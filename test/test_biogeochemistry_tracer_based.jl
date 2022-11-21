using Oceananigans, Printf, KernelAbstractions, LinearAlgebra
using Oceananigans.Units: minutes, hour, hours, day, days, years
using Oceananigans.Biogeochemistry: BiogeochemicalModel, BiogeochemicalForcing
using Oceananigans.Grids: znode
using Oceananigans.Utils: work_layout
using Oceananigans.Architectures: device
using Oceananigans.TimeSteppers: RungeKutta3TimeStepper, QuasiAdamsBashforth2TimeStepper

import Oceananigans.Biogeochemistry: required_biogeochemical_tracers, update_biogeochemical_state!

parameters = (max_growth_rate = 1/day, light_e_folding_distance = 5.0, mortality = 0.1/day, remineralisation_rate = 0.09/day, nutrient_limitation_half_saturation = 2.5)

@inline nutrient_limitation(N, K) = N/(N+K)

function nutrient_reaction(x, y, z, t, N, P, D, params)
    phytoplankton_growth = params.max_growth_rate*exp(z/params.light_e_folding_distance)*nutrient_limitation(N, params.nutrient_limitation_half_saturation)*P
    detritus_remineralisation = params.remineralisation_rate*D

    return detritus_remineralisation - phytoplankton_growth
end

function phytoplankton_reaction(x, y, z, t, N, P, params)
    phytoplankton_growth = params.max_growth_rate*exp(z/params.light_e_folding_distance)*nutrient_limitation(N, params.nutrient_limitation_half_saturation)*P
    phytoplankton_death = params.mortality*P

    return phytoplankton_growth - phytoplankton_death
end

function detritus_reaction(x, y, z, t, P, D, params)
    phytoplankton_death = params.mortality*P
    detritus_remineralisation = params.remineralisation_rate*D

    return phytoplankton_death - detritus_remineralisation
end

nutrient_forcing = BiogeochemicalForcing(nutrient_reaction, field_dependencies=(:N, :P, :D), parameters=parameters)
phytoplankton_forcing = BiogeochemicalForcing(phytoplankton_reaction, field_dependencies=(:N, :P), parameters=parameters)
detritus_forcing = BiogeochemicalForcing(detritus_reaction, field_dependencies=(:P, :D), parameters=parameters)

NutrientPhytoplanktonDetritus = BiogeochemicalModel((:N, :P, :D), (N=nutrient_forcing, P=phytoplankton_forcing, D=detritus_forcing))
#NutrientPhytoplanktonDetritusSinking = BiogeochemicalModel((:N, :P, :D), (N=nutrient_forcing, P=phytoplankton_forcing, D=detritus_forcing); sinking_velocities=(D = 200/day, ))

grid = RectilinearGrid(size=(1, 1, 50), extent=(20, 20, 30)) 

Δt = .1days

model = NonhydrostaticModel(; grid,
                            tracers = (:N, :P, :D),
                            biogeochemistry = NutrientPhytoplanktonDetritus,
                            auxiliary_fields = (Δt = [Δt], ))

set!(model, N = 10.0, P = 1.0, D = 0.0)

simulation = Simulation(model, Δt = Δt, stop_time=20days)

progress(sim) = @printf("Iteration: %d, time: %s, Δt: %s\n",
                        iteration(sim), prettytime(sim), prettytime(sim.Δt))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

# and a basic `JLD2OutputWriter` that writes velocities and both
# the two-dimensional and horizontally-averaged plankton concentration,

outputs = (N = model.tracers.N, P = model.tracers.P, D = model.tracers.D)

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
N_timeseries = FieldTimeSeries(filepath, "N")
P_timeseries = FieldTimeSeries(filepath, "P")
D_timeseries = FieldTimeSeries(filepath, "D")


times = P_timeseries.times
xp, yp, zp = nodes(P_timeseries)
# Finally, we animate plankton mixing and blooming,

using GLMakie

fig = Figure(resolution = (1200, 1000))

ax_N = Axis(fig[1, 1]; xlabel = "x (m)", ylabel = "z (m)")
hm_N = heatmap!(ax_N, times, zp, N_timeseries[1, 1, 1:grid.Nz, 1:end]'; colormap = :matter)
Colorbar(fig[1, 2], hm_N; label = "Nutrient concentration (μM)", flipaxis = false)

ax_P = Axis(fig[2, 1]; xlabel = "x (m)", ylabel = "z (m)")
hm_P = heatmap!(ax_P, times, zp, P_timeseries[1, 1, 1:grid.Nz, 1:end]'; colormap = :matter)
Colorbar(fig[2, 2], hm_P; label = "Plankton concentration (μM)", flipaxis = false)

ax_D = Axis(fig[3, 1]; xlabel = "x (m)", ylabel = "z (m)")
hm_D = heatmap!(ax_D, times, zp, D_timeseries[1, 1, 1:grid.Nz, 1:end]'; colormap = :matter)
Colorbar(fig[3, 2], hm_D; label = "Detritus concentration (μM)", flipaxis = false)