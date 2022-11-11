using Oceananigans, Printf, KernelAbstractions, LinearAlgebra
using Oceananigans.Units: minutes, hour, hours, day, days, years
using Oceananigans.Biogeochemistry: AbstractBiogeochemistry
using Oceananigans.Grids: znode
using Oceananigans.Utils: work_layout
using Oceananigans.Architectures: device
using Oceananigans.TimeSteppers: RungeKutta3TimeStepper, QuasiAdamsBashforth2TimeStepper

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

grid = RectilinearGrid(size=(1, 1, 50), extent=(20, 20, 30)) 

Δt = .2days

model = NonhydrostaticModel(; grid,
                            tracers = (:N, :P, :D),
                            biogeochemistry = NutrientPhytoplanktonDetritus(),
                            auxiliary_fields = (Δt = [Δt], ))

set!(model, N = 10.0, P = 1.0, D = 0.0)

simulation = Simulation(model, Δt = Δt, stop_time=3years)

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