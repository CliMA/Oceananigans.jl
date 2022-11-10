using Oceananigans, Printf, KernelAbstractions, LinearAlgebra
using Oceananigans.Units: minutes, hour, hours, day, days
using Oceananigans.Biogeochemistry: AbstractBiogeochemistry
using Oceananigans.Grids: znode
using Oceananigans.Utils: work_layout
using Oceananigans.Architectures: device
using Oceananigans.TimeSteppers: RungeKutta3TimeStepper, QuasiAdamsBashforth2TimeStepper

import Oceananigans.Biogeochemistry: required_biogeochemical_tracers, update_biogeochemical_state!

struct PositivityPreservingNutrientPhytoplanktonDetritus{FT} <: AbstractBiogeochemistry
    max_growth_rate :: FT
    light_e_folding_distance :: FT
    mortality :: FT
    remineralisation_rate :: FT
    nutrient_limitation_half_saturation :: FT
end

@inline PositivityPreservingNutrientPhytoplanktonDetritus() = PositivityPreservingNutrientPhytoplanktonDetritus(1/day, 5.0, 0.1/day, 0.09/day, 2.5)

required_biogeochemical_tracers(::PositivityPreservingNutrientPhytoplanktonDetritus) = (:N, :P, :D)

@inline nutrient_limitation(N, K) = N/(N+K)

# not going to specify any forcings, could include some external source

@kernel function positivity_preserving_nutrient_phytoplankton_detritus_tendencies!(N, P, D, Δt, bgc)
    i, j, k = @index(Global, NTuple)
    z = znode(Center(), k, grid)

    Nⁿ, Pⁿ, Dⁿ = @inbounds N[i, j, k], P[i, j, k], D[i, j, k]
    
    phytoplankton_growth = bgc.max_growth_rate*exp(z/bgc.light_e_folding_distance)*nutrient_limitation(Nⁿ, bgc.nutrient_limitation_half_saturation)*Pⁿ
    phytoplankton_death = bgc.mortality*Pⁿ
    detritus_remineralisation = bgc.remineralisation_rate*Dⁿ

    # since this has such simple interactions it is probably easier to not use matrices for this but this is more general
    C⃗ⁿ = [Nⁿ, Pⁿ, Dⁿ]
    P⃗ = [0.0                                0.0                             detritus_remineralisation; 
            phytoplankton_growth 0.0                              0.0; 
            0.0                                phytoplankton_death 0.0]
    D⃗ = [Nⁿ + Δt*phytoplankton_growth 0.0                                          0.0;
            0.0                                             Pⁿ + Δt*phytoplankton_death 0.0;
            0.0                                             0.0                                           detritus_remineralisation]

    Nⁿ⁺¹, Pⁿ⁺¹, Dⁿ⁺¹ = (C⃗ⁿ.*C⃗ⁿ)\(D⃗ - Δt.*P⃗)

    @inbounds N[i, j, k], P[i, j, k], D[i, j, k] = Nⁿ⁺¹, Pⁿ⁺¹, Dⁿ⁺¹
end

@inline get_stage_Δt(timestepper::RungeKutta3TimeStepper, val_stage::Val{1}, Δt) = timestepper.γ¹*Δt
@inline get_stage_Δt(timestepper::RungeKutta3TimeStepper, val_stage::Val{2}, Δt) = (timestepper.γ² + timestepper.ζ²) * Δt
@inline get_stage_Δt(timestepper::RungeKutta3TimeStepper, val_stage::Val{3}, Δt) = (timestepper.γ³ + timestepper.ζ³) * Δt
@inline get_stage_Δt(timestepper::QuasiAdamsBashforth2TimeStepper, ::Val, Δt) = Δt

@inline update_biogeochemical_state!(bgc::PositivityPreservingNutrientPhytoplanktonDetritus, model) = update_biogeochemical_state!(bgc, model, Val(model.clock.iteration))
@inline update_biogeochemical_state!(bgc::PositivityPreservingNutrientPhytoplanktonDetritus, model, ::Val{0}) = nothing
@inline function update_biogeochemical_state!(bgc::PositivityPreservingNutrientPhytoplanktonDetritus, model, ::Val)
    Δt = model.auxiliary_fields.Δt[1]
    stage = model.clock.stage
    stage_Δt = get_stage_Δt(model.timestepper, Val(stage), Δt)

    workgroup, worksize = work_layout(model.grid, :xyz)

    tendencies_kernel! = positivity_preserving_nutrient_phytoplankton_detritus_tendencies!(device(model.architecture), workgroup, worksize)
    tendencies_event = tendencies_kernel!(model.tracers.N, model.tracers.P, model.tracers.D, stage_Δt, bgc)
    wait(tendencies_event)
end

grid = RectilinearGrid(size=(64, ), extent=(64, ), topology=(Flat, Flat, Bounded))

Δt = 0.001days

model = NonhydrostaticModel(; grid,
                            timestepper = :RungeKutta3,
                            tracers = (:N, :P, :D),
                            biogeochemistry = PositivityPreservingNutrientPhytoplanktonDetritus(1/day, 5.0, 0.1/day, 0.09/day, 0.01),
                            auxiliary_fields = (Δt = [Δt], ))

set!(model, N = 10.0, P = 0.1, D = 0.0)

simulation = Simulation(model, Δt = Δt, stop_time=20days)#Δt=1day, stop_time=20days)

progress(sim) = @printf("Iteration: %d, time: %s, Δt: %s\n",
                        iteration(sim), prettytime(sim), prettytime(sim.Δt))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

# and a basic `JLD2OutputWriter` that writes velocities and both
# the two-dimensional and horizontally-averaged plankton concentration,


outputs = (N = model.tracers.N, P = model.tracers.P, D = model.tracers.D)

simulation.output_writers[:simple_output] =
    JLD2OutputWriter(model, outputs,
                     schedule = TimeInterval(20minutes),
                     filename = "pos_pres.jld2",
                     overwrite_existing = true)

# !!! info "Using multiple output writers"
#     Because each output writer is associated with a single output `schedule`,
#     it often makes sense to use _different_ output writers for different types of output.
#     For example, smaller outputs that consume less disk space may be written more
#     frequently without threatening the capacity of your hard drive.
#     An arbitrary number of output writers may be added to `simulation.output_writers`.
#
# The simulation is set up. Let there be plankton:

#=run!(simulation)


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

using CairoMakie

fig = Figure(resolution = (1200, 1000))

ax_N = Axis(fig[1, 1]; xlabel = "x (m)", ylabel = "z (m)")
hm_N = heatmap!(ax_N, times, zp, N_timeseries[1, 1, 1:grid.Nz, 1:end]; colormap = :matter)
Colorbar(fig[1, 2], hm_N; label = "Nutrient concentration (μM)", flipaxis = false)

ax_P = Axis(fig[2, 1]; xlabel = "x (m)", ylabel = "z (m)")
hm_P = heatmap!(ax_P, times, zp, P_timeseries[1, 1, 1:grid.Nz, 1:end]; colormap = :matter)
Colorbar(fig[2, 2], hm_P; label = "Plankton concentration (μM)", flipaxis = false)

ax_D = Axis(fig[3, 1]; xlabel = "x (m)", ylabel = "z (m)")
hm_D = heatmap!(ax_D, times, zp, D_timeseries[1, 1, 1:grid.Nz, 1:end]; colormap = :matter)
Colorbar(fig[3, 2], hm_D; label = "Detritus concentration (μM)", flipaxis = false)

save("pos_pres.png", fig)=#