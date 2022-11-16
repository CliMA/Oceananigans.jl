using Oceananigans, Printf, KernelAbstractions
using Oceananigans.Units: minutes, hour, hours, day, days
using Oceananigans.Biogeochemistry: AbstractContinuousFormBiogeochemistry
using Oceananigans.Grids: znode
using Oceananigans.Architectures: device
using Oceananigans.Utils: launch!

import Oceananigans.Biogeochemistry:
    required_biogeochemical_tracers,
    required_biogeochemical_auxiliary_fields,
    biogeochemical_drift_velocity,
    biogeochemical_advection_scheme, 
    update_biogeochemical_state!

struct SimplePlanktonGrowthDeath{FT, P, W, A} <: AbstractContinuousFormBiogeochemistry
     growth_rate :: FT
     light_limit :: FT
     mortality_rate :: FT

     water_light_attenuation_coefficient :: FT
     phytoplankton_light_attenuation_coefficient :: FT
     phytoplankton_light_attenuation_exponent :: FT
     surface_PAR :: P

     sinking_velocity :: W     
     advection_scheme :: A
end

function SimplePlanktonGrowthDeath(; growth_rate,
                                     light_limit,
                                     mortality_rate,
                                     water_light_attenuation_coefficient = 0.12,
                                     phytoplankton_light_attenuation_coefficient = 0.06,
                                     phytoplankton_light_attenuation_exponent = 0.6,
                                     surface_PAR = 100.0,
                                     sinking_velocity = 0,
                                     advection_scheme = nothing)

    return SimplePlanktonGrowthDeath(growth_rate,
                                     light_limit,
                                     mortality_rate,
                                     water_light_attenuation_coefficient,
                                     phytoplankton_light_attenuation_coefficient,
                                     phytoplankton_light_attenuation_exponent,
                                     isa(surface_PAR, Number) ? t -> surface_PAR : surface_PAR,
                                     sinking_velocity,
                                     advection_scheme)
end

######
###### Functions we have to define
######

@inline required_biogeochemical_tracers(::SimplePlanktonGrowthDeath) = (:P, )
@inline required_biogeochemical_auxiliary_fields(::SimplePlanktonGrowthDeath) = (:PAR, )
@inline biogeochemical_drift_velocity(bgc::SimplePlanktonGrowthDeath, ::Val{:P}) = (0.0, 0.0, bgc.w)
@inline biogeochemical_advection_scheme(bgc::SimplePlanktonGrowthDeath, ::Val{:P}) = bgc.advection

@inline function (bgc::SimplePlanktonGrowthDeath)(::Val{:P}, x, y, z, t, P, PAR)
    μ₀ = bgc.growth_rate
    k = bgc.light_limit
    m = bgc.mortality_rate

    (μ₀ * (1 - exp(-PAR/k)) - m) * P
end

@kernel function update_PAR!(PAR, grid, P, t, bgc) 
    i, j = @index(Global, NTuple) 

    surface_PAR = bgc.surface_PAR(t)

    z = grid.zᵃᵃᶜ[grid.Nz]

    ∫chl = - P[i, j, grid.Nz]^bgc.phytoplankton_light_attenuation_exponent*z
    PAR[i, j, grid.Nz] =  surface_PAR*exp(bgc.water_light_attenuation_coefficient * z - bgc.phytoplankton_light_attenuation_coefficient * ∫chl)
    for k=grid.Nz-1:-1:1
        z = grid.zᵃᵃᶜ[k]
        dz = grid.zᵃᵃᶜ[k+1] - z 

        ∫chl += P[i, j, grid.Nz]^bgc.phytoplankton_light_attenuation_exponent*dz

        PAR[i, j, k] =  PAR[i, j, k+1]*exp(- bgc.water_light_attenuation_coefficient * dz - bgc.phytoplankton_light_attenuation_coefficient * ∫chl)
    end
end 

@inline function update_biogeochemical_state!(bgc::SimplePlanktonGrowthDeath, model)
    # Assuming light is attenuated like PAR*exp(-∫(kʷ + Chl*kᶜʰˡ)dz)
    par_calculation =  launch!(model.architecture, model.grid, :xy, update_PAR!,
                               model.auxiliary_fields.PAR, model.grid, model.tracers.P, model.clock.time, bgc,
                               dependencies = Event(device(model.architecture)))
    wait(device(model.architecture), par_calculation)
end

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
                                            light_limit = 3.5,
                                            mortality_rate = 0.1/day,
                                            surface_PAR = t -> 100*max(0.0, sin(t*π/(12hours))))

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

simulation = Simulation(model, Δt=2minutes, stop_time=5days)

wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=2minutes)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

progress(sim) = @printf("Iteration: %d, time: %s, Δt: %s\n",
                        iteration(sim), prettytime(sim), prettytime(sim.Δt))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

outputs = (w = model.velocities.w,
           P = model.tracers.P,
           PAR = model.auxiliary_fields.PAR,
           avg_P = Average(model.tracers.P, dims=(1, 2)))

simulation.output_writers[:simple_output] =
    JLD2OutputWriter(model, outputs,
                     schedule = TimeInterval(20minutes),
                     filename = "convecting_NPD.jld2",
                     overwrite_existing = true)

run!(simulation)
#=
# Plot to sanity check

filepath = simulation.output_writers[:simple_output].filepath
using CairoMakie

w_timeseries = FieldTimeSeries(filepath, "w")
P_timeseries = FieldTimeSeries(filepath, "P")
PAR_timeseries = FieldTimeSeries(filepath, "PAR")

times = w_timeseries.times

xw, yw, zw = nodes(w_timeseries)
xp, yp, zp = nodes(P_timeseries)


@info "Making a movie about plankton..."

n = Observable(1)

title = @lift @sprintf("t = %s", prettytime(times[$n]))

wₙ = @lift interior(w_timeseries[$n], :, 1, :)
Pₙ = @lift interior(P_timeseries[$n], :, 1, :)
PARₙ = @lift interior(PAR_timeseries[$n], :, 1, :)

w_lim = maximum(abs, interior(w_timeseries))
w_lims = (-w_lim, w_lim)

P_lims = (minimum(P_timeseries), maximum(P_timeseries))
PAR_lims = (minimum(PAR_timeseries), maximum(PAR_timeseries))

fig = Figure(resolution = (1200, 1000))

ax_w = Axis(fig[2, 2]; xlabel = "x (m)", ylabel = "z (m)", aspect = 1)
ax_P = Axis(fig[3, 2]; xlabel = "x (m)", ylabel = "z (m)", aspect = 1)
ax_PAR = Axis(fig[2, 4]; xlabel = "x (m)", ylabel = "z (m)", aspect = 1)

fig[1, 1:3] = Label(fig, title, tellwidth=false)

hm_w = heatmap!(ax_w, xw, zw, wₙ; colormap = :balance, colorrange = w_lims)
Colorbar(fig[2, 1], hm_w; label = "Vertical velocity (m s⁻¹)", flipaxis = false)

hm_P = heatmap!(ax_P, xp, zp, Pₙ; colormap = :matter, colorrange = P_lims)
Colorbar(fig[3, 1], hm_P; label = "Plankton 'concentration'", flipaxis = false)

hm_PAR = heatmap!(ax_PAR, xp, zp, PARₙ; colormap = :matter, colorrange = PAR_lims)
Colorbar(fig[2, 3], hm_PAR; label = "Light availability (W/m²)'", flipaxis = false)

# And, finally, we record a movie.

frames = 1:length(times)

@info "Making an animation of convecting plankton..."

record(fig, "convecting_plankton.mp4", frames, framerate=8) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    n[] = i
end
=#
#####
##### Example using SomethingBiogeochemistry
#####

