using Oceananigans, Printf, KernelAbstractions
using Oceananigans.Units: minutes, hour, hours, day, days
using Oceananigans.Biogeochemistry: AbstractContinuousFormBiogeochemistry, BasicBiogeochemistry, all_fields_present
using Oceananigans.Grids: znode
using Oceananigans.Forcings: maybe_constant_field
using Oceananigans.Architectures: device, architecture
using Oceananigans.Utils: launch!
using Oceananigans.Advection: CenteredSecondOrder
using Oceananigans.Fields: Field, TracerFields, CenterField

import Oceananigans.Biogeochemistry:
       required_biogeochemical_tracers,
       required_biogeochemical_auxiliary_fields,
       biogeochemical_drift_velocity,
       biogeochemical_advection_scheme,
       biogeochemical_auxiliary_fieilds,
       update_biogeochemical_state!

struct SimplePlanktonGrowthDeath{FT, W, A, LA} <: AbstractContinuousFormBiogeochemistry
    growth_rate :: FT
    light_limit :: FT
    mortality_rate :: FT

    sinking_velocity :: W     
    advection_scheme :: A

    light_attenuation_model :: LA

    function SimplePlanktonGrowthDeath(; grid,
                                         growth_rate::FT = 1/day,
                                         light_limit::FT = 3.5,
                                         mortality_rate::FT = 0.3/day,
                                         sinking_velocity::FT = 200/day,
                                         advection_scheme::A = sinking_velocity == 0 ? nothing : CenteredSecondOrder(),
                                         light_attenuation_model::LA = SimplePhyotosyntheticallyActiveRadiation(; grid)) where {FT, A, LA}

        u, v, w = maybe_constant_field.((0.0, 0.0, - sinking_velocity))
        sinking_velocity = (; u, v, w)
        W = typeof(sinking_velocity)

        return new{FT, W, A, LA}(growth_rate,
                                 light_limit,
                                 mortality_rate,
                                 sinking_velocity,
                                 advection_scheme,
                                 light_attenuation_model)
    end 
end


######
###### Functions we have to define to setup the biogeochemical mdoel
######

@inline required_biogeochemical_tracers(::SimplePlanktonGrowthDeath) = (:P, )
@inline required_biogeochemical_auxiliary_fields(::SimplePlanktonGrowthDeath) = (:PAR, )
@inline biogeochemical_drift_velocity(bgc::SimplePlanktonGrowthDeath, ::Val{:P}) = bgc.sinking_velocity
@inline biogeochemical_advection_scheme(bgc::SimplePlanktonGrowthDeath, ::Val{:P}) = bgc.advection_scheme

@inline biogeochemical_auxiliary_fieilds(bgc::SimplePlanktonGrowthDeath) = (PAR = bgc.light_attenuation_model.par_field, )

@inline function (bgc::SimplePlanktonGrowthDeath)(::Val{:P}, x, y, z, t, P, PAR)
   μ₀ = bgc.growth_rate
   k = bgc.light_limit
   m = bgc.mortality_rate

   (μ₀ * (1 - exp(-PAR/k)) - m) * P
end

#=
# Note, if we subtypted AbstractBiogeochemistry we would write
@inline function (bgc::SimplePlanktonGrowthDeath)(i, j, k, grid, ::Val{:P}, clock, fields)
    z = znode(Center(), k, grid)
    P = @inbounds fields.P[i, j, k]
    return (bgc.μ₀ * exp(z / bgc.λ) - bgc.m) * P
end
=#

#####
##### Setting up the integration of the Photosynthetically Available Radiation
#####

struct SimplePhyotosyntheticallyActiveRadiation{FT, SP, P, A}
    water_light_attenuation_coefficient :: FT
    phytoplankton_light_attenuation_coefficient :: FT
    phytoplankton_light_attenuation_exponent :: FT
    surface_PAR :: SP

    par_field :: P
    attenuating_fields :: A

    function SimplePhyotosyntheticallyActiveRadiation(; grid,
                                                        water_light_attenuation_coefficient :: FT = 0.12,
                                                        phytoplankton_light_attenuation_coefficient :: FT = 0.06,
                                                        phytoplankton_light_attenuation_exponent :: FT = 0.6,
                                                        surface_PAR :: SP = t -> 100.0*max(0.0, sin(t*π/(12hours))),
                                                        attenuating_fields :: A = (:P, )) where {FT, SP, A}

        PAR_field = CenterField(grid)

        P = typeof(PAR_field)

        return new{FT, SP, P, A}(water_light_attenuation_coefficient,
                                 phytoplankton_light_attenuation_coefficient,
                                 phytoplankton_light_attenuation_exponent,
                                 surface_PAR,
                                 PAR_field,
                                 attenuating_fields)
        end
end

@kernel function update_PhotosyntheticallyActiveRatiation!(light_attenuation_model::SimplePhyotosyntheticallyActiveRadiation,
                                                           P, grid, t) 
    i, j = @index(Global, NTuple)
    
    PAR⁰ = light_attenuation_model.surface_PAR(t)
    e, kʷ, χ = light_attenuation_model.phytoplankton_light_attenuation_exponent, light_attenuation_model.water_light_attenuation_coefficient, light_attenuation_model.phytoplankton_light_attenuation_coefficient

    zᶜ = znodes(Center, grid)
    zᶠ = znodes(Face, grid)
    
    ∫chl = @inbounds - (zᶜ[grid.Nz] - zᶠ[grid.Nz]) * P[i, j, grid.Nz] ^ e
    @inbounds light_attenuation_model.par_field[i, j, grid.Nz] =  PAR⁰ * exp(kʷ * zᶜ[grid.Nz] - χ * ∫chl)

    @inbounds for k in grid.Nz-1:-1:1
        ∫chl += (zᶜ[k + 1] - zᶠ[k])*P[i, j, k + 1]^e + (zᶠ[k] - zᶜ[k])*P[i, j, k]^e
        light_attenuation_model.par_field[i, j, k] =  PAR⁰*exp(kʷ * zᶜ[k] - χ * ∫chl)
    end
end 


# Call the integration
function (light_attenuation_model::SimplePhyotosyntheticallyActiveRadiation)(model)
    arch = architecture(model.grid)
    event = launch!(arch, model.grid, :xy, update_PhotosyntheticallyActiveRatiation!, 
                    light_attenuation_model,
                    model.tracers.P, 
                    model.grid, 
                    model.clock.time)
    wait(event)
end

@inline update_biogeochemical_state!(bgc::SimplePlanktonGrowthDeath, model) = bgc.light_attenuation_model(model)

#####
##### Setup the model
#####

grid = RectilinearGrid(size = (64, 64),
                       extent = (64, 64),
                       halo = (3, 3),
                       topology = (Periodic, Flat, Bounded))

buoyancy_flux_bc = FluxBoundaryCondition(1e-8)

N² = 1e-4 # s⁻²
buoyancy_gradient_bc = GradientBoundaryCondition(N²)
buoyancy_bcs = FieldBoundaryConditions(top = buoyancy_flux_bc, bottom = buoyancy_gradient_bc)

biogeochemistry = SimplePlanktonGrowthDeath(; grid)

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

simulation = Simulation(model, Δt=2minutes, stop_time=5day)

wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=2minutes)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

progress(sim) = @printf("Iteration: %d, time: %s, Δt: %s\n",
                        iteration(sim), prettytime(sim), prettytime(sim.Δt))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

outputs = (w = model.velocities.w,
           P = model.tracers.P,
           PAR = model.biogeochemistry.light_attenuation_model.par_field,
           avg_P = Average(model.tracers.P, dims=(1, 2)))

simulation.output_writers[:simple_output] =
    JLD2OutputWriter(model, outputs,
                     schedule = TimeInterval(20minutes),
                     filename = "biogeochemistry_test.jld2",
                     overwrite_existing = true)

run!(simulation)
#=
#####
##### Example using BasicBiogeochemistry
#####

@inline growth(x, y, z, t, P, PAR, μ₀, k, m, other_params...) = (μ₀ * (1 - exp(-PAR/k)) - m) * P

biogeochemistry_parameters = (
    growth_rate = 1/day,
    light_limit = 3.5,
    mortality_rate = 0.1/day,
)

biogeochemistry = BasicBiogeochemistry(tracers = :P, 
                                           auxiliary_fields = :PAR, 
                                           transitions = (; P=growth),
                                           parameters = biogeochemistry_parameters,
                                           drift_velocities = (P = (0.0, 0.0, -200/day), ),
                                           light_attenuation_model = SimplePhyotosyntheticallyActiveRadiation())

grid = RectilinearGrid(size = (64, 64),
extent = (64, 64),
halo = (3, 3),
topology = (Periodic, Flat, Bounded))

buoyancy_flux_bc = FluxBoundaryCondition(1e-8)

N² = 1e-4 # s⁻²
buoyancy_gradient_bc = GradientBoundaryCondition(N²)
buoyancy_bcs = FieldBoundaryConditions(top = buoyancy_flux_bc, bottom = buoyancy_gradient_bc)

PAR = CenterField(grid)

model = NonhydrostaticModel(; grid, biogeochemistry,
                            advection = WENO(; grid),
                            timestepper = :RungeKutta3,
                            closure = ScalarDiffusivity(ν=1e-4, κ=1e-4),
                            coriolis = FPlane(f=1e-4),
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            boundary_conditions = (; b=buoyancy_bcs),
                            auxiliary_fields = (; PAR))

mixed_layer_depth = 32 # m
stratification(z) = z < -mixed_layer_depth ? N² * z : - N² * mixed_layer_depth
noise(z) = 1e-4 * N² * grid.Lz * randn() * exp(z / 4)
initial_buoyancy(x, y, z) = stratification(z) + noise(z)

set!(model, b=initial_buoyancy, P = 1.0)

simulation = Simulation(model, Δt=2minutes, stop_time=1day)

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
                            filename = "biogeochemistry_test.jld2",
                            overwrite_existing = true)

run!(simulation)

#= Plot to sanity check

filepath = simulation.output_writers[:simple_output].filepath
#using CairoMakie

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
Colorbar(fig[2, 1], hm_w; label = "Vertical velocity (m/s)", flipaxis = false)

hm_P = heatmap!(ax_P, xp, zp, Pₙ; colormap = :matter, colorrange = P_lims)
Colorbar(fig[3, 1], hm_P; label = "Plankton concentration (mmmol N/m³)", flipaxis = false)

hm_PAR = heatmap!(ax_PAR, xp, zp, PARₙ; colormap = :matter, colorrange = PAR_lims)
Colorbar(fig[2, 3], hm_PAR; label = "Photosynthetically Available Ratiation (einstein/m²/s)", flipaxis = false)

# And, finally, we record a movie.

frames = 1:length(times)

@info "Making an animation of convecting plankton..."

record(fig, "biogeochemistry_test.mp4", frames, framerate=8) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    n[] = i
end
=#=#