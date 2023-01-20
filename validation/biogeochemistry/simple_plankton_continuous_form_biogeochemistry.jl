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

struct SimplePlanktonGrowthDeath{FT, W, SP, A, P} <: AbstractContinuousFormBiogeochemistry
    growth_rate :: FT
    light_limit :: FT
    mortality_rate :: FT
    water_light_attenuation_coefficient :: FT
    phytoplankton_light_attenuation_coefficient :: FT
    phytoplankton_light_attenuation_exponent :: FT
    sinking_velocity :: W
    surface_PAR :: SP
    advection_scheme :: A
    PAR :: P
end

@inline default_surface_radiation(t) = 100 * max(0, sin(t * π / 12hours))

function SimplePlanktonGrowthDeath(FT=Float64; grid,
                                   growth_rate = 1/day,
                                   light_limit = 3.5,
                                   mortality_rate = 0.3/day,
                                   sinking_velocity = 0.0,
                                   water_light_attenuation_coefficient = 0.01,
                                   phytoplankton_light_attenuation_coefficient = 0.3,
                                   phytoplankton_light_attenuation_exponent = 0.6,
                                   surface_PAR = default_surface_radiation,
                                   advection_scheme = nothing)

    if sinking_velocity != 0
        advection_scheme = CenteredSecondOrder()
    end

    u, v, w = maybe_constant_field.((0.0, 0.0, - sinking_velocity))
    sinking_velocity = (; u, v, w)
    W = typeof(sinking_velocity)

    PAR = CenterField(grid)
    P = typeof(PAR)

    return SimplePlanktonGrowthDeath(FT(growth_rate)
                                     FT(light_limit),
                                     FT(mortality_rate),
                                     FT(water_light_attenuation_coefficient),
                                     FT(phytoplankton_light_attenuation_coefficient),
                                     FT(phytoplankton_light_attenuation_exponent),
                                     sinking_velocity,
                                     surface_PAR,
                                     advection_scheme,
                                     PAR)
end 



######
###### Functions we have to define to setup the biogeochemical mdoel
######

const SPGD = SimplePlanktonGrowthDeath

@inline          required_biogeochemical_tracers(::SPGD) = tuple(:P)
@inline required_biogeochemical_auxiliary_fields(::SPGD) = tuple(:PAR)
@inline       biogeochemical_auxiliary_fields(bgc::SPGD) = (; PAR = bgc.PAR)
@inline   biogeochemical_drift_velocity(bgc::SPGD, ::Val{:P}) = bgc.sinking_velocity
@inline biogeochemical_advection_scheme(bgc::SPGD, ::Val{:P}) = bgc.advection_scheme

@inline function (bgc::SimplePlanktonGrowthDeath)(::Val{:P}, x, y, z, t, P, PAR)
   μ₀ = bgc.growth_rate
   k = bgc.light_limit
   m = bgc.mortality_rate
   return (μ₀ * (1 - exp(-PAR / k)) - m) * P
end

#####
##### Setting up the integration of the Photosynthetically Available Radiation
#####

@kernel function update_PhotosyntheticallyActiveRatiation!(bgc, P, PAR, grid, t) 
    i, j = @index(Global, NTuple)
    
    PAR⁰ = bgc.surface_PAR(t)
    e  = bgc.phytoplankton_light_attenuation_exponent
    kʷ = bgc.water_light_attenuation_coefficient
    χ  = bgc.phytoplankton_light_attenuation_coefficient

    zᶜ = znodes(Center, grid)
    zᶠ = znodes(Face, grid)
    
    ∫chl = @inbounds - (zᶜ[grid.Nz] - zᶠ[grid.Nz]) * P[i, j, grid.Nz] ^ e
    @inbounds PAR[i, j, grid.Nz] =  PAR⁰ * exp(kʷ * zᶜ[grid.Nz] - χ * ∫chl)

    @unroll for k in grid.Nz-1:-1:1
        @inbounds begin
            ∫chl += (zᶜ[k + 1] - zᶠ[k])*P[i, j, k + 1]^e + (zᶠ[k] - zᶜ[k])*P[i, j, k]^e
            PAR[i, j, k] =  PAR⁰*exp(kʷ * zᶜ[k] - χ * ∫chl)
        end
    end
end 

# Call the integration
@inline function update_biogeochemical_state!(bgc::SimplePlanktonGrowthDeath, model)
    arch = architecture(model.grid)
    event = launch!(arch, model.grid, :xy, update_PhotosyntheticallyActiveRatiation!, 
                    bgc,
                    model.tracers.P, 
                    bgc.PAR,
                    model.grid, 
                    model.clock.time)
    wait(event)
end

#####
##### Set up the model
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
           PAR = model.biogeochemistry.PAR,
           avg_P = Average(model.tracers.P, dims=(1, 2)))

filename = "simple_plankton_continuous_form_biogeochemistry"

simulation.output_writers[:simple_output] = JLD2OutputWriter(model, outputs; filename,
                                                             schedule = TimeInterval(20minutes),
                                                             overwrite_existing = true)

run!(simulation)

