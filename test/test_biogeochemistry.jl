using Oceananigans, Printf
using Oceananigans.Units: minutes, hour, hours, day
using Oceananigans.Biogeochemistry: AbstractContinuousFormBiogeochemistry
using Oceananigans.Grids: znode

import Oceananigans.Biogeochemistry:
    required_biogeochemical_tracers,
    biogeochemical_drift_velocity,
    biogeochemical_advection_scheme

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
@inline biogeochemical_drift_velocity(bgc::SimplePlanktonGrowthDeath, ::Val{:P}) = (0.0, 0.0, bgc.w)
@inline biogeochemical_advection_scheme(bgc::SimplePlanktonGrowthDeath, ::Val{:P}) = bgc.advection

@inline function (bgc::SimplePlanktonGrowthDeath)(::Val{:P}, x, y, z, t, P)
    μ₀ = bgc.growth_rate
    λ = bgc.light_penetration_depth
    m = bgc.mortality_rate

    (μ₀ * exp(z / λ) - m) * P
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

#####
##### Example using Biogeochemistry
#####

