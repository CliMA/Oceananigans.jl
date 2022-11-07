using Oceananigans
using Oceananigans.Units: minutes, hour, hours, day
using CairoMakie, Measures
using Printf

#####
##### Build a model for plankton/nutrient/detritus interactions
#####
using Oceananigans.Biogeochemistry: AbstractBiogeochemistry
import Oceananigans.Biogeochemistry: required_tracers

struct NutrientsPlanktonDetritus{FT} <: AbstractBiogeochemistry
    nutrient_limitation_saturation :: FT
    mortality_rate :: FT
    nitrification :: FT
end

required_tracers(::NutrientsPlanktonDetritus) = (:N, :P, :D)

@inline function (bgc::NutrientsPlanktonDetritus)(i, j, k, grid, ::Val{:N}, clock, fields)
    P = @inbounds fields.P[i, j, k]
    N = @inbounds fields.N[i, j, k]
    D = @inbounds fields.D[i, j, k]

    return bgc.nitrification*D - P*N/(N+bgc.nutrient_limitation_saturation) 
end

@inline function (bgc::NutrientsPlanktonDetritus)(i, j, k, grid, ::Val{:P}, clock, fields)
    P = @inbounds fields.P[i, j, k]
    N = @inbounds fields.N[i, j, k]
    return P*N/(N+bgc.nutrient_limitation_saturation) - bgc.mortality_rate*P
end

@inline function (bgc::NutrientsPlanktonDetritus)(i, j, k, grid, ::Val{:D}, clock, fields)
    P = @inbounds fields.P[i, j, k]
    D = @inbounds fields.D[i, j, k]

    return bgc.mortality_rate*P - bgc.nitrification*D
end

grid = RectilinearGrid(size=(64, 64), extent=(64, 64), halo=(3, 3), topology=(Periodic, Flat, Bounded))

buoyancy_flux(x, y, t, params) = params.initial_buoyancy_flux * exp(-t^4 / (24 * params.shut_off_time^4))
buoyancy_flux_parameters = (initial_buoyancy_flux = 1e-8, shut_off_time = 2hours)
buoyancy_flux_bc = FluxBoundaryCondition(buoyancy_flux, parameters = buoyancy_flux_parameters)

N² = 1e-4 # s⁻²
buoyancy_gradient_bc = GradientBoundaryCondition(N²)
buoyancy_bcs = FieldBoundaryConditions(top = buoyancy_flux_bc, bottom = buoyancy_gradient_bc)

model = NonhydrostaticModel(; grid,
                            advection = UpwindBiasedFifthOrder(),
                            timestepper = :RungeKutta3,
                            closure = ScalarDiffusivity(ν=1e-4, κ=1e-4),
                            coriolis = FPlane(f=1e-4),
                            tracers = (:b, :N, :P, :D), # P for Plankton
                            buoyancy = BuoyancyTracer(),
                            biogeochemistry = NutrientsPlanktonDetritus(1.0, 0.1/day, 0.01/day),
                            boundary_conditions = (; b=buoyancy_bcs))

mixed_layer_depth = 32 # m

stratification(z) = z < -mixed_layer_depth ? N² * z : - N² * mixed_layer_depth
noise(z) = 1e-4 * N² * grid.Lz * randn() * exp(z / 4)
initial_buoyancy(x, y, z) = stratification(z) + noise(z)

set!(model, b=initial_buoyancy, N=1)

simulation = Simulation(model, Δt=2minutes, stop_time=24hours)

wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=2minutes)

simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))


progress(sim) = @printf("Iteration: %d, time: %s, Δt: %s\n",
                        iteration(sim), prettytime(sim), prettytime(sim.Δt))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

# and a basic `JLD2OutputWriter` that writes velocities and both
# the two-dimensional and horizontally-averaged plankton concentration,

outputs = (w = model.velocities.w,
           N = model.tracers.N,
           P = model.tracers.P,
           D = model.tracers.D)

simulation.output_writers[:simple_output] =
    JLD2OutputWriter(model, outputs,
                     schedule = TimeInterval(20minutes),
                     filename = "convecting_NPD.jld2",
                     overwrite_existing = true)

# !!! info "Using multiple output writers"
#     Because each output writer is associated with a single output `schedule`,
#     it often makes sense to use _different_ output writers for different types of output.
#     For example, smaller outputs that consume less disk space may be written more
#     frequently without threatening the capacity of your hard drive.
#     An arbitrary number of output writers may be added to `simulation.output_writers`.
#
# The simulation is set up. Let there be plankton:

run!(simulation)