using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using Oceananigans.TimeSteppers: time_step!

grid = RectilinearGrid(size=64, z=(-256, 0), topology=(Flat, Flat, Bounded))
coriolis = FPlane(f=1e-4)

N² = 1e-6
Qᵇ = +1e-8
Qᵘ = -2e-4 #

b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ))
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))

closure = CATKEVerticalDiffusivity()

model = HydrostaticFreeSurfaceModel(; grid, closure, coriolis,
                                    tracers = (:b, :e),
                                    buoyancy = BuoyancyTracer(),
                                    boundary_conditions = (; b=b_bcs, u=u_bcs))
                                    
bᵢ(z) = N² * z
set!(model, b=bᵢ, e=1e-6)

simulation = Simulation(model, Δt=10minutes, stop_iteration=1000)

closurename = string(nameof(typeof(closure)))

diffusivities = (κᵘ = model.diffusivity_fields.κᵘ,
                 κᶜ = model.diffusivity_fields.κᶜ)

outputs = merge(model.velocities, model.tracers, diffusivities)

simulation.output_writers[:fields] =
    JLD2OutputWriter(model, outputs,
                     #schedule = TimeInterval(10minutes),
                     schedule = IterationInterval(100),
                     filename = "windy_convection_" * closurename,
                     overwrite_existing = true)

progress(sim) = @info string("Iter: ", iteration(sim), " t: ", prettytime(sim))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

@info "Running a simulation of $model..."

time_step!(model, 10minutes)

@time for n = 1:100
    time_step!(model, 10minutes)
end


