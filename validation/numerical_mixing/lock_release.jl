using Oceananigans
using Oceananigans.Units
using Oceananigans.Utils: prettytime
using Oceananigans.Advection: WENOVectorInvariant
using Oceananigans.AbstractOperations: GridMetricOperation  
using Printf
using SeawaterPolynomials
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

grid = RectilinearGrid(size = (128, 20), 
                          x = (0, 64kilometers), 
                          z = (-20, 0), 
                       halo = (6, 6),
                   topology = (Bounded, Flat, Bounded))

buoyancy = SeawaterBuoyancy(equation_of_state = TEOS10EquationOfState())

model = HydrostaticFreeSurfaceModel(; grid, buoyancy,
                         momentum_advection = WENO(),
                           tracer_advection = Oceananigans.Advection.RotatedAdvection(WENO()),
                                    closure = nothing, 
                                    tracers = (:T, :S),
                               free_surface = SplitExplicitFreeSurface(; substeps = 10))

g = model.free_surface.gravitational_acceleration

Tᵢ(x, z) = x < 32kilometers ? 30 : 5
Sᵢ(x, z) = 32.5 - (Lz + z) / z

set!(model, T = Tᵢ, S = Sᵢ)

Δt = 1

@info "the time step is $Δt"

simulation = Simulation(model; Δt, stop_iteration = 10000000, stop_time = 17hours) 

field_outputs = merge(model.velocities, model.tracers)

simulation.output_writers[:other_variables] = JLD2OutputWriter(model, field_outputs, 
                                                               overwrite_existing = true,
                                                               schedule = IterationInterval(100),
                                                               filename = "lock_release") 

RPE = Field(RPEDensityOperation(grid, tracers = model.tracers, buoyancy = model.buoyancy))
compute!(RPE)
RPE_init_field = deepcopy(RPE)

function progress(sim)
    w  = interior(sim.model.velocities.w, :, :, sim.model.grid.Nz+1)
    u  = sim.model.velocities.u
    T  = sim.model.tracers.b

    compute!(RPE)
    msg0 = @sprintf("Time: %s iteration %d ", prettytime(sim.model.clock.time), sim.model.clock.iteration)
    msg1 = @sprintf("extrema w: %.2e %.2e ",  maximum(w),  minimum(w))
    msg3 = @sprintf("extrema T: %.2e %.2e ", maximum(T),  minimum(T))
    msg4 = @sprintf("total RPE: %6.3e ", total_RPE(RPE))
    @info msg0 * msg1 * msg3 * msg4

    return nothing
end

RPE_init = total_RPE(RPE_init_field)
delta_RPE = Float64[]

function save_RPE(sim)
    compute!(RPE)
    push!(delta_RPE, total_RPE(RPE) - RPE_init)  
    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))
simulation.callbacks[:save_RPE] = Callback(save_RPE, IterationInterval(100)) 
run!(simulation)

ρ = Oceananigans.Models.seawater_density(model)
ρ = compute!(Field(ρ))