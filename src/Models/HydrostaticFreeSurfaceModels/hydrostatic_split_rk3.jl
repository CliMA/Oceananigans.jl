# import Oceananigans.TimeSteppers: time_step!

# function time_step!(model::AbstractModel{<:SplitRungeKutta3TimeStepper}, Δt; callbacks=[])
#     Δt == 0 && @warn "Δt == 0 may cause model blowup!"

#     # Be paranoid and update state at iteration 0, in case run! is not used:
#     model.clock.iteration == 0 && update_state!(model, callbacks; compute_tendencies = true)

#     cache_previous_fields!(model)
#     β¹ = model.timestepper.β¹
#     β² = model.timestepper.β²

#     ####
#     #### First stage
#     ####

#     # First stage: n -> n + 1/3
#     model.clock.stage = 1
#     Δτ = Δt / β¹
#     compute_flux_bc_tendencies!(model)
#     split_rk3_substep!(model, Δt / β¹)
#     compute_pressure_correction!(model, Δt / β¹)
#     make_pressure_correction!(model, Δt / β¹)
#     update_state!(model, callbacks; compute_tendencies = true)

#     ####
#     #### Second stage
#     ####

#     # Second stage: n -> n + 1/2
#     model.clock.stage = 2

#     compute_flux_bc_tendencies!(model)
#     split_rk3_substep!(model, Δt / β²)
#     compute_pressure_correction!(model, Δt / β²)
#     make_pressure_correction!(model, Δt / β²)
#     update_state!(model, callbacks; compute_tendencies = true)

#     ####
#     #### Third stage
#     ####

#     # Third stage: n -> n + 1
#     model.clock.stage = 3

#     compute_flux_bc_tendencies!(model)
#     split_rk3_substep!(model, Δt)
#     compute_pressure_correction!(model, Δt)
#     make_pressure_correction!(model, Δt)
#     update_state!(model, callbacks; compute_tendencies = true)

#     step_lagrangian_particles!(model, Δt)

#     tick!(model.clock, Δt)

#     return nothing
# end
