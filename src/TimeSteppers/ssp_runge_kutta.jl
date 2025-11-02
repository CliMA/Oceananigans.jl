
using Oceananigans.Architectures: architecture
using Oceananigans: fields

struct SSPRungeKuttaTimeStepper{N, TG, TE, PF, TI} <: AbstractTimeStepper
    Gⁿ :: TG
    G⁻ :: TE
    Ψ⁻ :: PF # prognostic state at the previous timestep
    implicit_solver :: TI
end

function SSPRungeKuttaTimeStepper(grid, prognostic_fields, args...;
                                  stages = 3,
                                  implicit_solver::TI = nothing,
                                  Gⁿ::TG = map(similar, prognostic_fields),
                                  Ψ⁻::PF = map(similar, prognostic_fields),
                                  G⁻::TE = map(similar, prognostic_fields)) where {TI, TG, TE, PF}

    if stages != 3 && stages != 4
        error("SSP Runge-Kutta schemes with $stages stages are not implemented. Available schemes are only SSPRK3 for the moment.")
    end

    return SSPRungeKuttaTimeStepper{stages, TG, TE, PF, TI}(Gⁿ, G⁻, Ψ⁻, implicit_solver)
end

function time_step!(model::AbstractModel{<:SSPRungeKuttaTimeStepper{N}}, Δt; callbacks=[]) where N

    if model.clock.iteration == 0
        update_state!(model, callbacks)
    end

    cache_previous_fields!(model)
    grid = model.grid

    ####
    #### Loop over the stages
    ####

    for stage in 1:N
        # Update the clock stage
        model.clock.stage = stage
        ssprk_substep!(model, Δt, callbacks)
        ssprk_cache_previous_tendencies!(model)

        # Update the state
        update_state!(model, callbacks)
    end

    # Finalize step
    step_lagrangian_particles!(model, Δt)
    tick!(model.clock, Δt)

    return nothing
end

#####
##### These functions need to be implemented by every model independently
#####

ssprk_substep!(model::AbstractModel, Δt, callbacks)  = error("ssprk_substep! not implemented for $(typeof(model))")
ssprk_cache_previous_tendencies!(model::AbstractModel) = error("ssprk_cache_previous_tendencies! not implemented for $(typeof(model))")