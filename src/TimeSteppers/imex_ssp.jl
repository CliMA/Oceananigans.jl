
struct IMEXSSPTimeStepper{TG, TE, TD, PF, TI} <: AbstractTimeStepper
    Gⁿ::TG               # Previous tendencies
    Gⁱ::TE                
    Dⁱ::TD                  
    Ψ⁻::PF               # Previous prognostic fields
    implicit_solver::TI  # Implicit solver
end

function IMEXSSPTimeStepper(grid, prognostic_fields, args...;
                            implicit_solver = nothing,
                            Gⁿ = map(similar, prognostic_fields),
                            Ψ⁻ = map(similar, prognostic_fields),
                            kwargs...)

    Gⁱ = (n₁ = deepcopy(Gⁿ),
          n₂ = deepcopy(Gⁿ),
          n₃ = deepcopy(Gⁿ))

    if :η ∈ keys(Ψ⁻)
        Dⁱ = (; n₁ = (; η = deepcopy(Ψ⁻.η)),
                n₂ = (; η = deepcopy(Ψ⁻.η)), 
                n₃ = (; η = deepcopy(Ψ⁻.η)))

        Ψ⁻ = merge(Ψ⁻, (; η₁ = deepcopy(Ψ⁻.η), η₂ = deepcopy(Ψ⁻.η), η₃ = deepcopy(Ψ⁻.η)))
    else
        Dⁱ = nothing
    end

    return IMEXSSPTimeStepper(Gⁿ, Gⁱ, Dⁱ, Ψ⁻, implicit_solver)
end


function time_step!(model::AbstractModel{<:IMEXSSPTimeStepper}, Δt; callbacks=[])

    if model.clock.iteration == 0
        update_state!(model, callbacks)
    end

    cache_previous_fields!(model)
    grid = model.grid

    ####
    #### Loop over the stages
    ####

    model.clock.stage = 1
    ssp_substep1!(model, Δt, callbacks)
    update_state!(model, callbacks)
    
    model.clock.stage = 2
    ssp_substep2!(model, Δt, callbacks)
    update_state!(model, callbacks)
    
    model.clock.stage = 3
    ssp_substep3!(model, Δt, callbacks)
    update_state!(model, callbacks)
    
    # Finalize step
    step_lagrangian_particles!(model, Δt)
    tick!(model.clock, Δt)

    return nothing
end

ssp_substep1!(model, Δt, callbacks) = throw(ErrorException("ssp_substep1! not implemented for model=$(typeof(model))"))
ssp_substep2!(model, Δt, callbacks) = throw(ErrorException("ssp_substep2! not implemented for model=$(typeof(model))"))
ssp_substep3!(model, Δt, callbacks) = throw(ErrorException("ssp_substep3! not implemented for model=$(typeof(model))"))