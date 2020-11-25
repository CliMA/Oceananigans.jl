"""
    QuasiAdamsBashforth2TimeStepper{T, TG} <: AbstractTimeStepper

Holds tendency fields and the parameter `χ` for a modified second-order
Adams-Bashforth timestepping method.
"""
struct QuasiAdamsBashforth2TimeStepper{T, TG} <: AbstractTimeStepper
     χ :: T
    Gⁿ :: TG
    G⁻ :: TG
end

"""
    QuasiAdamsBashforth2TimeStepper(arch, grid, tracers, χ=0.1;
                                    Gⁿ = TendencyFields(arch, grid, tracers),
                                    G⁻ = TendencyFields(arch, grid, tracers))

Return an QuasiAdamsBashforth2TimeStepper object with tendency fields on `arch` and
`grid` with AB2 parameter `χ`. The tendency fields can be specified via optional
kwargs.
"""
function QuasiAdamsBashforth2TimeStepper(arch, grid, tracers, χ=0.1;
                                         Gⁿ = TendencyFields(arch, grid, tracers),
                                         G⁻ = TendencyFields(arch, grid, tracers))

    return QuasiAdamsBashforth2TimeStepper{eltype(grid), typeof(Gⁿ)}(χ, Gⁿ, G⁻)
end

#####
##### Time steppping
#####

"""
    time_step!(model::AbstractModel{<:QuasiAdamsBashforth2TimeStepper}, Δt; euler=false)

Step forward `model` one time step `Δt` with a 2nd-order Adams-Bashforth method and
pressure-correction substep. Setting `euler=true` will take a forward Euler time step.
"""
function time_step!(model::AbstractModel{<:QuasiAdamsBashforth2TimeStepper}, Δt; euler=false)

    χ = ifelse(euler, convert(eltype(model.grid), -0.5), model.timestepper.χ)

    # Be paranoid and update state at iteration 0, in case run! is not used:
    model.clock.iteration == 0 && update_state!(model)

    calculate_tendencies!(model)

    ab2_step!(model, Δt, χ) # full step for tracers, fractional step for velocities.

    calculate_pressure_correction!(model, Δt)
    pressure_correct_velocities!(model, Δt)

    tick!(model.clock, Δt)
    update_state!(model)
    store_tendencies!(model)

    return nothing
end

#####
##### Tracer time stepping and predictor velocity updating
#####

function ab2_step!(model, Δt, χ)

    workgroup, worksize = work_layout(model.grid, :xyz)

    barrier = Event(device(model.architecture))

    step_field_kernel! = ab2_step_field!(device(model.architecture), workgroup, worksize)

    model_fields = fields(model)

    events = []

    for (i, field) in enumerate(model_fields)
        Gⁿ = model.timestepper.Gⁿ[i]
        G⁻ = model.timestepper.G⁻[i]

        field_event = step_field_kernel!(field, Δt, x, Gⁿ, G⁻, dependencies=Event(device(model.architecture)))

        push!(events, field_event)
    end




    #step_velocities_kernel! = ab2_step_velocities!(device(model.architecture), workgroup, worksize)
    #step_tracer_kernel! = ab2_step_tracer!(device(model.architecture), workgroup, worksize)

    #velocities_event = step_velocities_kernel!(model.velocities, Δt, χ,
    #                                           model.timestepper.Gⁿ,
    #                                           model.timestepper.G⁻,
    #                                           dependencies=Event(device(model.architecture)))

    #events = [velocities_event]

    #for i in 1:length(model.tracers)
    #    @inbounds c = model.tracers[i]
    #    @inbounds Gcⁿ = model.timestepper.Gⁿ[i+3]
    #    @inbounds Gc⁻ = model.timestepper.G⁻[i+3]
    #    event = step_tracer_kernel!(c, Δt, χ, Gcⁿ, Gc⁻, dependencies=barrier)
    #    push!(events, event)
    #end

    wait(device(model.architecture), MultiEvent(Tuple(events)))

    return nothing
end

"""
Time step via

    `U^{n+1} = U^n + Δt ( (3/2 + χ) * G^{n} - (1/2 + χ) G^{n-1} )`

"""

@kernel function ab2_step_field!(U, Δt, χ::FT, Gⁿ, G⁻) where FT
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        U[i, j, k] += Δt * (  (FT(1.5) + χ) * Gⁿ[i, j, k] - (FT(0.5) + χ) * G⁻[i, j, k] )

    end
end
