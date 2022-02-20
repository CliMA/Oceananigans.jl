using Oceananigans.Fields: FunctionField, location
using Oceananigans.TurbulenceClosures: implicit_step!
using Oceananigans.Architectures: device_event

mutable struct QuasiAdamsBashforth2TimeStepper{FT, GT, IT} <: AbstractTimeStepper
                  χ :: FT
        previous_Δt :: FT
                 Gⁿ :: GT
                 G⁻ :: GT
    implicit_solver :: IT
end

"""
    QuasiAdamsBashforth2TimeStepper(grid, tracers,
                                    χ = 0.1;
                                    implicit_solver::IT = nothing,
                                    Gⁿ = TendencyFields(grid, tracers),
                                    G⁻ = TendencyFields(grid, tracers)) where IT

Return an QuasiAdamsBashforth2TimeStepper object with tendency fields and
`grid` with AB2 parameter `χ`. The tendency fields can be specified via optional
kwargs.
"""
function QuasiAdamsBashforth2TimeStepper(grid, tracers,
                                         χ = 0.1;
                                         implicit_solver::IT = nothing,
                                         Gⁿ = TendencyFields(grid, tracers),
                                         G⁻ = TendencyFields(grid, tracers)) where IT

    FT = eltype(grid)
    GT = typeof(Gⁿ)

    return QuasiAdamsBashforth2TimeStepper{FT, GT, IT}(χ, Inf, Gⁿ, G⁻, implicit_solver)
end

function reset!(timestepper::QuasiAdamsBashforth2TimeStepper)
    timestepper.previous_Δt = Inf
    return nothing
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
    Δt == 0 && @warn "Δt == 0 may cause model blowup!"

    # Shenanigans for properly starting the AB2 loop with an Euler step
    euler = euler || (Δt != model.timestepper.previous_Δt)
    χ = ifelse(euler, convert(eltype(model.grid), -0.5), model.timestepper.χ)

    euler && @debug "Taking a forward Euler step."

    model.timestepper.previous_Δt = Δt

    # Be paranoid and update state at iteration 0
    model.clock.iteration == 0 && update_state!(model)

    calculate_tendencies!(model)

    ab2_step!(model, Δt, χ) # full step for tracers, fractional step for velocities.

    calculate_pressure_correction!(model, Δt)
    pressure_correct_velocities!(model, Δt)

    tick!(model.clock, Δt)
    update_state!(model)
    store_tendencies!(model)
    update_particle_properties!(model, Δt)

    return nothing
end

#####
##### Time stepping in each step
#####

""" Generic implementation. """
function ab2_step!(model, Δt, χ)

    workgroup, worksize = work_layout(model.grid, :xyz)

    arch = model.architecture
    barrier = device_event(arch)

    step_field_kernel! = ab2_step_field!(device(arch), workgroup, worksize)

    model_fields = prognostic_fields(model)

    events = []

    for (i, field) in enumerate(model_fields)

        field_event = step_field_kernel!(field, Δt, χ,
                                         model.timestepper.Gⁿ[i],
                                         model.timestepper.G⁻[i],
                                         dependencies = device_event(arch))

        push!(events, field_event)

        # TODO: function tracer_index(model, field_index) = field_index - 3, etc...
        tracer_index = i - 3 # assumption

        implicit_step!(field,
                       model.timestepper.implicit_solver,
                       model.clock,
                       Δt,
                       model.closure,
                       tracer_index,
                       model.diffusivity_fields,
                       model.tracers,
                       dependencies = field_event)
    end

    wait(device(model.architecture), MultiEvent(Tuple(events)))

    return nothing
end

"""
Time step via

    `U^{n+1} = U^n + Δt ((3/2 + χ) * G^{n} - (1/2 + χ) G^{n-1})`

"""
@kernel function ab2_step_field!(u, Δt, χ, Gⁿ, G⁻)
    i, j, k = @index(Global, NTuple)

    T = eltype(u)
    one_point_five = convert(T, 1.5)
    oh_point_five = convert(T, 0.5)

    @inbounds u[i, j, k] += Δt * ((one_point_five + χ) * Gⁿ[i, j, k] - (oh_point_five + χ) * G⁻[i, j, k])
end

@kernel ab2_step_field!(::FunctionField, args...) = nothing

