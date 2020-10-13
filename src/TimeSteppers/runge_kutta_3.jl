"""
    RungeKutta3TimeStepper{FT, TG} <: AbstractTimeStepper

Holds parameters and tendency fields for a low storage, third-order Runge-Kutta-Wray
time-stepping scheme described by Le and Moin (1991).
"""
struct RungeKutta3TimeStepper{FT, TG} <: AbstractTimeStepper
    γ¹ :: FT
    γ² :: FT
    γ³ :: FT
    ζ² :: FT
    ζ³ :: FT
    Gⁿ :: TG
    G⁻ :: TG
end

"""
    RungeKutta3TimeStepper(arch, grid, tracers,
                           Gⁿ = TendencyFields(arch, grid, tracers),
                           G⁻ = TendencyFields(arch, grid, tracers))

Return an `RungeKutta3TimeStepper` object with tendency fields on `arch` and
`grid`. The tendency fields can be specified via optional kwargs.
"""
function RungeKutta3TimeStepper(arch, grid, tracers;
                                Gⁿ = TendencyFields(arch, grid, tracers),
                                G⁻ = TendencyFields(arch, grid, tracers))

    γ¹ = 8 // 15
    γ² = 5 // 12
    γ³ = 3 // 4

    ζ² = -17 // 60
    ζ³ = -5 // 12

    return RungeKutta3TimeStepper{eltype(grid), typeof(Gⁿ)}(γ¹, γ², γ³, ζ², ζ³, Gⁿ, G⁻)
end

#####
##### Time steppping
#####

"""
    time_step!(model::IncompressibleModel{<:RungeKutta3TimeStepper}, Δt; euler=false)

Step forward `model` one time step `Δt` with a 3rd-order Runge-Kutta method.
The 3rd-order Runge-Kutta method takes three intermediate substep stages to 
achieve a single timestep. A pressure correction step is applied at each intermediate
stage.
"""
function time_step!(model::IncompressibleModel{<:RungeKutta3TimeStepper}, Δt)

    γ¹ = model.timestepper.γ¹
    γ² = model.timestepper.γ²
    γ³ = model.timestepper.γ³

    ζ² = model.timestepper.ζ²
    ζ³ = model.timestepper.ζ³

    first_stage_Δt  = γ¹ * Δt
    second_stage_Δt = (γ² + ζ²) * Δt
    third_stage_Δt  = (γ³ + ζ³) * Δt

    #
    # First stage
    #
    
    precomputations!(model)
    calculate_tendencies!(model)

    rk3_substep!(model, Δt, γ¹, nothing)

    calculate_pressure_correction!(model, first_stage_Δt)
    pressure_correct_velocities!(model, first_stage_Δt)

    tick!(model.clock, first_stage_Δt; stage=true)

    #
    # Second stage
    #

    store_tendencies!(model)
    precomputations!(model)
    calculate_tendencies!(model)

    rk3_substep!(model, Δt, γ², ζ²)

    calculate_pressure_correction!(model, second_stage_Δt)
    pressure_correct_velocities!(model, second_stage_Δt)

    tick!(model.clock, second_stage_Δt; stage=true)

    #
    # Third stage
    #

    store_tendencies!(model)
    precomputations!(model)
    calculate_tendencies!(model)
    
    rk3_substep!(model, Δt, γ³, ζ³)

    calculate_pressure_correction!(model, third_stage_Δt)
    pressure_correct_velocities!(model, third_stage_Δt)

    tick!(model.clock, third_stage_Δt)

    return nothing
end

#####
##### Tracer time stepping and predictor velocity updating
#####

function rk3_substep!(model, Δt, γⁿ, ζⁿ)

    workgroup, worksize = work_layout(model.grid, :xyz)

    barrier = Event(device(model.architecture))

    substep_velocities_kernel! = rk3_substep_velocities!(device(model.architecture), workgroup, worksize)
    substep_tracer_kernel! = rk3_substep_tracer!(device(model.architecture), workgroup, worksize)

    velocities_event = substep_velocities_kernel!(model.velocities,
                                                  Δt, γⁿ, ζⁿ,
                                                  model.timestepper.Gⁿ,
                                                  model.timestepper.G⁻;
                                                  dependencies=barrier)

    events = [velocities_event]

    for i in 1:length(model.tracers)
        @inbounds c = model.tracers[i]
        @inbounds Gcⁿ = model.timestepper.Gⁿ[i+3]
        @inbounds Gc⁻ = model.timestepper.G⁻[i+3]

        tracer_event = substep_tracer_kernel!(c, Δt, γⁿ, ζⁿ, Gcⁿ, Gc⁻, dependencies=barrier)

        push!(events, tracer_event)
    end

    wait(device(model.architecture), MultiEvent(Tuple(events)))

    return nothing
end

"""
Time step tracers via the 3rd-order Runge-Kutta method

    `c^{m+1} = c^m + Δt (γⁿ Gc^{m} + ζⁿ Gc^{m-1})`,

where `m` denotes the substage. 
"""
@kernel function rk3_substep_tracer!(c, Δt, γⁿ, ζⁿ, Gcⁿ, Gc⁻)
    i, j, k = @index(Global, NTuple)

    @inbounds c[i, j, k] += Δt * (γⁿ * Gcⁿ[i, j, k] + ζⁿ * Gc⁻[i, j, k])
end

"""
Time step tracers from the first to the second stage via
the 3rd-order Runge-Kutta method

    `c^{2} = c^1 + Δt γ¹ Gc^{1}`.
"""
@kernel function rk3_substep_tracer!(c, Δt, γ¹, ::Nothing, Gc¹, Gc⁰)
    i, j, k = @index(Global, NTuple)

    @inbounds c[i, j, k] += Δt * γ¹ * Gc¹[i, j, k]
end

"""
Time step velocity fields with a 3rd-order Runge-Kutta method.
"""
@kernel function rk3_substep_velocities!(U, Δt, γⁿ, ζⁿ, Gⁿ, G⁻)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        U.u[i, j, k] += Δt * (γⁿ * Gⁿ.u[i, j, k] + ζⁿ * G⁻.u[i, j, k])
        U.v[i, j, k] += Δt * (γⁿ * Gⁿ.v[i, j, k] + ζⁿ * G⁻.v[i, j, k])
        U.w[i, j, k] += Δt * (γⁿ * Gⁿ.w[i, j, k] + ζⁿ * G⁻.w[i, j, k])
    end
end

"""
Time step velocity fields with a 3rd-order Runge-Kutta method.
"""
@kernel function rk3_substep_velocities!(U, Δt, γ¹, ::Nothing, G¹, G⁰)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        U.u[i, j, k] += Δt * γ¹ * G¹.u[i, j, k]
        U.v[i, j, k] += Δt * γ¹ * G¹.v[i, j, k]
        U.w[i, j, k] += Δt * γ¹ * G¹.w[i, j, k]
    end
end

