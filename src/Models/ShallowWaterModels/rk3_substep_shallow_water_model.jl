using Oceananigans.Utils: work_layout
using Oceananigans.Architectures: device
using Oceananigans.TimeSteppers: rk3_substep_field!

import Oceananigans.TimeSteppers: rk3_substep!

function rk3_substep!(model::ShallowWaterModel, Δt, γⁿ, ζⁿ)

    workgroup, worksize = work_layout(model.grid, :xyz)

    barrier = Event(device(model.architecture))

    substep_solution_kernel! = rk3_substep_solution!(device(model.architecture), workgroup, worksize)
    substep_tracer_kernel! = rk3_substep_tracer!(device(model.architecture), workgroup, worksize)


    solution_event = substep_solution_kernel!(model.solution,
                                              Δt, γⁿ, ζⁿ,
                                              model.timestepper.Gⁿ,
                                              model.timestepper.G⁻;
                                              dependencies=barrier)

    events = [solution_event]

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
Time step solution fields with a 3rd-order Runge-Kutta method.
"""
@kernel function rk3_substep_solution!(U, Δt, γⁿ, ζⁿ, Gⁿ, G⁻)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        U.uh[i, j, k] += Δt * (γⁿ * Gⁿ.uh[i, j, k] + ζⁿ * G⁻.uh[i, j, k])
        U.vh[i, j, k] += Δt * (γⁿ * Gⁿ.vh[i, j, k] + ζⁿ * G⁻.vh[i, j, k])
        U.h[i, j, k]  += Δt * (γⁿ * Gⁿ.h[i, j, k]  + ζⁿ * G⁻.h[i, j, k])
    end
end

"""
Time step solution fields with a 3rd-order Runge-Kutta method.
"""
@kernel function rk3_substep_solution!(U, Δt, γ¹, ::Nothing, G¹, G⁰)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        U.uh[i, j, k] += Δt * γ¹ * G¹.uh[i, j, k]
        U.vh[i, j, k] += Δt * γ¹ * G¹.vh[i, j, k]
        U.h[i, j, k]  += Δt * γ¹ * G¹.h[i, j, k]
    end
end
