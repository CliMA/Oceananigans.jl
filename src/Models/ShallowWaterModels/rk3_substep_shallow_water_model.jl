using Oceananigans.Utils: work_layout
using Oceananigans.Architectures: device
using Oceananigans.TimeSteppers: rk3_substep_field!

import Oceananigans.TimeSteppers: rk3_substep!

function rk3_substep!(model::ShallowWaterModel, Δt, γⁿ, ζⁿ)

    workgroup, worksize = work_layout(model.grid, :xyz)

    substep_solution_kernel! = rk3_substep_solution!(device(model.architecture), workgroup, worksize)
    substep_tracer_kernel! = rk3_substep_tracer!(device(model.architecture), workgroup, worksize)


    solution_event = substep_solution_kernel!(model.solution,
                                              Δt, γⁿ, ζⁿ,
                                              model.timestepper.Gⁿ,
                                              model.timestepper.G⁻)


    for i in 1:length(model.tracers)
        @inbounds c = model.tracers[i]
        @inbounds Gcⁿ = model.timestepper.Gⁿ[i+3]
        @inbounds Gc⁻ = model.timestepper.G⁻[i+3]

        substep_tracer_kernel!(c, Δt, γⁿ, ζⁿ, Gcⁿ, Gc⁻)
    end


    return nothing
end

"""
Time step solution fields with a 3rd-order Runge-Kutta method.
"""
@kernel function rk3_substep_solution!(U, Δt, γⁿ, ζⁿ, Gⁿ, G⁻)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        U[1][i, j, k] += Δt * (γⁿ * Gⁿ[1][i, j, k] + ζⁿ * G⁻[1][i, j, k])
        U[2][i, j, k] += Δt * (γⁿ * Gⁿ[2][i, j, k] + ζⁿ * G⁻[2][i, j, k])
        U[3][i, j, k] += Δt * (γⁿ * Gⁿ[3][i, j, k] + ζⁿ * G⁻[3][i, j, k])
    end
end

"""
Time step solution fields with a 3rd-order Runge-Kutta method.
"""
@kernel function rk3_substep_solution!(U, Δt, γ¹, ::Nothing, G¹, G⁰)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        U[1][i, j, k] += Δt * γ¹ * G¹[1][i, j, k]
        U[2][i, j, k] += Δt * γ¹ * G¹[2][i, j, k]
        U[3][i, j, k] += Δt * γ¹ * G¹[3][i, j, k]
    end
end
