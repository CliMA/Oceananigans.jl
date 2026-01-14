using Oceananigans.Architectures: architecture
using Oceananigans.Utils: configure_kernel
using Oceananigans.TimeSteppers: _rk3_substep_field!

import Oceananigans.TimeSteppers: rk3_substep!

"""
    rk3_substep!(model::ShallowWaterModel, Δt, γⁿ, ζⁿ, callbacks)

Perform a single RK3 substep for `ShallowWaterModel`.

Advances the solution fields (`uh`, `vh`, `h`) and any tracers using the RK3 scheme:
`U += Δt * (γⁿ * Gⁿ + ζⁿ * G⁻)`

where `Gⁿ` and `G⁻` are the current and previous tendencies.
"""
function rk3_substep!(model::ShallowWaterModel, Δt, γⁿ, ζⁿ, callbacks)

    compute_tendencies!(model, callbacks)
    grid = model.grid

    launch!(architecture(grid), grid, :xyz, _rk_substep_solution!, 
            model.solution,
            Δt, γⁿ, ζⁿ,
            model.timestepper.Gⁿ,
            model.timestepper.G⁻)

    _tracer_kernel!, _ = configure_kernel(architecture(grid), grid, :xyz, _rk3_substep_field!)

    for i in 1:length(model.tracers)
        @inbounds c = model.tracers[i]
        @inbounds Gcⁿ = model.timestepper.Gⁿ[i+3]
        @inbounds Gc⁻ = model.timestepper.G⁻[i+3]

        _tracer_kernel!(c, Δt, γⁿ, ζⁿ, Gcⁿ, Gc⁻)
    end

    return nothing
end

"""
Time step solution fields with a 3rd-order Runge-Kutta method.
"""
@kernel function _rk_substep_solution!(U, Δt, γⁿ, ζⁿ, Gⁿ, G⁻)
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
@kernel function _rk_substep_solution!(U, Δt, γ¹, ::Nothing, G¹, G⁰)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        U[1][i, j, k] += Δt * γ¹ * G¹[1][i, j, k]
        U[2][i, j, k] += Δt * γ¹ * G¹[2][i, j, k]
        U[3][i, j, k] += Δt * γ¹ * G¹[3][i, j, k]
    end
end
