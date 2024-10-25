using Oceananigans.Architectures: architecture
using Oceananigans: fields

"""
    SSPRK3TimeStepper{FT, TG} <: AbstractTimeStepper

Holds parameters and tendency fields for a low storage, third-order Runge-Kutta-Wray
time-stepping scheme described by [LeMoin1991](@citet).
"""
struct SSPRK3TimeStepper{FT, TG, TI} <: AbstractTimeStepper
    γ¹ :: FT
    γ² :: FT
    γ³ :: FT
    ζ² :: FT
    ζ³ :: FT
    Gⁿ :: TG
    previous_model_fields :: TG
    implicit_solver :: TI
end

"""
    RungeKutta3TimeStepper(grid, tracers;
                            implicit_solver = nothing,
                            Gⁿ = TendencyFields(grid, tracers),
                            G⁻ = TendencyFields(grid, tracers))

Return a 3rd-order Runge0Kutta timestepper (`RungeKutta3TimeStepper`) on `grid` and with `tracers`.
The tendency fields `Gⁿ` and `G⁻` can be specified via  optional `kwargs`.

The scheme described by [LeMoin1991](@citet). In a nutshel, the 3rd-order
Runge Kutta timestepper steps forward the state `Uⁿ` by `Δt` via 3 substeps. A pressure correction
step is applied after at each substep.

The state `U` after each substep `m` is

```julia
Uᵐ⁺¹ = Uᵐ + Δt * (γᵐ * Gᵐ + ζᵐ * Gᵐ⁻¹)
```

where `Uᵐ` is the state at the ``m``-th substep, `Gᵐ` is the tendency
at the ``m``-th substep, `Gᵐ⁻¹` is the tendency at the previous substep,
and constants ``γ¹ = 8/15``, ``γ² = 5/12``, ``γ³ = 3/4``,
``ζ¹ = 0``, ``ζ² = -17/60``, ``ζ³ = -5/12``.

The state at the first substep is taken to be the one that corresponds to the ``n``-th timestep,
`U¹ = Uⁿ`, and the state after the third substep is then the state at the `Uⁿ⁺¹ = U⁴`.
"""
function SSPRK3TimeStepper(grid, tracers;
                           implicit_solver::TI = nothing,
                           Gⁿ::TG = TendencyFields(grid, tracers),
                           previous_model_fields = TendencyFields(grid, tracers)) where {TI, TG}

    !isnothing(implicit_solver) &&
        @warn("Implicit-explicit time-stepping with RungeKutta3TimeStepper is not tested. " * 
                "\n implicit_solver: $(typeof(implicit_solver))")

    γ¹ = 1
    γ² = 1 // 4
    γ³ = 2 // 3

    ζ² = 3 // 4
    ζ³ = 1 // 3

    FT = eltype(grid)

    return SSPRK3TimeStepper{FT, TG, TI}(γ¹, γ², γ³, ζ², ζ³, Gⁿ, previous_model_fields, implicit_solver)
end