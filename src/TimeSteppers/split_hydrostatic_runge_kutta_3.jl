using Oceananigans.Architectures: architecture
using Oceananigans: fields

"""
    SplitRungeKutta3TimeStepper{FT, TG, PF, TI} <: AbstractTimeStepper

Hold parameters and tendency fields for a low storage, third-order Runge-Kutta-Wray
time-stepping scheme described by [Lan et al. (2022)](@cite Lan2022).
"""
struct SplitRungeKutta3TimeStepper{FT, TG, PF, TI} <: AbstractTimeStepper
    β¹ :: FT
    β² :: FT
    Gⁿ :: TG
    Ψ⁻ :: PF # prognostic state at the previous timestep
    implicit_solver :: TI
end

"""
    SplitRungeKutta3TimeStepper(grid, prognostic_fields, args...;
                                implicit_solver::TI = nothing,
                                Gⁿ::TG = map(similar, prognostic_fields),
                                Ψ⁻::PF = map(similar, prognostic_fields),
                                kwargs...) where {TI, TG, PF}

Return a 3rd-order `SplitRungeKutta3TimeStepper` on `grid` and with `tracers`.
The tendency fields `Gⁿ` and `G⁻`, and the previous state `Ψ⁻` can be modified
via optional `kwargs`.

The scheme is described by [Knoth and Wensch (2014)](@cite knoth2014). In a nutshell,
the 3rd-order Runge-Kutta timestepper steps forward the state `Uⁿ` by `Δt` via 3 substeps.
A barotropic velocity correction step is applied after at each substep.

The state `U` after each substep `m` is equivalent to an Euler step with a modified time step:

```julia
Δt̃   = Δt / βᵐ
Uᵐ⁺¹ = Uⁿ + Δt̃ * Gᵐ
```

where `Uᵐ` is the state at the ``m``-th substep, `Uⁿ` is the state at the ``n``-th timestep,
`Gᵐ` is the tendency at the ``m``-th substep, and constants `β¹ = 3`, `β² = 2`, `β³ = 1`.

The state at the first substep is taken to be the one that corresponds to the ``n``-th timestep,
`U¹ = Uⁿ`, and the state after the third substep is then the state at the `Uⁿ⁺¹ = U³`.

References
==========

Knoth, O., and Wensch, J. (2014). Generalized Split-Explicit Runge-Kutta Methods for the
    compressible Euler equations. Monthly Weather Review, 142, 2067-2081,
    https://doi.org/10.1175/MWR-D-13-00068.1.
"""
function SplitRungeKutta3TimeStepper(grid, prognostic_fields, args...;
                                     implicit_solver::TI = nothing,
                                     Gⁿ::TG = map(similar, prognostic_fields),
                                     Ψ⁻::PF = map(similar, prognostic_fields),
                                     kwargs...) where {TI, TG, PF}

    @warn("Split barotropic-baroclinic time stepping with SplitRungeKutta3TimeStepper is and experimental.\n" *
          "Use at own risk, and report any issues encountered at [https://github.com/CliMA/Oceananigans.jl/issues](https://github.com/CliMA/Oceananigans.jl/issues).")

    FT = eltype(grid)
    β¹ = 3
    β² = 2

    return SplitRungeKutta3TimeStepper{FT, TG, PF, TI}(β¹, β², Gⁿ, Ψ⁻, implicit_solver)
end

@kernel function _euler_substep_field!(field, Δt, Gⁿ, Ψ⁻)
    i, j, k = @index(Global, NTuple)
    @inbounds field[i, j, k] = Ψ⁻[i, j, k] + Δt * Gⁿ[i, j, k]
end