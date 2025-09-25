using Oceananigans.Architectures: architecture
using Oceananigans: fields

"""
    SplitRungeKuttaTimeStepper{FT, TG, PF, TI} <: AbstractTimeStepper

Hold parameters and tendency fields for a low storage, nth-order Runge-Kutta time-stepping scheme
"""
struct SplitRungeKuttaTimeStepper{B, TG, PF, TI} <: AbstractTimeStepper
    β  :: B
    Gⁿ :: TG
    Ψ⁻ :: PF # prognostic state at the previous timestep
    implicit_solver :: TI
end

"""
    SplitRungeKuttaTimeStepper(grid, prognostic_fields, args...;
                               implicit_solver::TI = nothing,
                               Gⁿ::TG = map(similar, prognostic_fields),
                               Ψ⁻::PF = map(similar, prognostic_fields),
                               kwargs...) where {TI, TG, PF}

Return a nth-order `SplitRungeKuttaTimeStepper` on `grid` and with `tracers`.
The tendency fields `Gⁿ` and `G⁻`, and the previous state `Ψ⁻` can be modified
via optional `kwargs`.

The scheme is described by [Knoth and Wensch (2014)](@cite knoth2014). In a nutshell,
the nth-order low-storage Runge-Kutta timestepper steps forward the state `Uⁿ` by `Δt` via n substeps.
A barotropic velocity correction step is applied after at each substep.

The state `U` after each substep `m` is equivalent to an Euler step with a modified time step:

```julia
Δt̃   = Δt / βᵐ
Uᵐ⁺¹ = Uⁿ + Δt̃ * Gᵐ
```

where `Uᵐ` is the state at the ``m``-th substep, `Uⁿ` is the state at the ``n``-th timestep,
`Gᵐ` is the tendency at the ``m``-th substep. The coefficients `β` can be specified by the user,
and default to `(3, 2, 1)` for a three-stage scheme. The number of stages is inferred from the length of the
`β` tuple.

The state at the first substep is taken to be the one that corresponds to the ``n``-th timestep,
`U¹ = Uⁿ`, and the state after the third substep is then the state at the `Uⁿ⁺¹ = U³`.

References
==========

Knoth, O., and Wensch, J. (2014). Generalized Split-Explicit Runge-Kutta methods for the
    compressible Euler equations. Monthly Weather Review, 142, 2067-2081,
    https://doi.org/10.1175/MWR-D-13-00068.1.
"""
function SplitRungeKuttaTimeStepper(grid, prognostic_fields, args...;
                                    implicit_solver::TI = nothing,
                                    coefficients = (3, 2, 1),
                                    Gⁿ::TG = map(similar, prognostic_fields),
                                    Ψ⁻::PF = map(similar, prognostic_fields),
                                    kwargs...) where {TI, TG, PF}

    @warn("Split barotropic-baroclinic time stepping with SplitRungeKuttaTimeStepper is experimental.\n" *
          "Use at own risk, and report any issues encountered at [https://github.com/CliMA/Oceananigans.jl/issues](https://github.com/CliMA/Oceananigans.jl/issues).")

    return SplitRungeKuttaTimeStepper{typeof(coefficients), TG, PF, TI}(coefficients, Gⁿ, Ψ⁻, implicit_solver)
end

# Simple constructor that only requires only the coefficients or the number of stages
function SplitRungeKuttaTimeStepper(; coefficients = nothing, stages = nothing) 
    if coefficients !== nothing && stages !== nothing
        error("Cannot specify both `coefficients` and `stages`.")
    end
    if coefficients == nothing 
        coefficients = tuple(collect(stages:-1:1)...)
    end
    return SplitRungeKuttaTimeStepper{typeof(coefficients), Nothing, Nothing, Nothing}(coefficients, nothing, nothing, nothing)
end

# Utility to compute low-storage coefficients from spectral coefficients. This is
# useful to minimize dispersion and dissipation errors:
# see Hu et al., Low-Dissipation and Low-Dispersion Runge–Kutta Schemes for Computational Acoustics, 1996
function spectral_coefficients(c::AbstractVector)
    N = length(c)
    b = similar(c)
    for i in 1:N-1
        b[i] = c[N - i] / c[N - i + 1] 
    end
    b[end] = 1
    return tuple(b...)
end

cache_previous_fields!(model) = nothing

function time_step!(model::AbstractModel{<:SplitRungeKuttaTimeStepper}, Δt; callbacks=[])
    Δt == 0 && @warn "Δt == 0 may cause model blowup!"

    if model.clock.iteration == 0
        update_state!(model, callbacks)
    end

    cache_previous_fields!(model)
    grid = model.grid

    ####
    #### Loop over the stages
    ####

    for (stage, β) in enumerate(model.timestepper.β)
        # Update the clock stage
        model.clock.stage = stage
        
        # Perform the substep
        rk_substep!(model, grid, Δt / β, callbacks)

        # Update the state
        update_state!(model, callbacks)
    end
    
    # Finalize step
    step_lagrangian_particles!(model, Δt)
    tick!(model.clock, Δt)

    return nothing
end
