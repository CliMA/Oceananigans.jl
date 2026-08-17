using Adapt: Adapt
using Oceananigans.Grids: Face, Center
using Oceananigans.TimeSteppers: SplitRungeKuttaTimeStepper, SSPRungeKuttaTimeStepper,
                                 MultiStageTimeStepper

import Oceananigans: prognostic_state

#####
##### Time reconstruction of the slow forcing across the barotropic sub-cycle.
#####
##### Holding the depth-integrated slow forcing fixed over the sub-cycle caps the split-explicit scheme at
##### global order two. Reconstructing it as a quadratic in time does not lift that cap -- the Runge--Kutta
##### stage states are themselves only O(Δt²)-accurate -- but it removes a barotropic--baroclinic resonance
##### near μ₀ = k c₀ Δt ≈ 5.7, at which the frozen scheme loses about 40% of its damping.
#####

"""
    struct FrozenSlowForcing

Hold the slow forcing fixed across the barotropic sub-cycle. This is the classical split-explicit
treatment and the default.
"""
struct FrozenSlowForcing end

"""
    struct StageQuadraticSlowForcing

Reconstruct the slow forcing as the quadratic through the three baroclinic Runge--Kutta stage values,
evaluated at the midpoint of each barotropic substep and applied on the final stage only.

Requires a `SplitRungeKuttaTimeStepper`, since it is built from the stage values, and is only useful with
a multi-stage barotropic substep (`RungeKutta3Scheme`): with `ForwardBackwardScheme` the substep's own
error dominates and every treatment of the forcing gives the same answer.
"""
struct StageQuadraticSlowForcing{U, V}
    Gᵁ¹ :: U   # depth-integrated slow forcing entering stage 1, at s = 0
    Gⱽ¹ :: V
    Gᵁ² :: U   # ... entering stage 2, at s = Δt/3
    Gⱽ² :: V
end

StageQuadraticSlowForcing() = StageQuadraticSlowForcing(nothing, nothing, nothing, nothing)

materialize_slow_forcing(::FrozenSlowForcing, grid, args...) = FrozenSlowForcing()

function materialize_slow_forcing(::StageQuadraticSlowForcing, grid, u_bcs, v_bcs)
    Gᵁ¹ = Field{Face, Center, Nothing}(grid, boundary_conditions = u_bcs)
    Gⱽ¹ = Field{Center, Face, Nothing}(grid, boundary_conditions = v_bcs)
    Gᵁ² = Field{Face, Center, Nothing}(grid, boundary_conditions = u_bcs)
    Gⱽ² = Field{Center, Face, Nothing}(grid, boundary_conditions = v_bcs)
    return StageQuadraticSlowForcing(Gᵁ¹, Gⱽ¹, Gᵁ², Gⱽ²)
end

Adapt.adapt_structure(to, sf::StageQuadraticSlowForcing) =
    StageQuadraticSlowForcing(Adapt.adapt(to, sf.Gᵁ¹), Adapt.adapt(to, sf.Gⱽ¹),
                              Adapt.adapt(to, sf.Gᵁ²), Adapt.adapt(to, sf.Gⱽ²))

# The stage values are rebuilt from scratch within every time step, so there is nothing to checkpoint.
prognostic_state(::FrozenSlowForcing) = nothing
prognostic_state(::StageQuadraticSlowForcing) = nothing

"""
    stage_sample_times(timestepper)

Times, as fractions of `Δt` from the start of the step, that the state entering each Runge-Kutta stage
approximates. These are the nodes of the reconstruction, and they differ between schemes: the low-storage
form advances from `Ψⁿ` by `γ = Δt/3, Δt/2, Δt`, so its stage states sit at `0, Δt/3, Δt/2`, whereas the
Shu-Osher form takes a full step first and sits at `0, Δt, Δt/2`.
"""
@inline stage_sample_times(::SplitRungeKuttaTimeStepper) = (0.0, 1/3, 1/2)
@inline stage_sample_times(::SSPRungeKuttaTimeStepper)   = (0.0, 1.0, 1/2)

"""
    stage_quadratic_coefficients(F¹, F², F³, t₂, t₃, Δt)

Coefficients of `F(s) = F₀ + F₁ s + F₂ s²`, the quadratic through the slow forcing sampled at `s = 0`,
`t₂ Δt` and `t₃ Δt`, with `s` measured from the start of the time step.

With the low-storage nodes `(1/3, 1/2)` this reduces to
`F₁ = (-5F¹ + 9F² - 4F³)/Δt`, `F₂ = (6F¹ - 18F² + 12F³)/Δt²`.
"""
@inline function stage_quadratic_coefficients(F¹, F², F³, t₂, t₃, Δt)
    a  = t₂ * Δt
    b  = t₃ * Δt
    Δ₁ = F² - F¹
    Δ₂ = F³ - F¹
    den = a * b * (b - a)
    F₁ = ( Δ₁ * b^2 - Δ₂ * a^2) / den
    F₂ = (-Δ₁ * b   + Δ₂ * a  ) / den
    return F¹, F₁, F₂
end

# `s` is the MIDPOINT of the current barotropic substep, measured from tⁿ. Sampling at either endpoint
# instead injects an O(Δτ) error which at fixed substep count is O(Δt), and costs two orders.
@inline slow_forcing(i, j, Gᴴ, ::Nothing, ::Nothing, s, Δt, t₂, t₃) = @inbounds Gᴴ[i, j, 1]

@inline function slow_forcing(i, j, Gᴴ, Gᴴ¹, Gᴴ², s, Δt, t₂, t₃)
    @inbounds F₀, F₁, F₂ = stage_quadratic_coefficients(Gᴴ¹[i, j, 1], Gᴴ²[i, j, 1], Gᴴ[i, j, 1], t₂, t₃, Δt)
    return F₀ + F₁ * s + F₂ * s^2
end

#####
##### Capturing the stage values, and selecting when the reconstruction is active
#####

@inline cache_stage_slow_forcing!(::FrozenSlowForcing, args...) = nothing

@inline function cache_stage_slow_forcing!(sf::StageQuadraticSlowForcing, GUⁿ, GVⁿ, stage)
    if stage == 1
        parent(sf.Gᵁ¹) .= parent(GUⁿ)
        parent(sf.Gⱽ¹) .= parent(GVⁿ)
    elseif stage == 2
        parent(sf.Gᵁ²) .= parent(GUⁿ)
        parent(sf.Gⱽ²) .= parent(GVⁿ)
    end
    return nothing
end

# Called once per `step_free_surface!`, on the host. Returning either the fields or `nothing`s specializes
# the substep kernel on the two cases, so the frozen path keeps the identical `Gᴴ[i, j, 1]` load.
@inline stage_forcing_fields(::FrozenSlowForcing, args...) = (nothing, nothing, nothing, nothing)

@inline function stage_forcing_fields(sf::StageQuadraticSlowForcing, timestepper, stage)
    final_stage = timestepper isa MultiStageTimeStepper && stage == timestepper.Nstages
    return final_stage ? (sf.Gᵁ¹, sf.Gⱽ¹, sf.Gᵁ², sf.Gⱽ²) : (nothing, nothing, nothing, nothing)
end

# The reconstruction's nodes are a property of the baroclinic scheme, not of the free surface, so they
# travel with the timestepper. Anything that is not a multi-stage scheme never reaches the reconstruction.
@inline reconstruction_nodes(timestepper::MultiStageTimeStepper) = stage_sample_times(timestepper)[2:3]
@inline reconstruction_nodes(timestepper) = (1/3, 1/2)
