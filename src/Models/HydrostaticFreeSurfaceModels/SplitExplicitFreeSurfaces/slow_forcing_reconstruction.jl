using Adapt: Adapt
using StaticArrays: SMatrix
using Oceananigans.Grids: Face, Center
using Oceananigans.TimeSteppers: SplitRungeKuttaTimeStepper, SSPRungeKuttaTimeStepper,
                                 MultiStageTimeStepper, ModifiedRungeKutta4TimeStepper

import Oceananigans: prognostic_state

#####
##### Time reconstruction of the slow forcing across the barotropic sub-cycle.
#####
##### Holding the depth-integrated slow forcing fixed over the sub-cycle caps the split-explicit scheme at
##### global order two. Reconstructing it in time does not lift that cap -- the Runge--Kutta stage states are
##### themselves only O(Δt²)-accurate -- but it removes a barotropic--baroclinic resonance near
##### μ₀ = k c₀ Δt ≈ 5.7, at which the frozen scheme loses about 40% of its damping. For the three-stage
##### composition the resonance is a matter of margin; for a four-stage composition it is fatal, and removing
##### it is what makes the fourth stage usable at all.
#####
##### Two reconstruction rules are provided. `StageQuadraticSlowForcing` fits the polynomial on the final stage
##### only, which is all the *order* can see. `ProgressiveSlowForcing` gives every stage the best polynomial its
##### already-computed stage forcings support -- constant, then linear, then quadratic -- which is what the
##### *stability* sees, since the growth constant collects the forcing error of all the stages.
#####

"""
    struct FrozenSlowForcing

Hold the slow forcing fixed across the barotropic sub-cycle. This is the classical split-explicit
treatment and the default.
"""
struct FrozenSlowForcing end

"""
    StageQuadraticSlowForcing(timestepper = SplitRungeKuttaTimeStepper(stages = 3))

Reconstruct the slow forcing as the quadratic through the baroclinic Runge--Kutta stage values, evaluated at
the midpoint of each barotropic substep and applied on the final stage only. Every earlier stage keeps the
frozen forcing.

The reconstruction is built for `timestepper`, whose composition fixes both the number of stage forcings to
store -- the stage in progress reads its forcing live, so `Nstages - 1` fields per horizontal velocity
component -- and the nodes at which they were sampled. Pairing it with a different composition in the model is
an error rather than a silent misfit.

Only useful with a multi-stage barotropic substep (`RungeKutta3Scheme`): with `ForwardBackwardScheme` the
substep's own error dominates and every treatment of the forcing gives the same answer.
"""
struct StageQuadraticSlowForcing{U, V, W, F, T}
    Gᵁ :: U   # the stage forcings preceding the current stage, one field each
    Gⱽ :: V
    weights :: W   # dimensionless reconstruction weights, one entry per stage, all of one type
    frozen :: F    # whether each stage's polynomial is a constant, and so takes the frozen path
    nodes :: T     # the composition's sample times, kept to check the pairing
end

"""
    ProgressiveSlowForcing(timestepper = ModifiedRungeKutta4TimeStepper())

Reconstruct the slow forcing *progressively*: each stage carries the best polynomial the already-computed
stage forcings support -- the first stage frozen, the second linear, the third and any later stage
quadratic -- at no additional forcing evaluations and with the same stored fields as
`StageQuadraticSlowForcing`.

This is the reconstruction the four-stage composition requires, hence the default. Only the final stage sets
the order, so for accuracy alone `StageQuadraticSlowForcing` is equivalent; the intermediate stages matter for
*stability*, because the perturbation the sub-cycle injects into the first baroclinic mode collects the
forcing error of every stage. Freezing the intermediate stages raises that injection by two orders of
magnitude, which is more than any four-stage stability polynomial can absorb while still paying for its fourth
stage.

See also [`ModifiedRungeKutta4TimeStepper`](@ref), the four-stage composition designed around this reconstruction.
"""
struct ProgressiveSlowForcing{U, V, W, F, T}
    Gᵁ :: U
    Gⱽ :: V
    weights :: W
    frozen :: F
    nodes :: T
end

const ReconstructedSlowForcing = Union{StageQuadraticSlowForcing, ProgressiveSlowForcing}

StageQuadraticSlowForcing(timestepper = SplitRungeKuttaTimeStepper(stages = 3)) = build_reconstruction(StageQuadraticSlowForcing, timestepper)

ProgressiveSlowForcing(timestepper = ModifiedRungeKutta4TimeStepper()) = build_reconstruction(ProgressiveSlowForcing, timestepper)

function build_reconstruction(SF, timestepper)
    M = timestepper.Nstages
    M > 1 || throw(ArgumentError("A slow-forcing reconstruction needs at least two stages, got $M."))

    τ = stage_sample_times(timestepper)
    N = M - 1

    degrees = Tuple(reconstruction_degree(SF, m, M) for m in 1:M)

    # One entry per stage, all of the same type so that indexing the table with the running stage stays
    # type-stable: a stage whose polynomial is a constant stores zeros and is flagged instead, since it takes
    # the frozen path in the kernel and never reads them. (It could not fit anything anyway: the first stage
    # has a single sample.)
    weights = Tuple(degrees[m] == 1 ? zero_reconstruction_weights(Val(N), Float64) :
                    dimensionless_reconstruction_weights(τ[1:m], Val(degrees[m]), Val(N), Float64)
                    for m in 1:M)

    frozen = Tuple(D == 1 for D in degrees)

    return SF(Tuple(nothing for n in 1:N), Tuple(nothing for n in 1:N), weights, frozen, τ)
end

@inline cached_stages(sf::ReconstructedSlowForcing) = length(sf.Gᵁ)
@inline maximum_reconstruction_degree(::Type{<:ReconstructedSlowForcing}) = 3

materialize_slow_forcing(::FrozenSlowForcing, grid, args...) = FrozenSlowForcing()

function materialize_slow_forcing(sf::ReconstructedSlowForcing, grid, u_bcs, v_bcs)
    N  = cached_stages(sf)
    Gᵁ = ntuple(n -> Field{Face, Center, Nothing}(grid, boundary_conditions = u_bcs), N)
    Gⱽ = ntuple(n -> Field{Center, Face, Nothing}(grid, boundary_conditions = v_bcs), N)
    return Base.typename(typeof(sf)).wrapper(Gᵁ, Gⱽ, sf.weights, sf.frozen, sf.nodes)
end

Adapt.adapt_structure(to, sf::ReconstructedSlowForcing) =
    Base.typename(typeof(sf)).wrapper(Adapt.adapt(to, sf.Gᵁ),
                                      Adapt.adapt(to, sf.Gⱽ),
                                      Adapt.adapt(to, sf.weights),
                                      Adapt.adapt(to, sf.frozen),
                                      Adapt.adapt(to, sf.nodes))

# The stage values are rebuilt from scratch within every time step, so there is nothing to checkpoint.
prognostic_state(::FrozenSlowForcing) = nothing
prognostic_state(::ReconstructedSlowForcing) = nothing

#####
##### The nodes: where in the step each stage forcing was sampled
#####

"""
    stage_sample_times(timestepper)

Times, as fractions of `Δt` from the start of the step, that the state entering each Runge-Kutta stage approximates. These are the nodes
of the reconstruction, and they are a property of the composition: the low-storage form advances from `Ψⁿ` by `γₘ = Δt / βₘ`, so the state
entering stage `m` approximates `γₘ₋₁`, whereas the Shu-Osher form takes a full step first and sits at `0, Δt, Δt/2`.

For `β = (3, 2, 1)` this returns `(0, 1/3, 1/2)`, and for the four-stage `β = (1/a, 3, 2, 1)` of [`ModifiedRungeKutta4TimeStepper`](@ref) it returns
`(0, a, 1/3, 1/2)` -- the leading node tracks the composition rather than being carried as a separate constant.
"""
@inline stage_sample_times(ts::SplitRungeKuttaTimeStepper{<:NTuple{M, Any}}) where M = (0.0, ntuple(m -> 1 / ts.β[m], Val(M - 1))...)
@inline stage_sample_times(::SSPRungeKuttaTimeStepper) = (0.0, 1.0, 1/2)

#####
##### The rule: how many coefficients each stage's polynomial carries
#####

@inline reconstruction_degree(SF::Type{<:ProgressiveSlowForcing},    stage, Nstages) = min(stage, maximum_reconstruction_degree(SF))
@inline reconstruction_degree(SF::Type{<:StageQuadraticSlowForcing}, stage, Nstages) = stage == Nstages ? min(Nstages, maximum_reconstruction_degree(SF)) : 1
@inline reconstruction_degree(sf::ReconstructedSlowForcing,          stage, Nstages) = reconstruction_degree(Base.typename(typeof(sf)).wrapper, stage, Nstages)

#####
##### The weights, built once per stage on the host
#####

# Square node set: the polynomial passes through every sample. Rectangular: more samples than coefficients, so
# the polynomial is the least-squares fit through them (this is the final stage of a four-stage composition).
@inline invert_vandermonde(V::SMatrix{N, N}) where N = inv(V)
@inline invert_vandermonde(V::SMatrix{M, D}) where {M, D} = inv(V' * V) * V'

"""
    dimensionless_reconstruction_weights(τ, Val(D), Val(N), FT)

Weights `w` of the polynomial `F(σ) = F₀ + F₁ σ + F₂ σ²` through the slow forcing sampled at the `M`
dimensionless nodes `τ`, with `σ = s/Δt`, laid out in the slots the substep kernel reads:

    Fₖ₋₁ = w[k][1] G¹ + … + w[k][N] Gᴺ + w[k][N+1] Gᴹ

The first `N` slots are the cached stage forcings and the last is the live forcing of the current stage, which
is always the final sample. Slots beyond the samples the stage has carry a zero weight, so the tuple length is
fixed by the composition rather than by the stage, and the substep kernel compiles to a single specialization.

Nothing here depends on `Δt`: the nodes are `τⱼ Δt`, so the fit factors into a shape fixed by the composition
and a scaling `Δt^{-(k-1)}` on the `k`-th coefficient. The shape is built once, when the reconstruction is
constructed, and [`scale_reconstruction_weights`](@ref) restores the units at each stage.
"""
function dimensionless_reconstruction_weights(τ::NTuple{M, Any}, ::Val{D}, ::Val{N}, ::Type{FT}) where {M, D, N, FT}
    V = SMatrix{M, D, FT}(ntuple(n -> FT(τ[mod1(n, M)])^(cld(n, M) - 1), Val(M * D)))
    X = invert_vandermonde(V)   # D × M

    # Sample M is the live forcing rather than a cache, so it lands in the last slot whatever the stage.
    @inline weight(k, j) = j == N + 1 ? X[k, M] : (j < M ? X[k, j] : zero(FT))

    return ntuple(Val(3)) do k
        k > D ? ntuple(j -> zero(FT), Val(N + 1)) : ntuple(j -> FT(weight(k, j)), Val(N + 1))
    end
end

"""
    scale_reconstruction_weights(w, Δt)

Restore the units of the stored dimensionless weights: the `k`-th coefficient of `F(s) = F₀ + F₁ s + F₂ s²`
carries `1/time^(k-1)`, so `Δt` enters only through that power. This is the whole per-stage cost of the
reconstruction -- a few multiplies on stack-resident tuples, with no fit and no matrix.
"""
@inline scale_reconstruction_weights(w::NTuple{3, Any}, Δt) = ntuple(k -> map(x -> x / Δt^(k - 1), w[k]), Val(3))

#####
##### Evaluation inside the substep kernel
#####

# `s` is the MIDPOINT of the current barotropic substep, measured from tⁿ. Sampling at either endpoint
# instead injects an O(Δτ) error which at fixed substep count is O(Δt), and costs two orders.
@inline slow_forcing(i, j, Gᴴ, ::Nothing, w, s) = @inbounds Gᴴ[i, j, 1]

@inline function slow_forcing(i, j, Gᴴ, Gᶜ::NTuple{N}, w, s) where N
    @inbounds begin
        Gᴹ = Gᴴ[i, j, 1]
        F₀ = reconstruction_sum(w[1], Gᶜ, Gᴹ, i, j, Val(N))
        F₁ = reconstruction_sum(w[2], Gᶜ, Gᴹ, i, j, Val(N))
        F₂ = reconstruction_sum(w[3], Gᶜ, Gᴹ, i, j, Val(N))
    end

    return F₀ + F₁ * s + F₂ * s^2
end

@inline reconstruction_sum(w, Gᶜ, Gᴹ, i, j, ::Val{N}) where N =
    @inbounds w[N+1] * Gᴹ + sum(ntuple(n -> w[n] * Gᶜ[n][i, j, 1], Val(N)))

#####
##### Capturing the stage values, and selecting when the reconstruction is active
#####

@inline cache_stage_slow_forcing!(::FrozenSlowForcing, args...) = nothing

@inline function cache_stage_slow_forcing!(sf::ReconstructedSlowForcing, GUⁿ, GVⁿ, stage)
    if stage ≤ cached_stages(sf)
        parent(sf.Gᵁ[stage]) .= parent(GUⁿ)
        parent(sf.Gⱽ[stage]) .= parent(GVⁿ)
    end
    return nothing
end

@inline zero_reconstruction_weights(::Val{N}, ::Type{FT}) where {N, FT} =
    ntuple(k -> ntuple(j -> zero(FT), Val(N + 1)), Val(3))

# Called once per `step_free_surface!`, on the host. Returning `nothing` for the caches specializes the substep
# kernel on the frozen case, which keeps the identical `Gᴴ[i, j, 1]` load; this covers every stage of
# `FrozenSlowForcing`, the intermediate stages of `StageQuadraticSlowForcing`, and the first stage of
# `ProgressiveSlowForcing`, all of which carry a single coefficient.
@inline stage_reconstruction(::FrozenSlowForcing, timestepper, stage, Δt::FT) where FT = (nothing, nothing, zero_reconstruction_weights(Val(1), FT))

# Anything that is not a multi-stage scheme has no stage values to reconstruct from.
@inline stage_reconstruction(sf::ReconstructedSlowForcing, timestepper, stage, Δt::FT) where FT = (nothing, nothing, zero_reconstruction_weights(Val(length(sf.Gᵁ)), FT))

@inline function stage_reconstruction(sf::ReconstructedSlowForcing, ts::MultiStageTimeStepper, stage, Δt::FT) where FT
    @inbounds sf.frozen[stage] && return (nothing, nothing, zero_reconstruction_weights(Val(length(sf.Gᵁ)), FT))
    @inbounds w̃ = sf.weights[stage]
    return (sf.Gᵁ, sf.Gⱽ, scale_reconstruction_weights(w̃, Δt))
end
