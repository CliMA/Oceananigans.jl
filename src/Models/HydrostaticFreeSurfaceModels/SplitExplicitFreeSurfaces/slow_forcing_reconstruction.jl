using Adapt: Adapt
using StaticArrays: SMatrix
using Oceananigans.Grids: Face, Center
using Oceananigans.TimeSteppers: SplitRungeKuttaTimeStepper, SSPRungeKuttaTimeStepper,
                                 MultiStageTimeStepper, ModifiedRungeKutta4TimeStepper

import Oceananigans: prognostic_state

#####
##### Time reconstruction of the slow forcing across the barotropic sub-cycle.
#####
##### Reconstructing the depth-integrated slow forcing in time does not lift the order-two cap of the
##### split-explicit scheme, but it removes the barotropic-baroclinic resonance near μ₀ = k c₀ Δt ≈ 5.7 at
##### which the frozen forcing loses about 40% of its damping.
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

The `timestepper` fixes the nodes at which the stage forcings are sampled and how many of them are stored:
the stage in progress reads its forcing live, so `Nstages - 1` fields per horizontal velocity component.
"""
struct StageQuadraticSlowForcing{U, V, W, F, T}
    Gᵁ :: U        # the stage forcings preceding the current stage, one field each
    Gⱽ :: V
    weights :: W   # dimensionless reconstruction weights, one entry per stage, all of one type
    frozen :: F    # whether each stage's polynomial is a constant, and so takes the frozen path
    nodes :: T     # the composition's sample times
end

"""
    ProgressiveSlowForcing(timestepper = ModifiedRungeKutta4TimeStepper())

Reconstruct the slow forcing *progressively*: each stage carries the best polynomial the already-computed
stage forcings support -- the first stage frozen, the second linear, the third and any later stage
quadratic -- with the same stored fields as `StageQuadraticSlowForcing`.

The intermediate stages do not change the order but they do change the stability, because the perturbation
the sub-cycle injects into the first baroclinic mode collects the forcing error of every stage. This is the
reconstruction [`ModifiedRungeKutta4TimeStepper`](@ref) requires.
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

    # All stages carry the same weight type so that indexing with the running stage stays type-stable; a
    # stage whose polynomial is a constant stores zeros and is flagged to take the frozen path instead.
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

# The stage values are rebuilt within every time step.
prognostic_state(::FrozenSlowForcing) = nothing
prognostic_state(::ReconstructedSlowForcing) = nothing

#####
##### The nodes: where in the step each stage forcing was sampled
#####

"""
    stage_sample_times(timestepper)

Times, as fractions of `Δt` from the start of the step, that the state entering each Runge-Kutta stage
approximates. The low-storage form advances from `Ψⁿ` by `γₘ = Δt / βₘ`, so the state entering stage `m`
approximates `γₘ₋₁` and `β = (3, 2, 1)` gives `(0, 1/3, 1/2)`; the Shu-Osher form sits at `(0, 1, 1/2)`.
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

# Square node set: the polynomial passes through every sample. Rectangular: least-squares fit.
@inline invert_vandermonde(V::SMatrix{N, N}) where N = inv(V)
@inline invert_vandermonde(V::SMatrix{M, D}) where {M, D} = inv(V' * V) * V'

"""
    dimensionless_reconstruction_weights(τ, Val(D), Val(N), FT)

Weights `w` of the polynomial `F(σ) = F₀ + F₁ σ + F₂ σ²` through the slow forcing sampled at the `M`
dimensionless nodes `τ`, with `σ = s/Δt`, laid out in the slots the substep kernel reads:

    Fₖ₋₁ = w[k][1] G¹ + … + w[k][N] Gᴺ + w[k][N+1] Gᴹ

The first `N` slots are the cached stage forcings and the last is the live forcing of the current stage, which
is always the final sample. Slots beyond the samples the stage has carry a zero weight, so the tuple length is
fixed by the composition rather than by the stage.

The nodes are `τⱼ Δt`, so the fit factors into a shape fixed by the composition and a scaling `Δt^{-(k-1)}`
on the `k`-th coefficient; the shape is built once and rescaled at each stage.
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
carries `1/time^(k-1)`, so `Δt` enters only through that power.
"""
@inline scale_reconstruction_weights(w::NTuple{3, Any}, Δt) = ntuple(k -> map(x -> x / Δt^(k - 1), w[k]), Val(3))

#####
##### Evaluation inside the substep kernel
#####

# `s` is the midpoint of the current barotropic substep, measured from tⁿ: sampling at either endpoint
# injects an O(Δτ) error and costs two orders.
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

# `nothing` caches specialize the substep kernel on the frozen case, keeping the plain `Gᴴ[i, j, 1]` load.
@inline stage_reconstruction(::FrozenSlowForcing, timestepper, stage, Δt::FT) where FT = (nothing, nothing, zero_reconstruction_weights(Val(1), FT))

# Anything that is not a multi-stage scheme has no stage values to reconstruct from.
@inline stage_reconstruction(sf::ReconstructedSlowForcing, timestepper, stage, Δt::FT) where FT = (nothing, nothing, zero_reconstruction_weights(Val(length(sf.Gᵁ)), FT))

@inline function stage_reconstruction(sf::ReconstructedSlowForcing, ts::MultiStageTimeStepper, stage, Δt::FT) where FT
    @inbounds sf.frozen[stage] && return (nothing, nothing, zero_reconstruction_weights(Val(length(sf.Gᵁ)), FT))
    @inbounds w̃ = sf.weights[stage]
    return (sf.Gᵁ, sf.Gⱽ, scale_reconstruction_weights(w̃, Δt))
end
