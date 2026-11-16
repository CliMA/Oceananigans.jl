using Adapt: Adapt
using StaticArrays: SMatrix
using Oceananigans.Grids: Face, Center
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ
using Oceananigans.TimeSteppers: SplitRungeKuttaTimeStepper, SSPRungeKuttaTimeStepper,
                                 MultiStageTimeStepper, ssp_quadrature_weights, ssp_stage_fractions

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
    HydrostaticPressureGradient()

Select the depth-integrated hydrostatic pressure gradient, `-∫ ∇ₕ p′ dz`, as the only part of the slow forcing
that [`ProgressiveSlowForcing`](@ref) reconstructs in time.
"""
struct HydrostaticPressureGradient{U, V}
    Pᵁ :: U   # the depth-integrated pressure gradient of the current stage
    Pⱽ :: V
end

HydrostaticPressureGradient() = HydrostaticPressureGradient(nothing, nothing)

Adapt.adapt_structure(to, pg::HydrostaticPressureGradient) = HydrostaticPressureGradient(Adapt.adapt(to, pg.Pᵁ), Adapt.adapt(to, pg.Pⱽ))

"""
    ProgressiveSlowForcing(timestepper = SplitRungeKuttaTimeStepper(stages = 3); reconstructed_terms = :all)

Reconstruct the slow forcing *progressively*: each stage carries the best polynomial the already-computed
stage forcings support -- the first stage frozen, the second linear, the third and any later stage
quadratic -- evaluated at the midpoint of each barotropic substep.

The `timestepper` must match the one that steps the model. It fixes the nodes at which the stage forcings are
sampled and how many of them are stored: the stage in progress reads its forcing live, so `Nstages - 1` fields
per horizontal velocity component.

`reconstructed_terms` selects what the polynomial is built from:

- `:all` reconstructs the whole depth-integrated slow forcing.
- `:pressure_gradient` reconstructs the depth-integrated hydrostatic pressure gradient and holds the rest of
  the forcing (Coriolis, advection, stresses, drag) at its value at the current stage, as `FrozenSlowForcing`
  does. Those terms are proportional to the barotropic velocity, and the polynomial of the final stage is
  evaluated beyond its last node, up to the end of the averaging window, where it amplifies their fast part.

The intermediate stages do not change the order but they do change the stability, because the perturbation
the sub-cycle injects into the first baroclinic mode collects the forcing error of every stage.
[`ModifiedRungeKutta4TimeStepper`](@ref) requires `ProgressiveSlowForcing(ModifiedRungeKutta4TimeStepper())`.
"""
struct ProgressiveSlowForcing{U, V, P, W, F, T}
    Gᵁ :: U                 # the reconstructed stage forcings preceding the current stage, one field each
    Gⱽ :: V
    pressure_gradient :: P  # `nothing` when the whole forcing is reconstructed
    weights :: W            # dimensionless reconstruction weights, one entry per stage, all of one type
    frozen :: F             # whether each stage's polynomial is a constant, and so takes the frozen path
    nodes :: T              # the composition's sample times
end

reconstructed_terms_selection(terms) =
    terms === :all               ? nothing :
    terms === :pressure_gradient ? HydrostaticPressureGradient() :
    throw(ArgumentError("reconstructed_terms must be :all or :pressure_gradient, got $terms."))

function ProgressiveSlowForcing(timestepper = SplitRungeKuttaTimeStepper(stages = 3); reconstructed_terms = :all)
    pressure_gradient = reconstructed_terms_selection(reconstructed_terms)
    M = timestepper.Nstages
    M > 1 || throw(ArgumentError("A slow-forcing reconstruction needs at least two stages, got $M."))

    τ = stage_sample_times(timestepper)
    N = M - 1

    degrees = Tuple(reconstruction_degree(m) for m in 1:M)

    # All stages carry the same weight type so that indexing with the running stage stays type-stable; a
    # stage whose polynomial is a constant stores zeros and is flagged to take the frozen path instead.
    weights = Tuple(degrees[m] == 1 ? zero_reconstruction_weights(Val(N), Float64) :
                    dimensionless_reconstruction_weights(τ[1:m], Val(degrees[m]), Val(N), Float64)
                    for m in 1:M)

    frozen = Tuple(D == 1 for D in degrees)

    return ProgressiveSlowForcing(Tuple(nothing for n in 1:N), Tuple(nothing for n in 1:N), pressure_gradient,
                                  weights, frozen, τ)
end

@inline cached_stages(sf::ProgressiveSlowForcing) = length(sf.Gᵁ)

materialize_slow_forcing(::FrozenSlowForcing, grid, args...) = FrozenSlowForcing()

function materialize_slow_forcing(sf::ProgressiveSlowForcing, grid, u_bcs, v_bcs)
    N  = cached_stages(sf)
    Gᵁ = ntuple(n -> Field{Face, Center, Nothing}(grid, boundary_conditions = u_bcs), N)
    Gⱽ = ntuple(n -> Field{Center, Face, Nothing}(grid, boundary_conditions = v_bcs), N)
    pressure_gradient = materialize_pressure_gradient(sf.pressure_gradient, grid, u_bcs, v_bcs)
    return ProgressiveSlowForcing(Gᵁ, Gⱽ, pressure_gradient, sf.weights, sf.frozen, sf.nodes)
end

materialize_pressure_gradient(::Nothing, args...) = nothing

materialize_pressure_gradient(::HydrostaticPressureGradient, grid, u_bcs, v_bcs) =
    HydrostaticPressureGradient(Field{Face, Center, Nothing}(grid, boundary_conditions = u_bcs),
                                Field{Center, Face, Nothing}(grid, boundary_conditions = v_bcs))

Adapt.adapt_structure(to, sf::ProgressiveSlowForcing) =
    ProgressiveSlowForcing(Adapt.adapt(to, sf.Gᵁ),
                           Adapt.adapt(to, sf.Gⱽ),
                           Adapt.adapt(to, sf.pressure_gradient),
                           Adapt.adapt(to, sf.weights),
                           Adapt.adapt(to, sf.frozen),
                           Adapt.adapt(to, sf.nodes))

# The stage values are rebuilt within every time step.
Oceananigans.prognostic_state(::FrozenSlowForcing) = nothing
Oceananigans.prognostic_state(::ProgressiveSlowForcing) = nothing

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

@inline reconstruction_degree(stage) = min(stage, 3)

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

@inline slow_forcing(i, j, Gᴴ, Gᶜ::NTuple, w, s) = @inbounds reconstruction_polynomial(i, j, Gᴴ[i, j, 1], Gᶜ, w, s)

"""
    PressureGradientStages(stages, current)

The stage values of the depth-integrated pressure gradient and its value at the current stage, which the substep
kernel reconstructs while the remainder of the slow forcing is held at the current stage.
"""
struct PressureGradientStages{S, C}
    stages :: S
    current :: C
end

Adapt.adapt_structure(to, p::PressureGradientStages) = PressureGradientStages(Adapt.adapt(to, p.stages), Adapt.adapt(to, p.current))

@inline function slow_forcing(i, j, Gᴴ, Pᶜ::PressureGradientStages, w, s)
    @inbounds Pᴹ = Pᶜ.current[i, j, 1]
    @inbounds Gᴹ = Gᴴ[i, j, 1]
    return Gᴹ - Pᴹ + reconstruction_polynomial(i, j, Pᴹ, Pᶜ.stages, w, s)
end

@inline function reconstruction_polynomial(i, j, Gᴹ, Gᶜ::NTuple{N}, w, s) where N
    F₀ = reconstruction_sum(w[1], Gᶜ, Gᴹ, i, j, Val(N))
    F₁ = reconstruction_sum(w[2], Gᶜ, Gᴹ, i, j, Val(N))
    F₂ = reconstruction_sum(w[3], Gᶜ, Gᴹ, i, j, Val(N))
    return F₀ + F₁ * s + F₂ * s^2
end

@inline reconstruction_sum(w, Gᶜ, Gᴹ, i, j, ::Val{N}) where N =
    @inbounds w[N+1] * Gᴹ + sum(ntuple(n -> w[n] * Gᶜ[n][i, j, 1], Val(N)))

#####
##### The depth-integrated hydrostatic pressure gradient
#####

@kernel function _compute_depth_integrated_pressure_gradient!(Pᵁ, Pⱽ, grid, p′)
    i, j = @index(Global, NTuple)

    locU = (Face(), Center(), Center())
    locV = (Center(), Face(), Center())

    Pˣ = zero(grid)
    Pʸ = zero(grid)

    for k in 1:grid.Nz
        @inbounds Pˣ -= Δzᶠᶜᶜ(i, j, k, grid) * ∂xᶠᶜᶜ(i, j, k, grid, p′) * !peripheral_node(i, j, k, grid, locU...)
        @inbounds Pʸ -= Δzᶜᶠᶜ(i, j, k, grid) * ∂yᶜᶠᶜ(i, j, k, grid, p′) * !peripheral_node(i, j, k, grid, locV...)
    end

    @inbounds Pᵁ[i, j, 1] = Pˣ
    @inbounds Pⱽ[i, j, 1] = Pʸ
end

compute_reconstructed_pressure_gradient!(slow_forcing, grid, p′) = nothing

compute_reconstructed_pressure_gradient!(sf::ProgressiveSlowForcing, grid, p′) =
    compute_depth_integrated_pressure_gradient!(sf.pressure_gradient, grid, p′)

compute_depth_integrated_pressure_gradient!(::Nothing, grid, p′) = nothing

function compute_depth_integrated_pressure_gradient!(pg::HydrostaticPressureGradient, grid, p′)
    launch!(architecture(grid), grid, :xy, _compute_depth_integrated_pressure_gradient!, pg.Pᵁ, pg.Pⱽ, grid, p′)
    fill_halo_regions!((pg.Pᵁ, pg.Pⱽ))
    return nothing
end

#####
##### Capturing the stage values, and selecting when the reconstruction is active
#####

@inline cache_stage_slow_forcing!(::FrozenSlowForcing, args...) = nothing

@inline reconstructed_forcing(::Nothing, GUⁿ, GVⁿ) = (GUⁿ, GVⁿ)
@inline reconstructed_forcing(pg::HydrostaticPressureGradient, GUⁿ, GVⁿ) = (pg.Pᵁ, pg.Pⱽ)

@inline function cache_stage_slow_forcing!(sf::ProgressiveSlowForcing, GUⁿ, GVⁿ, stage)
    if stage ≤ cached_stages(sf)
        Fᵁ, Fⱽ = reconstructed_forcing(sf.pressure_gradient, GUⁿ, GVⁿ)
        parent(sf.Gᵁ[stage]) .= parent(Fᵁ)
        parent(sf.Gⱽ[stage]) .= parent(Fⱽ)
    end
    return nothing
end

@inline zero_reconstruction_weights(::Val{N}, ::Type{FT}) where {N, FT} =
    ntuple(k -> ntuple(j -> zero(FT), Val(N + 1)), Val(3))

@inline reconstruction_timestep(::SSPRungeKuttaTimeStepper, stage, Δτ) = Δτ
@inline reconstruction_timestep(ts::SplitRungeKuttaTimeStepper, stage, Δτ) = @inbounds Δτ * ts.β[stage]

# `nothing` caches specialize the substep kernel on the frozen case, keeping the plain `Gᴴ[i, j, 1]` load.
@inline stage_reconstruction(::FrozenSlowForcing, timestepper, stage, Δτ::FT) where FT = (nothing, nothing, zero_reconstruction_weights(Val(1), FT))

# Anything that is not a multi-stage scheme has no stage values to reconstruct from.
@inline stage_reconstruction(sf::ProgressiveSlowForcing, timestepper, stage, Δτ::FT) where FT = (nothing, nothing, zero_reconstruction_weights(Val(length(sf.Gᵁ)), FT))

@inline function stage_reconstruction(sf::ProgressiveSlowForcing, ts::MultiStageTimeStepper, stage, Δτ::FT) where FT
    @inbounds sf.frozen[stage] && return (nothing, nothing, zero_reconstruction_weights(Val(length(sf.Gᵁ)), FT))
    @inbounds w̃ = sf.weights[stage]
    w = scale_reconstruction_weights(w̃, reconstruction_timestep(ts, stage, Δτ))
    Gᵁᶜ, Gⱽᶜ = stage_values(sf.pressure_gradient, sf.Gᵁ, sf.Gⱽ)
    return (Gᵁᶜ, Gⱽᶜ, w)
end

@inline stage_values(::Nothing, Gᵁ, Gⱽ) = (Gᵁ, Gⱽ)
@inline stage_values(pg::HydrostaticPressureGradient, Gᵁ, Gⱽ) =
    (PressureGradientStages(Gᵁ, pg.Pᵁ), PressureGradientStages(Gⱽ, pg.Pⱽ))
