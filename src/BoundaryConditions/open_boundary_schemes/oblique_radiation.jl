#####
##### ObliqueRadiation (Raymond & Kuo 1984) open boundary scheme
#####

"""
    ObliqueRadiation(; inflow_timescale = 0,
                       outflow_timescale = Inf,
                       phase_speed_weight = 0.3,
                       target_transport = nothing)

Raymond & Kuo (1984) two-dimensional radiation condition with nudging to the exterior value
(Marchesiello et al. 2001). With `φ₁`, `φ₂` the first and second interior values next to the
boundary, the normal and tangential phase-speed coefficients are estimated from the interior as

    rₙ = min(∂ₜφ ∂ₙφ, c),   rₜ = clamp(∂ₜφ ∂ₜ̂φ, -c, c),   c = max((∂ₙφ)² + (∂ₜ̂φ)², ε)

with `∂ₜφ = φ₁ⁿ - φ₁ⁿ⁺¹`, `∂ₙφ = φ₁ⁿ⁺¹ - φ₂ⁿ⁺¹` pointing out of the domain, and `∂ₜ̂φ` the upwinded
difference along the boundary. A phase speed pointing into the domain (`∂ₜφ ∂ₙφ < 0`) is set to
zero: waves are only radiated out. The coefficients `rₙ`, `rₜ` and `c` are averaged in time,

    r̄ = (1 - w) r̄ + w r,    w = phase_speed_weight,

once per time step (`phase_speed_weight = 1` turns the averaging off), and the boundary value is

    φᵇ = (c̄ φᵇ + r̄ₙ φ₁ⁿ⁺¹ - max(r̄ₜ, 0) δ₋φᵇ - min(r̄ₜ, 0) δ₊φᵇ) / (c̄ + r̄ₙ)

where `δ∓φᵇ` are the differences of the boundary values along the boundary. It is then nudged
toward the exterior value `φᵉˣᵗ`,

    φᵇ = (1 - Δt / (τ + Δt)) φᵇ + Δt / (τ + Δt) φᵉˣᵗ,

with `τ = inflow_timescale` when the phase speed points into the domain or vanishes, and
`τ = outflow_timescale` otherwise; `inflow_timescale = 0` imposes `φᵉˣᵗ` on inflow. The tangential
term is applied on the lateral boundaries only. `target_transport` pins the net transport through a
`NormalFlowBoundaryCondition`, as for [`NormalRadiation`](@ref).

References
==========
* Raymond, W. H., & Kuo, H. L. (1984). "A radiation boundary condition for multi-dimensional
  flows." Quarterly Journal of the Royal Meteorological Society, 110(464), 535-551.
* Marchesiello, P., McWilliams, J. C., & Shchepetkin, A. (2001). "Open boundary conditions
  for long-term integration of regional oceanic models." Ocean Modelling, 3(1-2), 1-20.

```jldoctest
using Oceananigans
using Oceananigans.BoundaryConditions: ObliqueRadiation

ObliqueRadiation()

# output
ObliqueRadiation{Float64}
├── inflow_timescale: 0.0
├── outflow_timescale: Inf
├── phase_speed_weight: 0.3
└── target_transport: Nothing
```
"""
struct ObliqueRadiation{FT, S, B, TF} <: AbstractRadiationScheme{FT}
    outflow_timescale  :: FT
    inflow_timescale   :: FT
    phase_speed_weight :: FT
    φᵇ  :: S
    φ₁  :: S
    φ₁ˡ :: S
    previous_boundary :: B # boundary values written during the previous iteration, double-buffered by iteration parity
    previous_interior :: B # first-interior values, likewise
    rₙ :: B                # time-averaged coefficients: [.., .., 1] at the start of the step, [.., .., 2] latest
    rₜ :: B
    c  :: B
    target_transport :: TF # prescribed net transport through the boundary, or nothing
end

function ObliqueRadiation(FT = defaults.FloatType;
                          inflow_timescale = 0,
                          outflow_timescale = Inf,
                          phase_speed_weight = 0.3,
                          target_transport = nothing)

    0 < phase_speed_weight ≤ 1 || throw(ArgumentError("phase_speed_weight must be in (0, 1], got $phase_speed_weight"))
    outflow_timescale = convert(FT, outflow_timescale)
    inflow_timescale = convert(FT, inflow_timescale)
    phase_speed_weight = convert(FT, phase_speed_weight)
    target_transport = convert_target_transport(FT, target_transport)
    return ObliqueRadiation(outflow_timescale, inflow_timescale, phase_speed_weight,
                            nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, target_transport)
end

Adapt.adapt_structure(to, r::ObliqueRadiation) =
    ObliqueRadiation(adapt(to, r.outflow_timescale),
                     adapt(to, r.inflow_timescale),
                     adapt(to, r.phase_speed_weight),
                     adapt(to, r.φᵇ),
                     adapt(to, r.φ₁),
                     adapt(to, r.φ₁ˡ),
                     adapt(to, r.previous_boundary),
                     adapt(to, r.previous_interior),
                     adapt(to, r.rₙ),
                     adapt(to, r.rₜ),
                     adapt(to, r.c),
                     adapt(to, r.target_transport))

function Base.show(io::IO, r::ObliqueRadiation)
    print(io, summary(r), '\n')
    print(io, "├── inflow_timescale: ",  prettysummary(r.inflow_timescale), '\n')
    print(io, "├── outflow_timescale: ", prettysummary(r.outflow_timescale), '\n')
    print(io, "├── phase_speed_weight: ", prettysummary(r.phase_speed_weight), '\n')
    print(io, "└── target_transport: ", prettysummary(r.target_transport))
end

# The velocity only selects inflow/outflow for `NormalRadiation`; `ObliqueRadiation` uses the phase speed.
@inline uses_boundary_velocity(::ObliqueRadiation) = false

has_target_transport(::ObliqueRadiation{<:Any, <:Any, <:Any, <:Nothing}) = false
has_target_transport(::ObliqueRadiation) = true

radiation_buffers(radiation::ObliqueRadiation, arch, FT, tangential_size) =
    (ntuple(_ -> zeros(arch, FT, tangential_size...), 3)...,
     ntuple(_ -> zeros(arch, FT, tangential_size..., 2), 5)...)

radiation_buffers(radiation::ObliqueRadiation) =
    (radiation.φᵇ, radiation.φ₁, radiation.φ₁ˡ, radiation.previous_boundary, radiation.previous_interior,
     radiation.rₙ, radiation.rₜ, radiation.c)

radiation_storage(radiation::ObliqueRadiation, (φᵇ, φ₁, φ₁ˡ, previous_boundary, previous_interior, rₙ, rₜ, c)) =
    ObliqueRadiation(radiation.outflow_timescale, radiation.inflow_timescale, radiation.phase_speed_weight,
                     φᵇ, φ₁, φ₁ˡ, previous_boundary, previous_interior, rₙ, rₜ, c, radiation.target_transport)

# Fills read the buffer written during the previous iteration and write the other one.
@inline written_buffer(clock) = clock.iteration % 2 + 1
@inline written_buffer(::Nothing) = 1

# Backward and forward differences along the boundary face, zero beyond its ends.
@inline function tangential_differences(φ, t, k, b)
    T = size(φ, 1)
    @inbounds begin
        φ₀ = φ[t, k, b]
        φ₋ = φ[max(t - 1, 1), k, b]
        φ₊ = φ[min(t + 1, T), k, b]
    end
    return φ₀ - φ₋, φ₊ - φ₀
end

# The phase-speed coefficients (rₙ, rₜ, c) estimated from the interior at this fill, and whether the phase
# speed points out of the domain.
@inline function oblique_phase_speeds(φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, δ₁₋, δ₁₊)
    ∂t_φ = φ₁ⁿ - φ₁ⁿ⁺¹
    ∂n_φ = φ₁ⁿ⁺¹ - φ₂ⁿ⁺¹
    s = ∂t_φ * (δ₁₋ + δ₁₊)
    FT = typeof(∂n_φ)
    ∂t̂_φ = ifelse(s > 0, δ₁₋, ifelse(s == 0, zero(FT), δ₁₊))
    radiating = ∂t_φ * ∂n_φ > 0
    ∂t_φ = ifelse(∂t_φ * ∂n_φ < 0, zero(FT), ∂t_φ)

    c  = max(∂n_φ^2 + ∂t̂_φ^2, eps(FT))
    rₙ = min(∂t_φ * ∂n_φ, c)
    rₜ = clamp(∂t_φ * ∂t̂_φ, -c, c)
    return rₙ, rₜ, c, radiating
end

# Radiate the boundary value with the time-averaged coefficients, then nudge it toward the exterior value.
@inline function oblique_radiation_update(φᵇⁿ, φ₁ⁿ⁺¹, δᵇ₋, δᵇ₊, r̄ₙ, r̄ₜ, c̄, radiating, φᵉˣᵗ, Δt, radiation)
    FT = typeof(φᵇⁿ)
    φʳ = (c̄ * φᵇⁿ + r̄ₙ * φ₁ⁿ⁺¹ - max(r̄ₜ, zero(FT)) * δᵇ₋ - min(r̄ₜ, zero(FT)) * δᵇ₊) / (c̄ + r̄ₙ)

    τ = ifelse(radiating, radiation.outflow_timescale, radiation.inflow_timescale)
    γ = Δt / (τ + Δt)
    φᵇⁿ⁺¹ = (1 - γ) * φʳ + γ * φᵉˣᵗ

    return ifelse(τ == 0, φᵉˣᵗ, φᵇⁿ⁺¹)
end

# The flow-direction arguments (`outflow`, `Cᵃ`) shared with `NormalRadiation` are not used: `ObliqueRadiation`
# distinguishes inflow from outflow by the direction of the phase speed.
@inline function radiation_update(radiation::ObliqueRadiation, t, k, clock, φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, outflow, Cᵃ)
    w = written_buffer(clock)
    r = 3 - w
    δᵇ₋, δᵇ₊ = tangential_differences(radiation.previous_boundary, t, k, r)
    δ₁₋, δ₁₊ = tangential_differences(radiation.previous_interior, t, k, r)
    rₙ, rₜ, c, radiating = oblique_phase_speeds(φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, δ₁₋, δ₁₊)

    # The averages advance once per time step: the first fill of a step promotes the latest average to the
    # start-of-step value, and every fill of the step averages from that start-of-step value.
    anchored = anchored_fill(clock)
    ω = radiation.phase_speed_weight
    @inbounds begin
        r̄ₙ⁰ = ifelse(anchored, radiation.rₙ[t, k, 2], radiation.rₙ[t, k, 1])
        r̄ₜ⁰ = ifelse(anchored, radiation.rₜ[t, k, 2], radiation.rₜ[t, k, 1])
        c̄⁰  = ifelse(anchored, radiation.c[t, k, 2],  radiation.c[t, k, 1])
    end
    r̄ₙ = (1 - ω) * r̄ₙ⁰ + ω * rₙ
    r̄ₜ = (1 - ω) * r̄ₜ⁰ + ω * rₜ
    c̄  = (1 - ω) * c̄⁰  + ω * c

    φᵇⁿ⁺¹ = oblique_radiation_update(φᵇⁿ, φ₁ⁿ⁺¹, δᵇ₋, δᵇ₊, r̄ₙ, r̄ₜ, c̄, radiating, φᵉˣᵗ, Δt, radiation)

    @inbounds begin
        radiation.rₙ[t, k, 1] = r̄ₙ⁰
        radiation.rₜ[t, k, 1] = r̄ₜ⁰
        radiation.c[t, k, 1]  = c̄⁰
        radiation.rₙ[t, k, 2] = r̄ₙ
        radiation.rₜ[t, k, 2] = r̄ₜ
        radiation.c[t, k, 2]  = c̄
        radiation.previous_boundary[t, k, w] = φᵇⁿ⁺¹
        radiation.previous_interior[t, k, w] = φ₁ⁿ⁺¹
    end

    return φᵇⁿ⁺¹
end
