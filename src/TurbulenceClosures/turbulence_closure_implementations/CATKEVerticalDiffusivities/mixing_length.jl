using Oceananigans.Operators

"""
    struct MixingLength{FT}

Contains mixing length parameters for CATKE vertical diffusivity.

The mixing length is the maximum of three mixing lengths:

```math
ℓᵩ = max(ℓᵟᵩ, ℓᴬᵩ, ℓ⋆ᵩ)
```            

where ``ℓᵟ`` is a grid-spacing-proportional length,
``ℓʰᵩ`` is a convective mixing length, and ``ℓ⋆ᵩ`` is
a mixing length in stably-stratified mixing.

Grid-spacing-proportional mixing length
=======================================

```math
ℓᵟᵩ = Cᵟϕ * Δz
```

Convective mixing length
========================

```math
ℓᴬᵩ = convecting ? Cᴬϕ * e^3/2 / Qᵇ : 0
```

Stably-stratified mixing length
===============================

```math
ℓ⋆ᵩ = σᵩ * min(d, Cᴸᵇ * √e / N)
```

where ``σᵩ(Ri)`` is a stability function that depends on
the local Richardson number, ...

The Richardson number is

    ``Ri = ∂z B / ( (∂z U)² + (∂z V)² )`` ,

where ``B`` is buoyancy and ``∂z`` denotes a vertical derviative.
The Richardson-number dependent diffusivities are multiplied by the stability
function

    1. ``σ(Ri) = σ⁻ * (1 + rσ * step(Ri, Riᶜ, Riʷ))``
    3. ``σ(Ri) = σ⁻ + (σ⁺ - σ⁻) * step(Ri, Riᶜ, Riʷ)``

σ⁻ = σ₀
rσ = (σ⁺ - σ⁻) / σ₀

where ``σ₀``, ``Δσ``, ``Riᶜ``, and ``Riʷ`` are free parameters,
and ``step`` is a smooth step function defined by

    ``step(x, c, w) = (1 + \tanh((x - c) / w)) / 2``.

The 8 free parameters in `RiDependentDiffusivityScaling` have been _experimentally_ calibrated
against large eddy simulations of ocean surface boundary layer turbulence in idealized
scenarios involving monotonic boundary layer deepening into variable stratification
due to constant surface momentum fluxes and/or destabilizing surface buoyancy flux.
See https://github.com/CliMA/LESbrary.jl for more information about the large eddy simulations.
The calibration was performed using a combination of Markov Chain Monte Carlo (MCMC)-based simulated
annealing and noisy Ensemble Kalman Inversion methods.
"""
Base.@kwdef struct MixingLength{FT}
    Cᴸᵇ   :: FT = 1.16
    Cᵟu   :: FT = 0.5
    Cᵟc   :: FT = 0.5
    Cᵟe   :: FT = 0.5
    Cᴬu   :: FT = 0.0
    Cᴬc   :: FT = 0.0
    Cᴬe   :: FT = 0.0
    Cᴷu⁻  :: FT = 0.15
    Cᴷuʳ  :: FT = 3.87
    Cᴷc⁻  :: FT = 0.40
    Cᴷcʳ  :: FT = 0.77
    Cᴷe⁻  :: FT = 0.13
    Cᴷeʳ  :: FT = 1.11
    CᴷRiʷ :: FT = 0.72
    CᴷRiᶜ :: FT = 0.76
end

#####
##### Mixing length
#####

@inline surface(i, j, k, grid)                = znode(Center(), Center(), Face(), i, j, grid.Nz+1, grid)
@inline bottom(i, j, k, grid)                 = znode(Center(), Center(), Face(), i, j, 1, grid)
@inline depthᶜᶜᶜ(i, j, k, grid)               = surface(i, j, k, grid) - znode(Center(), Center(), Center(), i, j, k, grid)
@inline height_above_bottomᶜᶜᶜ(i, j, k, grid) = znode(Center(), Center(), Center(), i, j, k, grid) - bottom(i, j, k, grid)

@inline wall_vertical_distanceᶜᶜᶜ(i, j, k, grid) = min(depthᶜᶜᶜ(i, j, k, grid), height_above_bottomᶜᶜᶜ(i, j, k, grid))

@inline function sqrt_∂z_b(i, j, k, grid, buoyancy, tracers)
    FT = eltype(grid)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    N²⁺ = max(zero(FT), N²)
    return sqrt(N²⁺)  
end

@inline function buoyancy_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, e, tracers, buoyancy)
    FT = eltype(grid)
    Cᴸᵇ = closure.mixing_length.Cᴸᵇ
    N⁺ = ℑzᵃᵃᶜ(i, j, k, grid, sqrt_∂z_b, buoyancy, tracers)

    @inbounds e⁺ = max(zero(FT), e[i, j, k])

    return @inbounds ifelse(N⁺ == 0, FT(Inf), Cᴸᵇ * sqrt(e⁺) / N⁺)
end

@inline function unscaled_stable_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, e, tracers, buoyancy)
    d = wall_vertical_distanceᶜᶜᶜ(i, j, k, grid)
    ℓᵇ = buoyancy_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, e, tracers, buoyancy)
    return min(d, ℓᵇ)
end

@inline function unscaled_convective_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, tracer_bcs)
    Qᵇ = top_buoyancy_flux(i, j, grid, buoyancy, tracer_bcs, clock, merge(velocities, tracers))
    N² = ℑzᵃᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    convecting = (N² < 0) & (Qᵇ > 0)
    @inbounds eijk = max(zero(eltype(grid)), tracers.e[i, j, k])
    return ifelse(convecting, sqrt(eijk^3) / Qᵇ, zero(eltype(grid))) 
end

@inline ϕ²(i, j, k, grid, ϕ, args...) = ϕ(i, j, k, grid, args...)^2

@inline function Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy)
    FT = eltype(grid)
    ∂z_u² = ℑxzᶜᵃᶜ(i, j, k, grid, ϕ², ∂zᶠᶜᶠ, velocities.u)
    ∂z_v² = ℑyzᵃᶜᶜ(i, j, k, grid, ϕ², ∂zᶜᶠᶠ, velocities.v)
    N² = ℑzᵃᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    return ifelse(N² == 0, zero(FT), N² / (∂z_u² + ∂z_v²))
end

@inline step(x, c, w) = (1 + tanh((x - c) / w)) / 2

@inline scale(Ri, σ⁻, rσ, c, w) = σ⁻ * (1 + rσ * step(Ri, c, w))

@inline function momentum_stable_mixing_scale(i, j, k, grid, closure, velocities, tracers, buoyancy)
    Ri = Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy)
    return scale(Ri,
                 closure.mixing_length.Cᴷu⁻,
                 closure.mixing_length.Cᴷuʳ,
                 closure.mixing_length.CᴷRiᶜ,
                 closure.mixing_length.CᴷRiʷ)
end

@inline function tracer_stable_mixing_scale(i, j, k, grid, closure, velocities, tracers, buoyancy)
    Ri = Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy)
    return scale(Ri,
                 closure.mixing_length.Cᴷc⁻,
                 closure.mixing_length.Cᴷcʳ,
                 closure.mixing_length.CᴷRiᶜ,
                 closure.mixing_length.CᴷRiʷ)
end

@inline function TKE_stable_mixing_scale(i, j, k, grid, closure, velocities, tracers, buoyancy)
    Ri = Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy)
    return scale(Ri,
                 closure.mixing_length.Cᴷe⁻,
                 closure.mixing_length.Cᴷeʳ,
                 closure.mixing_length.CᴷRiᶜ,
                 closure.mixing_length.CᴷRiʷ)
end

@inline function momentum_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, tracer_bcs)
    Cᴬu = closure.mixing_length.Cᴬu
    ℓᴬ = unscaled_convective_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, tracer_bcs)

    Cᵟu = closure.mixing_length.Cᵟu
    ℓᵟ = Δzᶜᶜᶜ(i, j, k, grid)

    σu = momentum_stable_mixing_scale(i, j, k, grid, closure, velocities, tracers, buoyancy)
    ℓ★ = unscaled_stable_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, tracers.e, tracers, buoyancy)

    return max(Cᴬu * ℓᴬ, Cᵟu * ℓᵟ, σu * ℓ★)
end

@inline function tracer_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, tracer_bcs)
    Cᴬc = closure.mixing_length.Cᴬc
    ℓᴬ = unscaled_convective_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, tracer_bcs)

    Cᵟc = closure.mixing_length.Cᵟc
    ℓᵟ = Δzᶜᶜᶜ(i, j, k, grid)

    σc = tracer_stable_mixing_scale(i, j, k, grid, closure, velocities, tracers, buoyancy)
    ℓ★ = unscaled_stable_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, tracers.e, tracers, buoyancy)

    return max(Cᴬc * ℓᴬ, Cᵟc * ℓᵟ, σc * ℓ★)
end

@inline function TKE_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, tracer_bcs)
    Cᴬe = closure.mixing_length.Cᴬe
    ℓᴬ = unscaled_convective_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, tracer_bcs)

    Cᵟe = closure.mixing_length.Cᵟe
    ℓᵟ = Δzᶜᶜᶜ(i, j, k, grid)

    σe = TKE_stable_mixing_scale(i, j, k, grid, closure, velocities, tracers, buoyancy)
    ℓ★ = unscaled_stable_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, tracers.e, tracers, buoyancy)

    return max(Cᴬe * ℓᴬ, Cᵟe * ℓᵟ, σe * ℓ★)
end

Base.show(io::IO, ML::MixingLength) =
    print(io, "MixingLength: \n" *
              "          Cᴸᵇ = $(ML.Cᴸᵇ), \n" *
              "          Cᵟu = $(ML.Cᵟu), \n" *
              "          Cᵟc = $(ML.Cᵟc), \n" *
              "          Cᵟe = $(ML.Cᵟe), \n" *
              "          Cᴬu = $(ML.Cᴬu), \n" *
              "          Cᴬc = $(ML.Cᴬc), \n" *
              "          Cᴬe = $(ML.Cᴬe), \n" *
              "         Cᴷu⁻ = $(ML.Cᴷu⁻), \n" *
              "         Cᴷc⁻ = $(ML.Cᴷc⁻), \n" *
              "         Cᴷe⁻ = $(ML.Cᴷe⁻), \n" *
              "         Cᴷuʳ = $(ML.Cᴷuʳ), \n" *
              "         Cᴷcʳ = $(ML.Cᴷcʳ), \n" *
              "         Cᴷeʳ = $(ML.Cᴷeʳ), \n" *
              "        CᴷRiʷ = $(ML.CᴷRiʷ), \n" *
              "        CᴷRiᶜ = $(ML.CᴷRiᶜ)")
