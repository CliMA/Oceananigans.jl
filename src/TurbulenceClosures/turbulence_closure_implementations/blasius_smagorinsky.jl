#####
##### Blasius Smagorinsky: the version of the Smagorinsky turbulence closure
##### used in the UK Met Office code "Blasius" (see Polton and Belcher 2007).
#####

"""
    BlasiusSmagorinsky{ML, FT}

Parameters for the version of the Smagorinsky closure used in the UK Met Office code
Blasius, according to Polton and Belcher (2007).
"""
struct BlasiusSmagorinsky{ML, FT, P, K} <: AbstractSmagorinsky{FT}
               Pr :: P
                ν :: FT
                κ :: K
    mixing_length :: ML

    function BlasiusSmagorinsky{FT}(Pr, ν, κ, mixing_length) where FT
        Pr = convert_diffusivity(FT, Pr)
         κ = convert_diffusivity(FT, κ)
        return new{typeof(mixing_length), FT, typeof(Pr), typeof(κ)}(Pr, ν, κ, mixing_length)
    end
end

"""
    BlasiusSmagorinsky(FT=Float64; Pr=1.0, ν=1.05e-6, κ=1.46e-7)

Returns a `BlasiusSmagorinsky` closure object of type `FT`.

Keyword arguments
=================
    - `Pr` : Turbulent Prandtl numbers for each tracer. Either a constant applied to every
             tracer, or a `NamedTuple` with fields for each tracer individually.
    - `ν`  : Constant background viscosity for momentum
    - `κ`  : Constant background diffusivity for tracer. Can either be a single number
             applied to all tracers, or `NamedTuple` of diffusivities corresponding to each
             tracer.

References
==========
Polton, J. A., and Belcher, S. E. (2007), "Langmuir turbulence and deeply penetrating jets
    in an unstratified mixed layer." Journal of Geophysical Research: Oceans.
"""
BlasiusSmagorinsky(FT=Float64; Pr=1.0, ν=ν₀, κ=κ₀, mixing_length=nothing) =
    BlasiusSmagorinsky{FT}(Pr, ν, κ, mixing_length)

function with_tracers(tracers, closure::BlasiusSmagorinsky{ML, FT}) where {ML, FT}
    Pr = tracer_diffusivities(tracers, closure.Pr)
     κ = tracer_diffusivities(tracers, closure.κ)
    return BlasiusSmagorinsky{FT}(Pr, closure.ν, κ, closure.mixing_length)
end

const BS = BlasiusSmagorinsky

@inline Lm²(i, j, k, grid, closure::BS{<:AbstractFloat}, args...) =
    closure.mixing_length^2

@inline Lm²(i, j, k, grid, closure::BS{<:Nothing}, args...) =
    geo_mean_Δᶠ(i, j, k, grid)^2

#####
##### Mixing length via Monin-Obukhov theory
#####

"""
    MoninObukhovMixingLength(FT=Float64; kwargs...)

Returns a `MoninObukhovMixingLength closure object of type `FT` with

    * Cκ : Von Karman constant
    * z₀ : roughness length
    * L₀ : 'base' mixing length
    * Qu : surface velocity flux (Qu = τ/ρ₀)
    * Qb : surface buoyancy flux

The surface velocity flux and buoyancy flux are restricted to constants for now.
"""
struct MoninObukhovMixingLength{L, FT, U, B}
    Cκ :: FT
    z₀ :: FT
    L₀ :: L
    Qu :: U
    Qb :: U
end

MoninObukhovMixingLength(FT=Float64; Qu, Qb, Cκ=0.4, z₀=0.0, L₀=nothing) =
    MoninObukhovMixingLength(FT(Cκ), FT(z₀), L₀, Qu, Qb)

const MO = MoninObukhovMixingLength

@inline ϕm(z, Lm) = ϕm(z, Lm.Qu, Lm.Qb, Lm.Cκ)

"""
    ϕm(z, Qu, Qb, Cκ)

Return the stability function for monentum at height
`z` in terms of the velocity flux `Qu`, buoyancy flux
`Qb`, and von Karman constant `Cκ`.
"""
@inline function ϕm(z::FT, Qu, Qb, Cκ) where FT
    if Qu == 0
        return zero(FT)
    elseif Qb > 0 # unstable
        return ( 1 - 15Cκ * Qb * z / Qu^FT(1.5) )^FT(-0.25)
    else # neutral or stable
        return 1 + 5Cκ * Qb * z / Qu^FT(1.5)
    end
end

@inline function Lm²(i, j, k, grid, clo::BlasiusSmagorinsky{<:MO, FT}, Σ², N²) where FT
    Lm = clo.mixing_length
    z = @inbounds grid.zC[i, j, k]
    Lm⁻² = (
             ( ϕm(z + Lm.z₀, Lm) * buoyancy_factor(Σ², N²)^FT(0.25) / (Lm.Cκ * (z + Lm.z₀)) )^2
            + 1 / L₀(i, j, k, grid, Lm)^2
           )
    return 1 / Lm⁻²
end

@inline L₀(i, j, k, grid, mixing_length::MO{<:Number})  = mixing_length.L₀
@inline L₀(i, j, k, grid, mixing_length::MO{<:Nothing}) = geo_mean_Δᶠ(i, j, k, grid)
@inline buoyancy_factor(Σ², N²::FT) where FT = ifelse(Σ²==0, zero(FT), max(zero(FT), one(FT) - N² / Σ²))

"""
    νₑ_blasius(Lm², Σ², N²)

Return the eddy viscosity for the BLASIUS version of constant Smagorinsky
(Polton and Belcher, 2007) given the squared mixing length scale `Lm²`,
strain tensor dot product `Σ²`, and buoyancy gradient / squared buoyancy
frequency `N²`.
"""
@inline νₑ_blasius(Lm², Σ², N²::FT) where FT = Lm² * sqrt(Σ²/2 * buoyancy_factor(Σ², N²))

@inline function νᶜᶜᶜ(i, j, k, grid, clo::BlasiusSmagorinsky{ML, FT}, buoyancy, U, C) where {ML, FT}
    Σ² = ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, U.u, U.v, U.w)
    N² = max(zero(FT), ℑzᵃᵃᶜ(i, j, k, grid, buoyancy_frequency_squared, buoyancy, C))
    Lm_sq = Lm²(i, j, k, grid, clo, Σ², N²)

    return νₑ_blasius(Lm_sq, Σ², N²) + clo.ν
end
