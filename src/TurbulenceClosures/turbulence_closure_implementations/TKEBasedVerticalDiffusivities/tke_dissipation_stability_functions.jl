abstract type AbstractConstantSchmidtStabilityFunctions end

const ConstantSchmidtStabilityTDVD = TKEDissipationVerticalDiffusivity{<:Any, <:Any, <:AbstractConstantSchmidtStabilityFunctions}

@inline function tke_stability_functionᶜᶜᶠ(i, j, k, grid, closure::ConstantSchmidtStabilityTDVD, args...)
    Cσe = closure.stability_functions.Cσe
    𝕊u = momentum_stability_functionᶜᶜᶠ(i, j, k, grid, closure, args...)
    return 𝕊u / Cσe
end

@inline function dissipation_stability_functionᶜᶜᶠ(i, j, k, grid, closure::ConstantSchmidtStabilityTDVD, args...)
    Cσϵ = closure.stability_functions.Cσϵ
    𝕊u = momentum_stability_functionᶜᶜᶠ(i, j, k, grid, closure, args...)
    return 𝕊u / Cσϵ
end

Base.@kwdef struct ConstantStabilityFunctions{FT} <: AbstractConstantSchmidtStabilityFunctions
    Cσe :: FT = 1.0
    Cσϵ :: FT = 1.2
    Cu₀ :: FT = 0.53 # √3
    Cc₀ :: FT = 0.53 # √3
    𝕊u₀ :: FT = 0.53 # √3
end

Adapt.@adapt_structure ConstantStabilityFunctions

Base.summary(s::ConstantStabilityFunctions{FT}) where FT = "ConstantStabilityFunctions{$FT}"

summarize_stability_functions(s::ConstantStabilityFunctions{FT}, prefix="", sep="│   ") where FT =
    string(prefix, "ConstantStabilityFunctions{$FT}:", '\n',
           "    ├── 𝕊u₀: ", prettysummary(s.𝕊u₀), '\n',
           "    ├── Cσe: ", prettysummary(s.Cσe), '\n',
           "    ├── Cσϵ: ", prettysummary(s.Cσϵ), '\n',
           "    ├── Cu₀: ", prettysummary(s.Cu₀), '\n',
           "    └── Cc₀: ", prettysummary(s.Cc₀))

const ConstantStabilityTDVD = TKEDissipationVerticalDiffusivity{<:Any, <:Any, <:ConstantStabilityFunctions}

@inline momentum_stability_functionᶜᶜᶠ(i, j, k, grid, c::ConstantStabilityTDVD, args...) = c.stability_functions.Cu₀
@inline   tracer_stability_functionᶜᶜᶠ(i, j, k, grid, c::ConstantStabilityTDVD, args...) = c.stability_functions.Cc₀

struct VariableStabilityFunctions{FT} <: AbstractConstantSchmidtStabilityFunctions
    Cσe :: FT
    Cσϵ :: FT
    Cu₀ :: FT
    Cu₁ :: FT
    Cu₂ :: FT
    Cc₀ :: FT
    Cc₁ :: FT
    Cc₂ :: FT
    Cd₀ :: FT
    Cd₁ :: FT
    Cd₂ :: FT
    Cd₃ :: FT
    Cd₄ :: FT
    Cd₅ :: FT
    𝕊u₀ :: FT
end

Adapt.@adapt_structure VariableStabilityFunctions

VariableStabilityFunctions{FT}(; kw...) where FT = VariableStabilityFunctions(FT; kw...)

function VariableStabilityFunctions(FT=Oceananigans.defaults.FloatType;
                                    Cσe = 1.0,
                                    Cσϵ = 1.2,
                                    Cu₀ = 0.1067,
                                    Cu₁ = 0.0173,
                                    Cu₂ = -0.0001205,
                                    Cc₀ = 0.1120,
                                    Cc₁ = 0.003766,
                                    Cc₂ = 0.0008871,
                                    Cd₀ = 1.0,
                                    Cd₁ = 0.2398,
                                    Cd₂ = 0.02872,
                                    Cd₃ = 0.005154,
                                    Cd₄ = 0.006930,
                                    Cd₅ = -0.0003372,
                                    𝕊u₀ = nothing)

    if isnothing(𝕊u₀)
        # Compute 𝕊u₀ for the logarithmic boundary layer where production
        # balances dissipation. For more information see the discussion
        # surrounding equation (13) by Umlauf and Burchard (2003).
        a = Cd₅ - Cu₂
        b = Cd₂ - Cu₀
        c = Cd₀
        𝕊u₀ = (2a / (-b - sqrt(b^2 - 4a * c)))^(1/4)
    end

    return VariableStabilityFunctions(convert(FT, Cσe),
                                      convert(FT, Cσϵ),
                                      convert(FT, Cu₀),
                                      convert(FT, Cu₁),
                                      convert(FT, Cu₂),
                                      convert(FT, Cc₀),
                                      convert(FT, Cc₁),
                                      convert(FT, Cc₂),
                                      convert(FT, Cd₀),
                                      convert(FT, Cd₁),
                                      convert(FT, Cd₂),
                                      convert(FT, Cd₃),
                                      convert(FT, Cd₄),
                                      convert(FT, Cd₅),
                                      convert(FT, 𝕊u₀))
end

Base.summary(s::VariableStabilityFunctions{FT}) where FT = "VariableStabilityFunctions{$FT}"

summarize_stability_functions(s::VariableStabilityFunctions{FT}, prefix="", sep="") where FT =
    string("VariableStabilityFunctions{$FT}:", '\n',
           "    ├── Cσe: ", prettysummary(s.Cσe), '\n',
           "    ├── Cσϵ: ", prettysummary(s.Cσϵ), '\n',
           "    ├── Cu₀: ", prettysummary(s.Cu₀), '\n',
           "    ├── Cu₁: ", prettysummary(s.Cu₁), '\n',
           "    ├── Cu₂: ", prettysummary(s.Cu₂), '\n',
           "    ├── Cc₀: ", prettysummary(s.Cc₀), '\n',
           "    ├── Cc₁: ", prettysummary(s.Cc₁), '\n',
           "    ├── Cc₂: ", prettysummary(s.Cc₂), '\n',
           "    ├── Cd₀: ", prettysummary(s.Cd₀), '\n',
           "    ├── Cd₁: ", prettysummary(s.Cd₁), '\n',
           "    ├── Cd₂: ", prettysummary(s.Cd₂), '\n',
           "    ├── Cd₃: ", prettysummary(s.Cd₃), '\n',
           "    ├── Cd₄: ", prettysummary(s.Cd₄), '\n',
           "    └── Cd₅: ", prettysummary(s.Cd₅))

@inline function square_time_scaleᶜᶜᶜ(i, j, k, grid, closure, tracers, buoyancy)
    e★ = turbulent_kinetic_energyᶜᶜᶜ(i, j, k, grid, closure, tracers)
    ϵ★ = dissipationᶜᶜᶜ(i, j, k, grid, closure, tracers, buoyancy)
    return e★^2 / ϵ★^2
end

@inline function shear_numberᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy)
    τ² = ℑzᵃᵃᶠ(i, j, k, grid, square_time_scaleᶜᶜᶜ, closure, tracers, buoyancy)
    S² = shearᶜᶜᶠ(i, j, k, grid, velocities.u, velocities.v)
    return τ² * S²
end

@inline function stratification_numberᶜᶜᶠ(i, j, k, grid, closure, tracers, buoyancy)
    τ² = ℑzᵃᵃᶠ(i, j, k, grid, square_time_scaleᶜᶜᶜ, closure, tracers, buoyancy)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    return τ² * N²
end

@inline maximum_stratification_number(closure) = 1e10 # ?

"""
Based on an argument for free convection, assuming a balance between
buoyancy production and dissipation.

See Umlauf and Burchard (2005) equation A.22.

Note that _another_ condition could arise depending on the time discretization,
as discussed in the text surrounding equation 45-46 in Umlauf and Buchard (2005).
"""
@inline function minimum_stratification_number(closure)
    m₀ = closure.stability_functions.Cc₀
    m₁ = closure.stability_functions.Cc₁
    m₂ = closure.stability_functions.Cc₂

    d₀ = closure.stability_functions.Cd₀
    d₁ = closure.stability_functions.Cd₁
    d₂ = closure.stability_functions.Cd₂
    d₃ = closure.stability_functions.Cd₃
    d₄ = closure.stability_functions.Cd₄
    d₅ = closure.stability_functions.Cd₅

    a = d₄ + m₁
    b = d₁ + m₀
    c = d₀

    αᴺmin = (- b + sqrt(b^2 - 4a*c)) / 2a

    # Reduce by the "safety factor"
    ϵ = closure.minimum_stratification_number_safety_factor
    αᴺmin *= ϵ

    return αᴺmin
end

@inline minimum_shear_number(closure::FlavorOfTD) = zero(eltype(closure))

"""
Based on the condition that shear aniostropy must increase.

See Umlauf and Burchard (2005) equation 44.
"""
@inline function maximum_shear_number(closure, αᴺ)
    n₀ = closure.stability_functions.Cu₀
    n₁ = closure.stability_functions.Cu₁
    n₂ = closure.stability_functions.Cu₂

    d₀ = closure.stability_functions.Cd₀
    d₁ = closure.stability_functions.Cd₁
    d₂ = closure.stability_functions.Cd₂
    d₃ = closure.stability_functions.Cd₃
    d₄ = closure.stability_functions.Cd₄
    d₅ = closure.stability_functions.Cd₅

    ϵ₀ = d₀ * n₀
    ϵ₁ = d₀ * n₁ + d₁ * n₀
    ϵ₂ = d₁ * n₁ + d₄ * n₀
    ϵ₃ = d₄ * n₁
    ϵ₄ = d₂ * n₀
    ϵ₅ = d₂ * n₁ + d₃ * n₀
    ϵ₆ = d₃ * n₁

    num = ϵ₀ + ϵ₁ * αᴺ + ϵ₂ * αᴺ^2 + ϵ₃ * αᴺ^3
    den = ϵ₄ + ϵ₅ * αᴺ + ϵ₆ * αᴺ^2

    return num / den
end

const VariableStabilityTDVD = TKEDissipationVerticalDiffusivity{<:Any, <:Any, <:VariableStabilityFunctions}

@inline function momentum_stability_functionᶜᶜᶠ(i, j, k, grid, closure::VariableStabilityTDVD, velocities, tracers, buoyancy)
    αᴺ = stratification_numberᶜᶜᶠ(i, j, k, grid, closure, tracers, buoyancy)
    αᴹ = shear_numberᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy)

    αᴺmin = minimum_stratification_number(closure)
    αᴺmax = maximum_stratification_number(closure)
    αᴺ = clamp(αᴺ, αᴺmin, αᴺmax)

    αᴹmin = minimum_shear_number(closure)
    αᴹmax = maximum_shear_number(closure, αᴺ)
    αᴹ = clamp(αᴹ, αᴹmin, αᴹmax)

    𝕊u = momentum_stability_function(closure, αᴺ, αᴹ)
    return 𝕊u
end

@inline function momentum_stability_function(closure::VariableStabilityTDVD, αᴺ::Number, αᴹ::Number)
    Cu₀ = closure.stability_functions.Cu₀
    Cu₁ = closure.stability_functions.Cu₁
    Cu₂ = closure.stability_functions.Cu₂

    Cd₀ = closure.stability_functions.Cd₀
    Cd₁ = closure.stability_functions.Cd₁
    Cd₂ = closure.stability_functions.Cd₂
    Cd₃ = closure.stability_functions.Cd₃
    Cd₄ = closure.stability_functions.Cd₄
    Cd₅ = closure.stability_functions.Cd₅

    num = Cu₀ +
          Cu₁ * αᴺ +
          Cu₂ * αᴹ

    den = Cd₀ + Cd₁ * αᴺ +
          Cd₂ * αᴹ +
          Cd₃ * αᴺ * αᴹ +
          Cd₄ * αᴺ^2 +
          Cd₅ * αᴹ^2

    return num / den
end

@inline function tracer_stability_functionᶜᶜᶠ(i, j, k, grid, closure::VariableStabilityTDVD, velocities, tracers, buoyancy)
    αᴺ = stratification_numberᶜᶜᶠ(i, j, k, grid, closure, tracers, buoyancy)
    αᴹ = shear_numberᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy)

    αᴺmin = minimum_stratification_number(closure)
    αᴺmax = maximum_stratification_number(closure)
    αᴺ = clamp(αᴺ, αᴺmin, αᴺmax)

    αᴹmin = minimum_shear_number(closure)
    αᴹmax = maximum_shear_number(closure, αᴺ)
    αᴹ = clamp(αᴹ, αᴹmin, αᴹmax)

    𝕊c = tracer_stability_function(closure, αᴺ, αᴹ)
    return 𝕊c
end

@inline function tracer_stability_function(closure::VariableStabilityTDVD, αᴺ::Number, αᴹ::Number)
    Cc₀ = closure.stability_functions.Cc₀
    Cc₁ = closure.stability_functions.Cc₁
    Cc₂ = closure.stability_functions.Cc₂

    Cd₀ = closure.stability_functions.Cd₀
    Cd₁ = closure.stability_functions.Cd₁
    Cd₂ = closure.stability_functions.Cd₂
    Cd₃ = closure.stability_functions.Cd₃
    Cd₄ = closure.stability_functions.Cd₄
    Cd₅ = closure.stability_functions.Cd₅

    num = Cc₀ +
          Cc₁ * αᴺ +
          Cc₂ * αᴹ

    den = Cd₀ +
          Cd₁ * αᴺ +
          Cd₂ * αᴹ +
          Cd₃ * αᴺ * αᴹ +
          Cd₄ * αᴺ^2 +
          Cd₅ * αᴹ^2

    return num / den
end
