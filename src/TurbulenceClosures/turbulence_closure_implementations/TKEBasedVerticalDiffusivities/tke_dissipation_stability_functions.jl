abstract type AbstractConstantSchmidtStabilityFunctions end

const ConstantSchmidtStabilityTDVD = TKEDissipationVerticalDiffusivity{<:Any, <:Any, <:AbstractConstantSchmidtStabilityFunctions}

@inline function tke_stability_function(i, j, k, grid, closure::ConstantSchmidtStabilityTDVD, args...)
    Cσe = closure.stability_functions.Cσe
    𝕊u = momentum_stability_function(i, j, k, grid, closure, args...)
    return 𝕊u / Cσe
end
        
@inline function dissipation_stability_function(i, j, k, grid, closure::ConstantSchmidtStabilityTDVD, args...)
    Cσϵ = closure.stability_functions.Cσϵ
    𝕊u = momentum_stability_function(i, j, k, grid, closure, args...)
    return 𝕊u / Cσϵ
end

Base.@kwdef struct ConstantStabilityFunctions{FT} <: AbstractConstantSchmidtStabilityFunctions
    Cσe :: FT = 1.0
    Cσϵ :: FT = 1.2
    Cu :: FT = 0.54 # √3
    Cc :: FT = 0.54 # √3
end

Base.summary(s::ConstantStabilityFunctions{FT}) where FT = "ConstantStabilityFunctions{$FT}"

summarize_stability_functions(s::ConstantStabilityFunctions{FT}, prefix="", sep="│   ") where FT =
    string(prefix, "ConstantStabilityFunctions{$FT}:", '\n',
           "    ├── Cσe: ", prettysummary(s.Cσe), '\n',
           "    ├── Cσϵ: ", prettysummary(s.Cσϵ), '\n',
           "    ├── Cu: ", prettysummary(s.Cu), '\n',
           "    └── Cc: ", prettysummary(s.Cc))

const ConstantStabilityTDVD = TKEDissipationVerticalDiffusivity{<:Any, <:Any, <:ConstantStabilityFunctions}

@inline momentum_stability_function(i, j, k, grid, c::ConstantStabilityTDVD, args...) = c.stability_functions.Cu
@inline   tracer_stability_function(i, j, k, grid, c::ConstantStabilityTDVD, args...) = c.stability_functions.Cc

Base.@kwdef struct VariableStabilityFunctions{FT} <: AbstractConstantSchmidtStabilityFunctions
    Cσe :: FT = 1.0
    Cσϵ :: FT = 1.2
    Cu₀ :: FT = 0.1067
    Cu₁ :: FT = 0.0173
    Cu₂ :: FT = -0.0001205
    Cc₀ :: FT = 0.1120
    Cc₁ :: FT = 0.003766
    Cc₂ :: FT = 0.0008871
    Cd₀ :: FT = 1.0
    Cd₁ :: FT = 0.2398
    Cd₂ :: FT = 0.02872
    Cd₃ :: FT = 0.005154
    Cd₄ :: FT = 0.006930
    Cd₅ :: FT = -0.0003372
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
    e = turbulent_kinetic_energyᶜᶜᶜ(i, j, k, grid, closure, tracers)
    ϵ = dissipationᶜᶜᶜ(i, j, k, grid, closure, tracers, buoyancy)
    return e^2 / ϵ^2
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

    ϵ = 0.73 # safety factor?
    ϵ = convert(typeof(c), ϵ)

    return ϵ * (- b + sqrt(b^2 - 4a*c)) / 2a
end

@inline minimum_shear_number(closure) = 0.0

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

@inline function momentum_stability_function(i, j, k, grid, closure::VariableStabilityTDVD, velocities, tracers, buoyancy)
    αᴺ = stratification_numberᶜᶜᶠ(i, j, k, grid, closure, tracers, buoyancy)
    αᴹ = shear_numberᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy)

    αᴺmin = minimum_stratification_number(closure)
    αᴺmax = maximum_stratification_number(closure)
    αᴺ = clamp(αᴺ, αᴺmin, αᴺmax)

    αᴹmin = minimum_shear_number(closure)
    αᴹmax = maximum_shear_number(closure, αᴺ)
    αᴹ = clamp(αᴹ, αᴹmin, αᴹmax)

    return momentum_stability_function(closure, αᴺ, αᴹ)
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

    num = Cu₀ + Cu₁ * αᴺ + Cu₂ * αᴹ
    den = Cd₀ + Cd₁ * αᴺ + Cd₂ * αᴹ + Cd₃ * αᴺ * αᴹ + Cd₄ * αᴺ^2 + Cd₅ * αᴹ^2

    return num / den
end

@inline function tracer_stability_function(i, j, k, grid, closure::VariableStabilityTDVD, velocities, tracers, buoyancy)
    αᴺ = stratification_numberᶜᶜᶠ(i, j, k, grid, closure, tracers, buoyancy)
    αᴹ = shear_numberᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy)

    αᴺmin = minimum_stratification_number(closure)
    αᴺmax = maximum_stratification_number(closure)
    αᴺ = clamp(αᴺ, αᴺmin, αᴺmax)

    αᴹmin = minimum_shear_number(closure)
    αᴹmax = maximum_shear_number(closure, αᴺ)
    αᴹ = clamp(αᴹ, αᴹmin, αᴹmax)

    return tracer_stability_function(closure::VariableStabilityTDVD, αᴺ::Number, αᴹ::Number)
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

    num = Cc₀ + Cc₁ * αᴺ + Cc₂ * αᴹ
    den = Cd₀ + Cd₁ * αᴺ + Cd₂ * αᴹ + Cd₃ * αᴺ * αᴹ + Cd₄ * αᴺ^2 + Cd₅ * αᴹ^2

    return num / den
end

