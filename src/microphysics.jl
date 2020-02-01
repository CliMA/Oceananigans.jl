import JULES.Operators: ρᵐ

abstract type AbstractMicrophysics end
struct VaporPlaceholder <: AbstractMicrophysics end
struct VaporLiquidIcePlaceholder <: AbstractMicrophysics end

validate_microphysics(::Nothing, tracers) = true

missing_tracers_error(names, mp) =
    "Must specify $names as tracers to use $(typeof(mp)) microphysics."

required_tracers(::VaporPlaceholder) = (:Qv,)
required_tracers(::VaporLiquidIcePlaceholder) = (:Qv, :Ql, :Qi)

function validate_microphysics(microphysics, tracers)
    C̃ = required_tracers(microphysics)
    for c in C̃
        c ∉ tracers && throw(ArgumentError(missing_tracers_error(C̃, microphysics)))
    end
    return true
end

#####
##### Moist density calculation
#####

@inline ρᵐ(i, j, k, grid, ::VaporPlaceholder, ρᵈ, C̃) =
    @inbounds ρᵈ[i, j, k] + C̃.Qv[i, j, k]

@inline ρᵐ(i, j, k, grid, ::VaporLiquidIcePlaceholder, ρᵈ, C̃) =
    @inbounds ρᵈ[i, j, k] + C̃.Qv[i, j, k] + C̃.Ql[i, j, k] + C̃.Qi[i, j, k]
