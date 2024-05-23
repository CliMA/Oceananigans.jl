Base.@kwdef struct ConstantStabilityFunctions{FT}
    Cμu :: FT = 0.1
    Cμc :: FT = 0.1
    Cμe :: FT = 0.1
    Cμϵ :: FT = 0.083
end

@inline momentum_stability_function(i, j, k, grid, stab::ConstantStabilityFunctions, args...)    = stab.Cμu
@inline tracer_stability_function(i, j, k, grid, stab::ConstantStabilityFunctions, args...)      = stab.Cμc
@inline tke_stability_function(i, j, k, grid, stab::ConstantStabilityFunctions, args...)         = stab.Cμe
@inline dissipation_stability_function(i, j, k, grid, stab::ConstantStabilityFunctions, args...) = stab.Cμϵ
