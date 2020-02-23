module Operators

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Operators

using Oceananigans: AbstractGrid

export
    kinetic_energy,
    hdivᶜᶜᵃ, divᶜᶜᶜ, ∇²,
    div_uc, ∂ⱼpuⱼ, ∂ⱼDᶜⱼ, ∂ⱼtᶜDᶜⱼ, ∂ⱼDᵖⱼ,
    div_ρuũ, div_ρvũ, div_ρwũ,
    ∂ⱼτ₁ⱼ, ∂ⱼτ₂ⱼ, ∂ⱼτ₃ⱼ, Q_dissipation

include("compressible_operators.jl")

end
