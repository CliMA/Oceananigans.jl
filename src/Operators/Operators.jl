module Operators

export
    Δx, Δy, ΔzF, ΔzC,
    Axᵃᵃᶜ, Axᵃᵃᶠ, Ayᵃᵃᶜ, Ayᵃᵃᶠ, Azᵃᵃᵃ,
    Vᵃᵃᶜ, Vᵃᵃᶠ,
    δxᶜᵃᵃ, δxᶠᵃᵃ, δyᵃᶜᵃ, δyᵃᶠᵃ, δzᵃᵃᶜ, δzᵃᵃᶠ,
    Ax_ψᵃᵃᶠ, Ax_ψᵃᵃᶜ, Ay_ψᵃᵃᶠ, Ay_ψᵃᵃᶜ, Az_ψᵃᵃᵃ,
    δᴶxᶜᵃᶜ, δᴶxᶜᵃᶠ, δᴶxᶠᵃᶜ, δᴶxᶠᵃᶠ, δᴶyᵃᶜᶜ, δᴶyᵃᶜᶠ, δᴶyᵃᶠᶜ, δᴶyᵃᶠᶠ, δᴶzᵃᵃᶜ, δᴶzᵃᵃᶠ,
    ∂xᶜᵃᵃ, ∂xᶠᵃᵃ, ∂yᵃᶜᵃ, ∂yᵃᶠᵃ, ∂zᵃᵃᶜ, ∂zᵃᵃᶠ,
    ∂²xᶜᵃᵃ, ∂²xᶠᵃᵃ, ∂²yᵃᶜᵃ, ∂²yᵃᶠᵃ, ∂²zᵃᵃᶜ, ∂²zᵃᵃᶠ,
    ∂⁴xᶜᵃᵃ, ∂⁴xᶠᵃᵃ, ∂⁴yᵃᶜᵃ, ∂⁴yᵃᶠᵃ, ∂⁴zᵃᵃᶜ, ∂⁴zᵃᵃᶠ,
    ℑxᶜᵃᵃ, ℑxᶠᵃᵃ, ℑyᵃᶜᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶜ, ℑzᵃᵃᶠ,
    ℑᴶxᶜᵃᵃ, ℑᴶxᶠᵃᵃ, ℑᴶyᵃᶜᵃ, ℑᴶyᵃᶠᵃ, ℑᴶzᵃᵃᶜ, ℑᴶzᵃᵃᶠ,
    ℑxyᶜᶜᵃ, ℑxyᶠᶜᵃ, ℑxyᶠᶠᵃ, ℑxyᶜᶠᵃ, ℑxzᶜᵃᶜ, ℑxzᶠᵃᶜ, ℑxzᶠᵃᶠ, ℑxzᶜᵃᶠ, ℑyzᵃᶜᶜ, ℑyzᵃᶠᶜ, ℑyzᵃᶠᶠ, ℑyzᵃᶜᶠ,
    ℑxyzᶜᶜᶠ, ℑxyzᶜᶠᶜ, ℑxyzᶠᶜᶜ, ℑxyzᶜᶠᶠ, ℑxyzᶠᶜᶠ, ℑxyzᶠᶠᶜ,
    hdivᶜᶜᵃ, divᶜᶜᶜ, div_xyᶜᶜᵃ, div_xzᶜᵃᶜ, div_yzᵃᶜᶜ,
    ∇², ∇²hᶜᶜᵃ, ∇²hᶠᶜᵃ, ∇²hᶜᶠᵃ, ∇⁴hᶜᶜᵃ, ∇⁴hᶠᶜᵃ, ∇⁴hᶜᶠᵃ

#####
##### Convinient aliases
#####

using Oceananigans.Grids: AbstractGrid, RegularCartesianGrid

const AG  = AbstractGrid
const RCG = RegularCartesianGrid

include("areas_and_volumes.jl")
include("difference_operators.jl")
include("derivative_operators.jl")
include("interpolation_operators.jl")
include("divergence_operators.jl")
include("laplacian_operators.jl")
include("interpolation_utils.jl")

end
