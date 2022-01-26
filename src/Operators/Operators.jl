module Operators

export Δzᵃᵃᶜ, Δzᵃᵃᶠ, Δzᶠᶜᶜ, Δzᶜᶠᶜ, Δzᶠᶜᶠ, Δzᶜᶠᶠ
export Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶠᶠᵃ, Δxᶜᶠᵃ
export Δyᶜᶜᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ, Δyᶜᶠᵃ
export Axᵃᵃᶜ, Axᵃᵃᶠ, Axᶜᶜᶜ, Axᶠᶜᶜ, Axᶠᶠᶜ, Axᶠᶜᶠ, Axᶜᶠᶜ, Axᶜᶜᶠ
export Ayᵃᵃᶜ, Ayᵃᵃᶠ, Ayᶜᶜᶜ, Ayᶜᶠᶜ, Ayᶠᶠᶜ, Ayᶜᶠᶠ, Ayᶠᶜᶜ, Ayᶜᶜᶠ
export Azᵃᵃᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ
export Vᵃᵃᶜ, Vᵃᵃᶠ, Vᶜᶜᶜ, Vᶠᶜᶜ, Vᶜᶠᶜ, Vᶜᶜᶠ
export δxᶜᵃᵃ, δxᶠᵃᵃ, δyᵃᶜᵃ, δyᵃᶠᵃ, δzᵃᵃᶜ, δzᵃᵃᶠ
export Ax_ψᵃᵃᶠ, Ax_ψᵃᵃᶜ, Ay_ψᵃᵃᶠ, Ay_ψᵃᵃᶜ, Az_ψᵃᵃᵃ
export Ax_uᶠᶜᶜ, Ay_vᶜᶠᶜ, Az_wᶜᶜᵃ
export Ax_ζᶠᶠᶜ, Ay_ζᶠᶠᶜ, Ax_ηᶠᶜᶠ, Az_ηᶠᶜᵃ, Ay_ξᶜᶠᶠ, Az_ξᶜᶠᵃ
export Az_wᶜᶜᵃ, Az_ηᶠᶜᵃ, Az_ξᶜᶠᵃ
export Ax_cᶜᶜᶜ, Ay_cᶜᶜᶜ, Az_cᶜᶜᵃ
export δᴶxᶜᵃᶜ, δᴶxᶜᵃᶠ, δᴶxᶠᵃᶜ, δᴶxᶠᵃᶠ, δᴶyᵃᶜᶜ, δᴶyᵃᶜᶠ, δᴶyᵃᶠᶜ, δᴶyᵃᶠᶠ, δᴶzᵃᵃᶜ, δᴶzᵃᵃᶠ
export ∂xᶜᵃᵃ, ∂xᶠᵃᵃ, ∂yᵃᶜᵃ, ∂yᵃᶠᵃ, ∂zᵃᵃᶜ, ∂zᵃᵃᶠ, ∂zᶠᶜᶜ, ∂zᶜᶠᶜ, ∂zᶠᶜᶠ, ∂zᶜᶠᶠ
export ∂xᶜᶜᵃ, ∂xᶠᶠᵃ, ∂xᶠᶜᵃ, ∂xᶜᶠᵃ, ∂yᶜᶜᵃ, ∂yᶠᶠᵃ, ∂yᶠᶜᵃ, ∂yᶜᶠᵃ
export ∂²xᶜᵃᵃ, ∂²xᶠᵃᵃ, ∂²yᵃᶜᵃ, ∂²yᵃᶠᵃ, ∂²zᵃᵃᶜ, ∂²zᵃᵃᶠ
export ∂³zᵃᵃᶜ, ∂³zᵃᵃᶠ
export ∂⁴xᶜᵃᵃ, ∂⁴xᶠᵃᵃ, ∂⁴yᵃᶜᵃ, ∂⁴yᵃᶠᵃ, ∂⁴zᵃᵃᶜ, ∂⁴zᵃᵃᶠ
export ℑxᶜᵃᵃ, ℑxᶠᵃᵃ, ℑyᵃᶜᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶜ, ℑzᵃᵃᶠ
export ℑᴶxᶜᵃᵃ, ℑᴶxᶠᵃᵃ, ℑᴶyᵃᶜᵃ, ℑᴶyᵃᶠᵃ, ℑᴶzᵃᵃᶜ, ℑᴶzᵃᵃᶠ
export ℑxyᶜᶜᵃ, ℑxyᶠᶜᵃ, ℑxyᶠᶠᵃ, ℑxyᶜᶠᵃ, ℑxzᶜᵃᶜ, ℑxzᶠᵃᶜ, ℑxzᶠᵃᶠ, ℑxzᶜᵃᶠ, ℑyzᵃᶜᶜ, ℑyzᵃᶠᶜ, ℑyzᵃᶠᶠ, ℑyzᵃᶜᶠ
export ℑxyzᶜᶜᶠ, ℑxyzᶜᶠᶜ, ℑxyzᶠᶜᶜ, ℑxyzᶜᶠᶠ, ℑxyzᶠᶜᶠ, ℑxyzᶠᶠᶜ
export divᶜᶜᶜ, div_xyᶜᶜᵃ, div_xzᶜᵃᶜ, div_yzᵃᶜᶜ, ζ₃ᶠᶠᵃ
export ∇²ᶜᶜᶜ, ∇²hᶜᶜᶜ, ∇²hᶠᶜᶜ, ∇²hᶜᶠᶜ

using Oceananigans.Grids

#####
##### Convenient aliases
#####

const AG   = AbstractGrid
const ARG  = AbstractRectilinearGrid
const RCG  = RectilinearGrid
const ACG  = AbstractCurvilinearGrid
const AHCG = AbstractHorizontallyCurvilinearGrid

include("difference_operators.jl")
include("interpolation_operators.jl")
include("interpolation_utils.jl")

include("spacings_and_areas_and_volumes.jl")
include("products_between_fields_and_grid_metrics.jl")

include("derivative_operators.jl")
include("divergence_operators.jl")
include("vorticity_operators.jl")
include("laplacian_operators.jl")

end
