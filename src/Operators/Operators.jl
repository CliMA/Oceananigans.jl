module Operators

export Δxᶠᶠᶠ, Δxᶠᶠᶜ, Δxᶠᶜᶠ, Δxᶠᶜᶜ, Δxᶜᶠᶠ, Δxᶜᶠᶜ, Δxᶜᶜᶠ, Δxᶜᶜᶜ
export Δyᶠᶠᶠ, Δyᶠᶠᶜ, Δyᶠᶜᶠ, Δyᶠᶜᶜ, Δyᶜᶠᶠ, Δyᶜᶠᶜ, Δyᶜᶜᶠ, Δyᶜᶜᶜ
export Δzᶠᶠᶠ, Δzᶠᶠᶜ, Δzᶠᶜᶠ, Δzᶠᶜᶜ, Δzᶜᶠᶠ, Δzᶜᶠᶜ, Δzᶜᶜᶠ, Δzᶜᶜᶜ

export Axᶠᶠᶠ, Axᶠᶠᶜ, Axᶠᶜᶠ, Axᶠᶜᶜ, Axᶜᶠᶠ, Axᶜᶠᶜ, Axᶜᶜᶠ, Axᶜᶜᶜ
export Ayᶠᶠᶠ, Ayᶠᶠᶜ, Ayᶠᶜᶠ, Ayᶠᶜᶜ, Ayᶜᶠᶠ, Ayᶜᶠᶜ, Ayᶜᶜᶠ, Ayᶜᶜᶜ
export Azᶠᶠᶠ, Azᶠᶠᶜ, Azᶠᶜᶠ, Azᶠᶜᶜ, Azᶜᶠᶠ, Azᶜᶠᶜ, Azᶜᶜᶠ, Azᶜᶜᶜ

export Vᶠᶠᶠ, Vᶠᶠᶜ, Vᶠᶜᶠ, Vᶠᶜᶜ, Vᶜᶠᶠ, Vᶜᶠᶜ, Vᶜᶜᶠ, Vᶜᶜᶜ

export δxᶜᵃᵃ, δxᶠᵃᵃ, δyᵃᶜᵃ, δyᵃᶠᵃ, δzᵃᵃᶜ, δzᵃᵃᶠ

export Ax_qᶠᶠᶠ, Ax_qᶠᶠᶜ, Ax_qᶠᶜᶠ, Ax_qᶠᶜᶜ, Ax_qᶜᶠᶠ, Ax_qᶜᶠᶜ, Ax_qᶜᶜᶠ, Ax_qᶜᶜᶜ
export Ay_qᶠᶠᶠ, Ay_qᶠᶠᶜ, Ay_qᶠᶜᶠ, Ay_qᶠᶜᶜ, Ay_qᶜᶠᶠ, Ay_qᶜᶠᶜ, Ay_qᶜᶜᶠ, Ay_qᶜᶜᶜ
export Az_qᶠᶠᶠ, Az_qᶠᶠᶜ, Az_qᶠᶜᶠ, Az_qᶠᶜᶜ, Az_qᶜᶠᶠ, Az_qᶜᶠᶜ, Az_qᶜᶜᶠ, Az_qᶜᶜᶜ

export δᴶxᶜᵃᶜ, δᴶxᶜᵃᶠ, δᴶxᶠᵃᶜ, δᴶxᶠᵃᶠ, δᴶyᵃᶜᶜ, δᴶyᵃᶜᶠ, δᴶyᵃᶠᶜ, δᴶyᵃᶠᶠ, δᴶzᵃᵃᶜ, δᴶzᵃᵃᶠ

export ∂xᶠᶠᶠ, ∂xᶠᶠᶜ, ∂xᶠᶜᶠ, ∂xᶠᶜᶜ, ∂xᶜᶠᶠ, ∂xᶜᶠᶜ, ∂xᶜᶜᶠ, ∂xᶜᶜᶜ
export ∂yᶠᶠᶠ, ∂yᶠᶠᶜ, ∂yᶠᶜᶠ, ∂yᶠᶜᶜ, ∂yᶜᶠᶠ, ∂yᶜᶠᶜ, ∂yᶜᶜᶠ, ∂yᶜᶜᶜ
export ∂zᶠᶠᶠ, ∂zᶠᶠᶜ, ∂zᶠᶜᶠ, ∂zᶠᶜᶜ, ∂zᶜᶠᶠ, ∂zᶜᶠᶜ, ∂zᶜᶜᶠ, ∂zᶜᶜᶜ

export ∂²xᶠᶠᶠ, ∂²xᶠᶠᶜ, ∂²xᶠᶜᶠ, ∂²xᶠᶜᶜ, ∂²xᶜᶠᶠ, ∂²xᶜᶠᶜ, ∂²xᶜᶜᶠ, ∂²xᶜᶜᶜ
export ∂²yᶠᶠᶠ, ∂²yᶠᶠᶜ, ∂²yᶠᶜᶠ, ∂²yᶠᶜᶜ, ∂²yᶜᶠᶠ, ∂²yᶜᶠᶜ, ∂²yᶜᶜᶠ, ∂²yᶜᶜᶜ
export ∂²zᶠᶠᶠ, ∂²zᶠᶠᶜ, ∂²zᶠᶜᶠ, ∂²zᶠᶜᶜ, ∂²zᶜᶠᶠ, ∂²zᶜᶠᶜ, ∂²zᶜᶜᶠ, ∂²zᶜᶜᶜ

export ∂³xᶠᶠᶠ, ∂³xᶠᶠᶜ, ∂³xᶠᶜᶠ, ∂³xᶠᶜᶜ, ∂³xᶜᶠᶠ, ∂³xᶜᶠᶜ, ∂³xᶜᶜᶠ, ∂³xᶜᶜᶜ
export ∂³yᶠᶠᶠ, ∂³yᶠᶠᶜ, ∂³yᶠᶜᶠ, ∂³yᶠᶜᶜ, ∂³yᶜᶠᶠ, ∂³yᶜᶠᶜ, ∂³yᶜᶜᶠ, ∂³yᶜᶜᶜ
export ∂³zᶠᶠᶠ, ∂³zᶠᶠᶜ, ∂³zᶠᶜᶠ, ∂³zᶠᶜᶜ, ∂³zᶜᶠᶠ, ∂³zᶜᶠᶜ, ∂³zᶜᶜᶠ, ∂³zᶜᶜᶜ

export ∂⁴xᶠᶠᶠ, ∂⁴xᶠᶠᶜ, ∂⁴xᶠᶜᶠ, ∂⁴xᶠᶜᶜ, ∂⁴xᶜᶠᶠ, ∂⁴xᶜᶠᶜ, ∂⁴xᶜᶜᶠ, ∂⁴xᶜᶜᶜ
export ∂⁴yᶠᶠᶠ, ∂⁴yᶠᶠᶜ, ∂⁴yᶠᶜᶠ, ∂⁴yᶠᶜᶜ, ∂⁴yᶜᶠᶠ, ∂⁴yᶜᶠᶜ, ∂⁴yᶜᶜᶠ, ∂⁴yᶜᶜᶜ
export ∂⁴zᶠᶠᶠ, ∂⁴zᶠᶠᶜ, ∂⁴zᶠᶜᶠ, ∂⁴zᶠᶜᶜ, ∂⁴zᶜᶠᶠ, ∂⁴zᶜᶠᶜ, ∂⁴zᶜᶜᶠ, ∂⁴zᶜᶜᶜ

export ℑxᶜᵃᵃ, ℑxᶠᵃᵃ, ℑyᵃᶜᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶜ, ℑzᵃᵃᶠ
export ℑᴶxᶜᵃᵃ, ℑᴶxᶠᵃᵃ, ℑᴶyᵃᶜᵃ, ℑᴶyᵃᶠᵃ, ℑᴶzᵃᵃᶜ, ℑᴶzᵃᵃᶠ
export ℑxyᶜᶜᵃ, ℑxyᶠᶜᵃ, ℑxyᶠᶠᵃ, ℑxyᶜᶠᵃ, ℑxzᶜᵃᶜ, ℑxzᶠᵃᶜ, ℑxzᶠᵃᶠ, ℑxzᶜᵃᶠ, ℑyzᵃᶜᶜ, ℑyzᵃᶠᶜ, ℑyzᵃᶠᶠ, ℑyzᵃᶜᶠ
export ℑxyzᶜᶜᶠ, ℑxyzᶜᶠᶜ, ℑxyzᶠᶜᶜ, ℑxyzᶜᶠᶠ, ℑxyzᶠᶜᶠ, ℑxyzᶠᶠᶜ
export divᶜᶜᶜ, div_xyᶜᶜᵃ, div_xzᶜᵃᶜ, div_yzᵃᶜᶜ, ζ₃ᶠᶠᶜ
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
