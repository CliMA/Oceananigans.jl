module Operators

export
    Δx, Δy, ΔzF, ΔzC,
    Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶠᶠᵃ, Δxᶜᶠᵃ,
    Δyᶜᶜᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ, Δyᶜᶠᵃ,
    Axᵃᵃᶜ, Axᵃᵃᶠ, Axᶠᶜᶜ,
    Ayᵃᵃᶜ, Ayᵃᵃᶠ, Ayᶜᶠᶜ,
    Azᵃᵃᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ,
    Vᵃᵃᶜ, Vᵃᵃᶠ, Vᶜᶜᶜ,
    δxᶜᵃᵃ, δxᶠᵃᵃ, δyᵃᶜᵃ, δyᵃᶠᵃ, δzᵃᵃᶜ, δzᵃᵃᶠ,
    Ax_ψᵃᵃᶠ, Ax_ψᵃᵃᶜ, Ay_ψᵃᵃᶠ, Ay_ψᵃᵃᶜ, Az_ψᵃᵃᵃ,
    δᴶxᶜᵃᶜ, δᴶxᶜᵃᶠ, δᴶxᶠᵃᶜ, δᴶxᶠᵃᶠ, δᴶyᵃᶜᶜ, δᴶyᵃᶜᶠ, δᴶyᵃᶠᶜ, δᴶyᵃᶠᶠ, δᴶzᵃᵃᶜ, δᴶzᵃᵃᶠ,
    ∂xᶜᵃᵃ, ∂xᶠᵃᵃ, ∂yᵃᶜᵃ, ∂yᵃᶠᵃ, ∂zᵃᵃᶜ, ∂zᵃᵃᶠ,
    ∂xᶜᶜᵃ, ∂xᶠᶠᵃ, ∂xᶠᶜᵃ, ∂xᶜᶠᵃ, ∂yᶜᶜᵃ, ∂yᶠᶠᵃ, ∂yᶠᶜᵃ, ∂yᶜᶠᵃ,
    ∂²xᶜᵃᵃ, ∂²xᶠᵃᵃ, ∂²yᵃᶜᵃ, ∂²yᵃᶠᵃ, ∂²zᵃᵃᶜ, ∂²zᵃᵃᶠ,
    ∂⁴xᶜᵃᵃ, ∂⁴xᶠᵃᵃ, ∂⁴yᵃᶜᵃ, ∂⁴yᵃᶠᵃ, ∂⁴zᵃᵃᶜ, ∂⁴zᵃᵃᶠ,
    ℑxᶜᵃᵃ, ℑxᶠᵃᵃ, ℑyᵃᶜᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶜ, ℑzᵃᵃᶠ,
    ℑᴶxᶜᵃᵃ, ℑᴶxᶠᵃᵃ, ℑᴶyᵃᶜᵃ, ℑᴶyᵃᶠᵃ, ℑᴶzᵃᵃᶜ, ℑᴶzᵃᵃᶠ,
    ℑxyᶜᶜᵃ, ℑxyᶠᶜᵃ, ℑxyᶠᶠᵃ, ℑxyᶜᶠᵃ, ℑxzᶜᵃᶜ, ℑxzᶠᵃᶜ, ℑxzᶠᵃᶠ, ℑxzᶜᵃᶠ, ℑyzᵃᶜᶜ, ℑyzᵃᶠᶜ, ℑyzᵃᶠᶠ, ℑyzᵃᶜᶠ,
    ℑxyzᶜᶜᶠ, ℑxyzᶜᶠᶜ, ℑxyzᶠᶜᶜ, ℑxyzᶜᶠᶠ, ℑxyzᶠᶜᶠ, ℑxyzᶠᶠᶜ,
    divᶜᶜᶜ, div_xyᶜᶜᵃ, div_xzᶜᵃᶜ, div_yzᵃᶜᶜ, ζ₃ᶠᶠᵃ,
    ∇², ∇²hᶜᶜᵃ, ∇²hᶠᶜᵃ, ∇²hᶜᶠᵃ, ∇⁴hᶜᶜᵃ, ∇⁴hᶠᶜᵃ, ∇⁴hᶜᶠᵃ

#####
##### Convinient aliases
#####

using Oceananigans.Grids: RegularRectilinearGrid, VerticallyStretchedRectilinearGrid, RegularLatitudeLongitudeGrid
using Oceananigans.Grids: AbstractGrid, AbstractRectilinearGrid, AbstractCurvilinearGrid, AbstractHorizontallyCurvilinearGrid

const AG  = AbstractGrid
const ARG = AbstractRectilinearGrid
const RCG = RegularRectilinearGrid
const ACG = AbstractCurvilinearGrid
const AHCG = AbstractHorizontallyCurvilinearGrid

include("areas_and_volumes.jl")
include("field_metric_products.jl")
include("difference_operators.jl")
include("derivative_operators.jl")
include("interpolation_operators.jl")
include("divergence_operators.jl")
include("vorticity_operators.jl")
include("laplacian_operators.jl")
include("interpolation_utils.jl")

end
