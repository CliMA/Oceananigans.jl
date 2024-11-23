using Oceananigans.Grids: ZStarGrid
using Oceananigans.Operators

import Oceananigans.Grids: dynamic_column_depthᶜᶜᵃ, 
                           dynamic_column_depthᶜᶠᵃ,
                           dynamic_column_depthᶠᶜᵃ,
                           dynamic_column_depthᶠᶠᵃ

const ZStarImmersedGrid   = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:ZStarGrid}
const ZStarGridOfSomeKind = Union{ZStarImmersedGrid, ZStarGrid}

@inline dynamic_column_depthᶜᶜᵃ(i, j, grid::ZStarGridOfSomeKind, η) = @inbounds static_column_depthᶜᶜᵃ(i, j, grid) +      η[i, j, grid.Nz+1]
@inline dynamic_column_depthᶜᶠᵃ(i, j, grid::ZStarGridOfSomeKind, η) = @inbounds static_column_depthᶜᶠᵃ(i, j, grid) +  ℑxᶠᵃᵃ(i, j, grid.Nz+1, grid, η)
@inline dynamic_column_depthᶠᶜᵃ(i, j, grid::ZStarGridOfSomeKind, η) = @inbounds static_column_depthᶠᶜᵃ(i, j, grid) +  ℑyᵃᶠᵃ(i, j, grid.Nz+1, grid, η)
@inline dynamic_column_depthᶠᶠᵃ(i, j, grid::ZStarGridOfSomeKind, η) = @inbounds static_column_depthᶠᶠᵃ(i, j, grid) + ℑxyᶠᶠᵃ(i, j, grid.Nz+1, grid, η)

# Convenience
@inline dynamic_column_depthᶜᶜᵃ(i, j, grid::ZStarGridOfSomeKind) = dynamic_column_depthᶜᶜᵃ(i, j, grid, grid.z.ηⁿ)
@inline dynamic_column_depthᶜᶠᵃ(i, j, grid::ZStarGridOfSomeKind) = dynamic_column_depthᶜᶠᵃ(i, j, grid, grid.z.ηⁿ)
@inline dynamic_column_depthᶠᶜᵃ(i, j, grid::ZStarGridOfSomeKind) = dynamic_column_depthᶠᶜᵃ(i, j, grid, grid.z.ηⁿ)
@inline dynamic_column_depthᶠᶠᵃ(i, j, grid::ZStarGridOfSomeKind) = dynamic_column_depthᶠᶠᵃ(i, j, grid, grid.z.ηⁿ)

# Fallbacks
@inline e₃ⁿ(i, j, k, grid::IBG, ℓx, ℓy, ℓz) = e₃ⁿ(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)
@inline e₃⁻(i, j, k, grid::IBG, ℓx, ℓy, ℓz) = e₃⁻(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)

@inline ∂t_e₃(i, j, k, grid::IBG) = ∂t_e₃(i, j, k, grid.underlying_grid)
