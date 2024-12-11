using Oceananigans.Grids: AbstractZStarGrid
using Oceananigans.Operators

import Oceananigans.Grids: dynamic_column_depthᶜᶜᵃ, 
                           dynamic_column_depthᶜᶠᵃ,
                           dynamic_column_depthᶠᶜᵃ,
                           dynamic_column_depthᶠᶠᵃ

import Oceananigans.Operators: σⁿ, σ⁻, ∂t_σ

const ZStarImmersedGrid   = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:AbstractZStarGrid}
const ZStarGridOfSomeKind = Union{ZStarImmersedGrid, AbstractZStarGrid}

@inline dynamic_column_depthᶜᶜᵃ(i, j, k, grid::ZStarGridOfSomeKind, η) = @inbounds static_column_depthᶜᶜᵃ(i, j, grid) +      η[i, j, k]
@inline dynamic_column_depthᶜᶠᵃ(i, j, k, grid::ZStarGridOfSomeKind, η) = @inbounds static_column_depthᶠᶜᵃ(i, j, grid) +  ℑyᵃᶠᵃ(i, j, k, grid, η)
@inline dynamic_column_depthᶠᶜᵃ(i, j, k, grid::ZStarGridOfSomeKind, η) = @inbounds static_column_depthᶜᶠᵃ(i, j, grid) +  ℑxᶠᵃᵃ(i, j, k, grid, η)
@inline dynamic_column_depthᶠᶠᵃ(i, j, k, grid::ZStarGridOfSomeKind, η) = @inbounds static_column_depthᶠᶠᵃ(i, j, grid) + ℑxyᶠᶠᵃ(i, j, k, grid, η)

# Convenience methods
@inline dynamic_column_depthᶜᶜᵃ(i, j, grid) = static_column_depthᶜᶜᵃ(i, j, grid)
@inline dynamic_column_depthᶜᶠᵃ(i, j, grid) = static_column_depthᶜᶠᵃ(i, j, grid)
@inline dynamic_column_depthᶠᶜᵃ(i, j, grid) = static_column_depthᶠᶜᵃ(i, j, grid)
@inline dynamic_column_depthᶠᶠᵃ(i, j, grid) = static_column_depthᶠᶠᵃ(i, j, grid)

@inline dynamic_column_depthᶜᶜᵃ(i, j, grid::ZStarGridOfSomeKind) = dynamic_column_depthᶜᶜᵃ(i, j, 1, grid, grid.z.ηⁿ)
@inline dynamic_column_depthᶜᶠᵃ(i, j, grid::ZStarGridOfSomeKind) = dynamic_column_depthᶜᶠᵃ(i, j, 1, grid, grid.z.ηⁿ)
@inline dynamic_column_depthᶠᶜᵃ(i, j, grid::ZStarGridOfSomeKind) = dynamic_column_depthᶠᶜᵃ(i, j, 1, grid, grid.z.ηⁿ)
@inline dynamic_column_depthᶠᶠᵃ(i, j, grid::ZStarGridOfSomeKind) = dynamic_column_depthᶠᶠᵃ(i, j, 1, grid, grid.z.ηⁿ)

# Fallbacks
@inline σⁿ(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = σⁿ(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)
@inline σ⁻(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = σ⁻(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)

@inline ∂t_σ(i, j, k, ibg::IBG) = ∂t_σ(i, j, k, ibg.underlying_grid)
