using Oceananigans.Grids: ZStarUnderlyingGrid
import Oceananigans.Grids: znode

#####
##### ZStar-specific vertical spacing functions
#####

const C = Center
const F = Face

const ZSG = ZStarUnderlyingGrid

@inline dynamic_column_depthᶜᶜᵃ(i, j, grid, η) = static_column_depthᶜᶜᵃ(i, j, grid) 
@inline dynamic_column_depthᶜᶠᵃ(i, j, grid, η) = static_column_depthᶜᶠᵃ(i, j, grid) 
@inline dynamic_column_depthᶠᶜᵃ(i, j, grid, η) = static_column_depthᶠᶜᵃ(i, j, grid) 
@inline dynamic_column_depthᶠᶠᵃ(i, j, grid, η) = static_column_depthᶠᶠᵃ(i, j, grid) 

# Fallbacks
@inline e₃ⁿ(i, j, k, grid, ℓx, ℓy, ℓz) = one(grid)
@inline e₃⁻(i, j, k, grid, ℓx, ℓy, ℓz) = one(grid)

@inline e₃ⁿ(i, j, k, grid::ZSG, ::C, ::C, ℓz) = dynamic_bottom_heightᶜᶜᵃ(i, j, 1, grid, grid.zᵃᵃᶠ.ηⁿ) / static_bottom_heightᶜᶜᵃ(i, j, grid)
@inline e₃ⁿ(i, j, k, grid::ZSG, ::F, ::C, ℓz) = dynamic_bottom_heightᶠᶜᵃ(i, j, 1, grid, grid.zᵃᵃᶠ.ηⁿ) / static_bottom_heightᶠᶜᵃ(i, j, grid)
@inline e₃ⁿ(i, j, k, grid::ZSG, ::C, ::F, ℓz) = dynamic_bottom_heightᶜᶠᵃ(i, j, 1, grid, grid.zᵃᵃᶠ.ηⁿ) / static_bottom_heightᶜᶠᵃ(i, j, grid)
@inline e₃ⁿ(i, j, k, grid::ZSG, ::F, ::F, ℓz) = dynamic_bottom_heightᶠᶠᵃ(i, j, 1, grid, grid.zᵃᵃᶠ.ηⁿ) / static_bottom_heightᶠᶠᵃ(i, j, grid)

@inline e₃⁻(i, j, k, grid::ZSG, ::C, ::C, ℓz) = dynamic_bottom_heightᶜᶜᵃ(i, j, 1, grid, grid.zᵃᵃᶠ.η⁻) / static_bottom_heightᶜᶜᵃ(i, j, grid)
@inline e₃⁻(i, j, k, grid::ZSG, ::F, ::C, ℓz) = dynamic_bottom_heightᶠᶜᵃ(i, j, 1, grid, grid.zᵃᵃᶠ.η⁻) / static_bottom_heightᶠᶜᵃ(i, j, grid)
@inline e₃⁻(i, j, k, grid::ZSG, ::C, ::F, ℓz) = dynamic_bottom_heightᶜᶠᵃ(i, j, 1, grid, grid.zᵃᵃᶠ.η⁻) / static_bottom_heightᶜᶠᵃ(i, j, grid)
@inline e₃⁻(i, j, k, grid::ZSG, ::F, ::F, ℓz) = dynamic_bottom_heightᶠᶠᵃ(i, j, 1, grid, grid.zᵃᵃᶠ.η⁻) / static_bottom_heightᶠᶠᵃ(i, j, grid)

@inline ∂t_grid(i, j, k, grid) = zero(grid)
@inline ∂t_grid(i, j, k, grid::ZSG) = @inbounds grid.Δzᵃᵃᶜ.∂t_s[i, j] 

@inline ηⁿ(i, j, grid, ℓx, ℓy) = one(grid)
@inline η⁻(i, j, grid, ℓx, ℓy) = one(grid)

@inline ηⁿ(i, j, grid::ZSG, ::C, ::C) =  @inbounds grid.zᵃᵃᶠ.ηⁿ[i, j, 1]
@inline ηⁿ(i, j, grid::ZSG, ::F, ::C) =  ℑxᶠᵃᵃ(i, j, 1, grid.zᵃᵃᶠ.η)
@inline ηⁿ(i, j, grid::ZSG, ::C, ::F) =  ℑyᵃᶠᵃ(i, j, 1, grid.zᵃᵃᶠ.η)
@inline ηⁿ(i, j, grid::ZSG, ::F, ::F) = ℑxyᶠᶠᵃ(i, j, 1, grid.zᵃᵃᶠ.η)

@inline η⁻(i, j, grid::ZSG, ::C, ::C) =  @inbounds grid.zᵃᵃᶠ.η⁻[i, j, 1]
@inline η⁻(i, j, grid::ZSG, ::F, ::C) =  ℑxᶠᵃᵃ(i, j, 1, grid.zᵃᵃᶠ.η)
@inline η⁻(i, j, grid::ZSG, ::C, ::F) =  ℑyᵃᶠᵃ(i, j, 1, grid.zᵃᵃᶠ.η)
@inline η⁻(i, j, grid::ZSG, ::F, ::F) = ℑxyᶠᶠᵃ(i, j, 1, grid.zᵃᵃᶠ.η)


# rnode for an ZStarUnderlyingGrid grid is scaled 
# TODO: fix this when bottom height is implemented
@inline znode(i, j, k, grid::ZSG, ℓx, ℓx, ℓx) = rnode(i, j, k, grid, ℓx, ℓy, ℓz) * e₃ⁿ(i, j, k, grid, c, c, c) + ηⁿ(i, j, grid, ℓx, ℓy)