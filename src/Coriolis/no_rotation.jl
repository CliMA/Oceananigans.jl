#####
##### Functions for non-rotating models
#####
@inline fᶠᶠᵃ(i, j, k, grid, ::Nothing) = zero(grid)

@inline x_f_cross_U(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U) where FT = zero(FT)
@inline y_f_cross_U(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U) where FT = zero(FT)
@inline z_f_cross_U(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U) where FT = zero(FT)
