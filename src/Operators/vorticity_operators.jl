""" Vertical circulation associated with horizontal velocities u, v. """
@inline Γᶠᶠᶜ(i, j, k, grid, u, v) = δxᶠᶠᶜ(i, j, k, grid, Δy_qᶜᶠᶜ, v) - δyᶠᶠᶜ(i, j, k, grid, Δx_qᶠᶜᶜ, u)

"""
     ζ₃ᶠᶠᶜ(i, j, k, grid, u, v)

 The vertical vorticity associated with horizontal velocities ``u`` and ``v``.
 """
@inline ζ₃ᶠᶠᶜ(i, j, k, grid, u, v) = Γᶠᶠᶜ(i, j, k, grid, u, v) / Azᶠᶠᶜ(i, j, k, grid)

@inline function ζ₃ᶠᶠᶜ(i, j, k, grid::OrthogonalSphericalShellGrid{FT}, u, v) where FT
    scaling = ifelse(on_corner(i, j, grid), convert(FT, 4/3), one(grid))
    return scaling * Γᶠᶠᶜ(i, j, k, grid, u, v) / Azᶠᶠᶜ(i, j, k, grid)
end

# Corner
@inline on_corner(i, j, grid) = (i == 1 & j == 1) | (i == grid.Nx+1 & j == 1) | (i == 1 & j == grid.Ny+1) | (i == grid.Nx+1 & j == grid.Ny+1)

# South-west, south-east, north-west, north-east corners
@inline on_south_west_corner(i, j) = (i == 1 & j == 1)
@inline on_south_east_corner(i, j, grid) = (i == grid.Nx+1 & j == 1)
@inline on_north_east_corner(i, j, grid) = (i == grid.Nx+1 & j == grid.Ny+1)
@inline on_north_west_corner(i, j, grid) = (i == 1 & j == grid.Ny+1)

#####
##### Vertical circulation at the corners of the cubed sphere needs to treated in a special manner.
##### See: https://github.com/CliMA/Oceananigans.jl/issues/1584
#####

"""
     Γᶠᶠᶜ(i, j, k, grid, u, v)

 The vertical circulation associated with horizontal velocities ``u`` and ``v``.
 """
@inline Γᶠᶠᶜ(i, j, k, grid::OrthogonalSphericalShellGrid, u, v) = 

    ifelse(on_south_west_corner(i, j) | on_north_west_corner(i, j, grid),
           Δy_qᶜᶠᶜ(i, j, k, grid, v) - Δx_qᶠᶜᶜ(i, j, k, grid, u) + Δx_qᶠᶜᶜ(i, j-1, k, grid, u), 
           ifelse(on_south_east_corner(i, j, grid) | on_north_east_corner(i, j, grid),
                  - Δy_qᶜᶠᶜ(i-1, j, k, grid, v) - Δx_qᶠᶜᶜ(i, j, k, grid, u) + Δx_qᶠᶜᶜ(i, j-1, k, grid, u), 
                  δxᶠᵃᵃ(i, j, k, grid, Δy_qᶜᶠᶜ, v) - δyᵃᶠᵃ(i, j, k, grid, Δx_qᶠᶜᶜ, u)))