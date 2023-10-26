"""
    Γᶠᶠᶜ(i, j, k, grid, u, v)

The circulation around a `Face`-`Face`-`Center` cell that is associated with horizontal
velocities ``u`` and ``v``.
"""
@inline Γᶠᶠᶜ(i, j, k, grid, u, v) = δxᶠᶠᶜ(i, j, k, grid, Δy_qᶜᶠᶜ, v) - δyᶠᶠᶜ(i, j, k, grid, Δx_qᶠᶜᶜ, u)

"""
    ζ₃ᶠᶠᶜ(i, j, k, grid, u, v)

The vertical vorticity associated with horizontal velocities ``u`` and ``v``.
"""
@inline ζ₃ᶠᶠᶜ(i, j, k, grid, u, v) = Γᶠᶠᶜ(i, j, k, grid, u, v) / Azᶠᶠᶜ(i, j, k, grid)

#####
##### Vertical circulation at the corners of the cubed sphere needs to treated in a special manner.
##### See: https://github.com/CliMA/Oceananigans.jl/issues/1584
#####

@inline function ζ₃ᶠᶠᶜ(i, j, k, grid::OrthogonalSphericalShellGrid{FT}, u, v) where FT
    scaling = ifelse(on_horizontal_corner(i, j, grid), convert(FT, 4/3), 1)
    return scaling * Γᶠᶠᶜ(i, j, k, grid, u, v) / Azᶠᶠᶜ(i, j, k, grid)
end

# Corners
@inline on_south_west_corner(i, j, grid) = (i == 1) & (j == 1)
@inline on_south_east_corner(i, j, grid) = (i == grid.Nx+1) & (j == 1)
@inline on_north_east_corner(i, j, grid) = (i == grid.Nx+1) & (j == grid.Ny+1)
@inline on_north_west_corner(i, j, grid) = (i == 1) & (j == grid.Ny+1)

""" Return true if `i`, `j` indices correspond to a horizontal corner of the `grid`. """
@inline on_horizontal_corner(i, j, grid) = on_south_west_corner(i, j, grid) |
                                           on_south_east_corner(i, j, grid) |
                                           on_north_west_corner(i, j, grid) |
                                           on_north_east_corner(i, j, grid)

@inline Γᶠᶠᶜ_sw_or_nw(i, j, k, grid, u, v) =   Δy_qᶜᶠᶜ(i, j, k, grid, v)   - Δx_qᶠᶜᶜ(i, j, k, grid, u) + Δx_qᶠᶜᶜ(i, j-1, k, grid, u)
@inline Γᶠᶠᶜ_se_or_ne(i, j, k, grid, u, v) = - Δy_qᶜᶠᶜ(i-1, j, k, grid, v) - Δx_qᶠᶜᶜ(i, j, k, grid, u) + Δx_qᶠᶜᶜ(i, j-1, k, grid, u)

@inline function Γᶠᶠᶜ(i, j, k, grid::OrthogonalSphericalShellGrid, u, v)
    on_sw_or_nw = on_south_west_corner(i, j, grid) | on_north_west_corner(i, j, grid)
    on_se_or_ne = on_south_east_corner(i, j, grid) | on_north_east_corner(i, j, grid)

    return ifelse(on_sw_or_nw, Γᶠᶠᶜ_sw_or_nw(i, j, k, grid, u, v),
                  ifelse(on_se_or_ne, Γᶠᶠᶜ_se_or_ne(i, j, k, grid, u, v),
                         δxᶠᶠᶜ(i, j, k, grid, Δy_qᶜᶠᶜ, v) - δyᶠᶠᶜ(i, j, k, grid, Δx_qᶠᶜᶜ, u)))
end
