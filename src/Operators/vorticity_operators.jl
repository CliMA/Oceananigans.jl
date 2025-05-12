using Oceananigans.Grids: ConformalCubedSpherePanel

""" Vertical circulation associated with horizontal velocities u, v. """
@inline Γᶠᶠᶜ(i, j, k, grid, u, v) = δxᶠᶠᶜ(i, j, k, grid, Δy_qᶜᶠᶜ, v) - δyᶠᶠᶜ(i, j, k, grid, Δx_qᶠᶜᶜ, u)

"""
    ζ₃ᶠᶠᶜ(i, j, k, grid, u, v)

The vertical vorticity associated with horizontal velocities ``u`` and ``v``.
"""
@inline ζ₃ᶠᶠᶜ(i, j, k, grid, u, v) = Γᶠᶠᶜ(i, j, k, grid, u, v) * Az⁻¹ᶠᶠᶜ(i, j, k, grid)

# South-west, south-east, north-west, north-east corners
@inline on_south_west_corner(i, j, grid) = (i == 1) & (j == 1)
@inline on_south_east_corner(i, j, grid) = (i == grid.Nx+1) & (j == 1)
@inline on_north_east_corner(i, j, grid) = (i == grid.Nx+1) & (j == grid.Ny+1)
@inline on_north_west_corner(i, j, grid) = (i == 1) & (j == grid.Ny+1)

#####
##### Vertical circulation at the corners of the cubed sphere needs to treated in a special manner.
##### See: https://github.com/CliMA/Oceananigans.jl/issues/1584
#####

"""
    Γᶠᶠᶜ(i, j, k, grid, u, v)

The vertical circulation associated with horizontal velocities ``u`` and ``v``.
"""
@inline function Γᶠᶠᶜ(i, j, k, grid::ConformalCubedSpherePanel, u, v)
    Hx, Hy = grid.Hx, grid.Hy
    ip = ifelse(i == 1 - Hx, 1, i)
    jp = ifelse(j == 1 - Hy, 1, j)
    Γ = ifelse(on_south_west_corner(i, j, grid) | on_north_west_corner(i, j, grid),
               Δy_qᶜᶠᶜ(ip, jp, k, grid, v) - Δx_qᶠᶜᶜ(ip, jp, k, grid, u) + Δx_qᶠᶜᶜ(ip, jp-1, k, grid, u),
               ifelse(on_south_east_corner(i, j, grid) | on_north_east_corner(i, j, grid),
                      - Δy_qᶜᶠᶜ(ip-1, jp, k, grid, v) + Δx_qᶠᶜᶜ(ip, jp-1, k, grid, u) - Δx_qᶠᶜᶜ(ip, jp, k, grid, u),
                      δxᶠᶠᶜ(ip, jp, k, grid, Δy_qᶜᶠᶜ, v) - δyᶠᶠᶜ(ip, jp, k, grid, Δx_qᶠᶜᶜ, u)
                     )
              )
    return Γ
end
