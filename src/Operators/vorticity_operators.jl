""" Vertical circulation associated with horizontal velocities u, v. """
@inline Γᶠᶠᵃ(i, j, k, grid, u, v) = Δy_vᶜᶠᵃ(i, j, k, grid, v) - Δy_vᶜᶠᵃ(i-1, j, k, grid, v) - Δx_uᶠᶜᵃ(i, j, k, grid, u) + Δx_uᶠᶜᵃ(i, j-1, k, grid, u)

""" Vertical vorticity associated with horizontal velocities u, v. """
@inline ζ₃ᶠᶠᵃ(i, j, k, grid, u, v) = Γᶠᶠᵃ(i, j, k, grid, u, v) / Azᶠᶠᵃ(i, j, k, grid)

#####
##### Vertical circulation at the corners of the cubed sphere needs to treated in a special manner.
##### See: https://github.com/CliMA/Oceananigans.jl/issues/1584
#####

@inline function Γᶠᶠᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid, u, v)
    # South-west corner
    if i == 1 && j == 1
        return Δy_vᶜᶠᵃ(i, j, k, grid, v) - Δx_uᶠᶜᵃ(i, j, k, grid, u) + Δx_uᶠᶜᵃ(i, j-1, k, grid, u)

    # South-east corner
    elseif i == grid.Nx+1 && j == 1
        return - Δy_vᶜᶠᵃ(i, j-1, k, grid, v) - Δx_uᶠᶜᵃ(i, j, k, grid, u) + Δx_uᶠᶜᵃ(i, j-1, k, grid, u)

    # North-west corner
    elseif i == 1 && j == grid.Ny+1
        return Δy_vᶜᶠᵃ(i, j, k, grid, v) - Δx_uᶠᶜᵃ(i, j, k, grid, u) + Δx_uᶠᶜᵃ(i, j-1, k, grid, u)

    # North-east corner
    elseif i == grid.Nx+1 && j == grid.Ny+1
        return - Δy_vᶜᶠᵃ(i-1, j, k, grid, v) - Δx_uᶠᶜᵃ(i, j, k, grid, u) + Δx_uᶠᶜᵃ(i, j-1, k, grid, u)

    # Not a corner
    else
        return Δy_vᶜᶠᵃ(i, j, k, grid, v) - Δy_vᶜᶠᵃ(i-1, j, k, grid, v) - Δx_uᶠᶜᵃ(i, j, k, grid, u) + Δx_uᶠᶜᵃ(i, j-1, k, grid, u)
    end
end
