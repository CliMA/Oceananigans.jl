""" Vertical circulation associated with horizontal velocities u, v. """
@inline Γᶠᶠᵃ(i, j, k, grid, u, v) = δxᶠᵃᵃ(i, j, k, grid, Δy_vᶜᶠᵃ, v) - δyᵃᶠᵃ(i, j, k, grid, Δx_uᶠᶜᵃ, u)

""" Vertical vorticity associated with horizontal velocities u, v. """
@inline ζ₃ᶠᶠᵃ(i, j, k, grid, u, v) = Γᶠᶠᵃ(i, j, k, grid, u, v) / Azᶠᶠᵃ(i, j, k, grid)

#####
##### Vertical circulation at the corners of the cubed sphere needs to treated in a special manner.
##### See: https://github.com/CliMA/Oceananigans.jl/issues/1584
#####

@inline function Γᶠᶠᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid, u, v)
    # South-west corner
    if i == 1 && j == 1
        return Δy_vᶜᶠᵃ(i, j, k, grid, v) - δyᵃᶠᵃ(i, j, k, grid, Δx_uᶠᶜᵃ, u)

    # South-east corner
    elseif i == grid.Nx && j == 1
        return - Δy_vᶜᶠᵃ(i, j-1, k, grid, v) - δyᵃᶠᵃ(i, j, k, grid, Δx_uᶠᶜᵃ, u)

    # North-west corner
    elseif i == 1 && j == grid.Ny
        return Δy_vᶜᶠᵃ(i, j, k, grid, v) - δyᵃᶠᵃ(i, j, k, grid, Δx_uᶠᶜᵃ, u)

    # North-east corner
    elseif i == grid.Nx && j == grid.Ny
        return - Δy_vᶜᶠᵃ(i-1, j, k, grid, v) - δyᵃᶠᵃ(i, j, k, grid, Δx_uᶠᶜᵃ, u)

    # Not at a corner
    else
        return δxᶠᵃᵃ(i, j, k, grid, Δy_vᶜᶠᵃ, v) - δyᵃᶠᵃ(i, j, k, grid, Δx_uᶠᶜᵃ, u)
    end
end
