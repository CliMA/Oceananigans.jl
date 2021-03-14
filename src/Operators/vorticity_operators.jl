""" Vertical circulation associated with horizontal velocities u, v. """
@inline Γᶠᶠᵃ(i, j, k, grid, u, v) = δxᶠᵃᵃ(i, j, k, grid, Δy_vᶜᶠᵃ, v) - δyᵃᶠᵃ(i, j, k, grid, Δx_uᶠᶜᵃ, u)

""" Vertical vorticity associated with horizontal velocities u, v. """
@inline ζ₃ᶠᶠᵃ(i, j, k, grid, u, v) = Γᶠᶠᵃ(i, j, k, grid, u, v) / Azᶠᶠᵃ(i, j, k, grid)
