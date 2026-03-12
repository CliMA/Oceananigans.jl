using Oceananigans.Grids: peripheral_node, Face, Center

""" Vertical circulation associated with horizontal velocities u, v. """
@inline Γᶠᶠᶜ(i, j, k, grid, u, v) = δxᶠᶠᶜ(i, j, k, grid, Δy_qᶜᶠᶜ, v) - δyᶠᶠᶜ(i, j, k, grid, Δx_qᶠᶜᶜ, u)

"""
    ζ₃ᶠᶠᶜ(i, j, k, grid, u, v)

The vertical vorticity associated with horizontal velocities ``u`` and ``v``.
"""
@inline function ζ₃ᶠᶠᶜ(i, j, k, grid, u, v) 
    ζᶠᶠᶜ = Γᶠᶠᶜ(i, j, k, grid, u, v) * Az⁻¹ᶠᶠᶜ(i, j, k, grid)
    immersed = peripheral_node(i, j, k, grid, Face(), Face(), Center())
    return ifelse(immersed, zero(grid), ζᶠᶠᶜ)
end
