""" Vertical circulation associated with horizontal velocities u, v. """
@inline Γᶠᶠᶜ(i, j, k, grid, u, v) = δxᶠᶠᶜ(i, j, k, grid, Δy_qᶜᶠᶜ, v) - δyᶠᶠᶜ(i, j, k, grid, Δx_qᶠᶜᶜ, u)

"""
$(TYPEDSIGNATURES)

The vertical vorticity associated with horizontal velocities ``u`` and ``v``.
"""
@inline ζ₃ᶠᶠᶜ(i, j, k, grid, u, v) = Γᶠᶠᶜ(i, j, k, grid, u, v) * Az⁻¹ᶠᶠᶜ(i, j, k, grid)
