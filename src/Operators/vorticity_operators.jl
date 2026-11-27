""" Vertical circulation associated with horizontal velocities u, v. """
@inline Γᶠᶠᶜ(i, j, k, grid, u, v) = δxᶠᶠᶜ(i, j, k, grid, Δy_qᶜᶠᶜ, v) - δyᶠᶠᶜ(i, j, k, grid, Δx_qᶠᶜᶜ, u)

"""
$(TYPEDSIGNATURES)

The vertical vorticity associated with horizontal velocities ``u`` and ``v``.
"""
@inline ζ₃ᶠᶠᶜ(i, j, k, grid, u, v) = Γᶠᶠᶜ(i, j, k, grid, u, v) * Az⁻¹ᶠᶠᶜ(i, j, k, grid)

""" The vertical vorticity, set to zero on peripheral nodes. """
@inline function free_slip_vorticityᶠᶠᶜ(i, j, k, grid, u, v)
    ζ = Az⁻¹ᶠᶠᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, Δy_qᶜᶠᶜ, v) - δyᵃᶠᵃ(i, j, k, grid, Δx_qᶠᶜᶜ, u))
    return ifelse(peripheral_node(i, j, k, grid, f, f, c), zero(grid), ζ)
end
