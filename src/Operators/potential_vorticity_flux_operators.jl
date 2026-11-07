#####
##### Energy-conserving potential vorticity flux for grids whose Δz varies horizontally
#####

@inline active_Ax_qᶠᶜᶜ(i, j, k, grid, u) = ifelse(immersed_peripheral_node(i, j, k, grid, Face(), Center(), Center()), zero(grid), Ax_qᶠᶜᶜ(i, j, k, grid, u))
@inline active_Ay_qᶜᶠᶜ(i, j, k, grid, v) = ifelse(immersed_peripheral_node(i, j, k, grid, Center(), Face(), Center()), zero(grid), Ay_qᶜᶠᶜ(i, j, k, grid, v))

# Mean thickness of the active velocity faces meeting at the corner (i, j): u-faces at (i, j) and (i, j-1),
# v-faces at (i, j) and (i-1, j). Zero if none is active, in which case all four transports vanish too.
@inline function corner_thicknessᶠᶠᶜ(i, j, k, grid)
    a₁ = !peripheral_node(i, j,   k, grid, Face(), Center(), Center())
    a₂ = !peripheral_node(i, j-1, k, grid, Face(), Center(), Center())
    a₃ = !peripheral_node(i,   j, k, grid, Center(), Face(), Center())
    a₄ = !peripheral_node(i-1, j, k, grid, Center(), Face(), Center())

    Σh = a₁ * Δzᶠᶜᶜ(i, j,   k, grid) + a₂ * Δzᶠᶜᶜ(i, j-1, k, grid) +
         a₃ * Δzᶜᶠᶜ(i,   j, k, grid) + a₄ * Δzᶜᶠᶜ(i-1, j, k, grid)

    return Σh / max(1, a₁ + a₂ + a₃ + a₄)
end

# Potential vorticity q = ζ / h at the corner, where ζ(i, j, k, grid, args...) is a vorticity (relative or planetary)
@inline function corner_potential_vorticityᶠᶠᶜ(i, j, k, grid, ζ, args...)
    h = corner_thicknessᶠᶠᶜ(i, j, k, grid)
    return ifelse(h == 0, zero(grid), ζ(i, j, k, grid, args...) / ifelse(h == 0, one(h), h))
end

@inline q_ℑx_Ay_vᶠᶠᶜ(i, j, k, grid, v, ζ, args...) = corner_potential_vorticityᶠᶠᶜ(i, j, k, grid, ζ, args...) * ℑxᶠᵃᵃ(i, j, k, grid, active_Ay_qᶜᶠᶜ, v)
@inline q_ℑy_Ax_uᶠᶠᶜ(i, j, k, grid, u, ζ, args...) = corner_potential_vorticityᶠᶠᶜ(i, j, k, grid, ζ, args...) * ℑyᵃᶠᵃ(i, j, k, grid, active_Ax_qᶠᶜᶜ, u)

# Sadourny (1975) energy-conserving flux of q = ζ / h for the transports 𝒰 = Aˣ u and 𝒱 = Aʸ v:
# the work Σ Δx Aˣ u ℑʸ(q ℑˣ𝒱) / Δx - Σ Δy Aʸ v ℑˣ(q ℑʸ𝒰) / Δy cancels corner by corner for any Δz.
@inline potential_vorticity_flux_xᶠᶜᶜ(i, j, k, grid, v, ζ, args...) = ℑyᵃᶜᵃ(i, j, k, grid, q_ℑx_Ay_vᶠᶠᶜ, v, ζ, args...) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)

@inline potential_vorticity_flux_yᶜᶠᶜ(i, j, k, grid, u, ζ, args...) = ℑxᶜᵃᵃ(i, j, k, grid, q_ℑy_Ax_uᶠᶠᶜ, u, ζ, args...) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
