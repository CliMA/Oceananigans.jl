using Oceananigans.Fields: Field, ZReducedField

"""
Compute the horizontal divergence of vertically-uniform quantity using
vertically-integrated face areas `∫ᶻ_Axᶠᶜᶜ` and `∫ᶻ_Ayᶜᶠᶜ`.
"""
@inline Az_∇h²ᶜᶜᶜ(i, j, k, grid, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, η) =
    (δxᶜᵃᵃ(i, j, k, grid, ∫ᶻ_Ax_∂x_ηᶠᶜᶜ, ∫ᶻ_Axᶠᶜᶜ, η) +
     δyᵃᶜᵃ(i, j, k, grid, ∫ᶻ_Ay_∂y_ηᶜᶠᶜ, ∫ᶻ_Ayᶜᶠᶜ, η))
