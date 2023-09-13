@inline x_dot_g_bᶠᶜᶜ(i, j, k, grid, buoyancy, C) = ĝ_x(buoyancy) * ℑxᶠᵃᵃ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy.model, C)
@inline y_dot_g_bᶜᶠᶜ(i, j, k, grid, buoyancy, C) = ĝ_y(buoyancy) * ℑyᵃᶠᵃ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy.model, C)
@inline z_dot_g_bᶜᶜᶠ(i, j, k, grid, buoyancy, C) = ĝ_z(buoyancy) * ℑzᵃᵃᶠ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy.model, C)

@inline x_dot_g_bᶠᶜᶜ(i, j, k, grid,  ::Buoyancy{M, NegativeZDirection}, C) where M = 0
@inline y_dot_g_bᶜᶠᶜ(i, j, k, grid,  ::Buoyancy{M, NegativeZDirection}, C) where M = 0
@inline z_dot_g_bᶜᶜᶠ(i, j, k, grid, b::Buoyancy{M, NegativeZDirection}, C) where M = ℑzᵃᵃᶠ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, b.model, C)