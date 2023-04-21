@inline x_dot_g_b(i, j, k, grid, buoyancy, C) = ĝ_x(buoyancy) * ℑxᶠᵃᵃ(i, j, k, grid, buoyancy_perturbation, buoyancy.model, C)
@inline y_dot_g_b(i, j, k, grid, buoyancy, C) = ĝ_y(buoyancy) * ℑyᵃᶠᵃ(i, j, k, grid, buoyancy_perturbation, buoyancy.model, C)
@inline z_dot_g_b(i, j, k, grid, buoyancy, C) = ĝ_z(buoyancy) * ℑzᵃᵃᶠ(i, j, k, grid, buoyancy_perturbation, buoyancy.model, C)

@inline x_dot_g_b(i, j, k, grid,  ::Buoyancy{M, NegativeZDirection}, C) where M = 0
@inline y_dot_g_b(i, j, k, grid,  ::Buoyancy{M, NegativeZDirection}, C) where M = 0
@inline z_dot_g_b(i, j, k, grid, b::Buoyancy{M, NegativeZDirection}, C) where M = ℑzᵃᵃᶠ(i, j, k, grid, buoyancy_perturbation, b.model, C)

