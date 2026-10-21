
@inline x_dot_g_bᶠᶜᶜ(i, j, k, grid, buoyancy, C) = ĝ_x(buoyancy) * ℑxᶠᶜᶜ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy.formulation, C)
@inline y_dot_g_bᶜᶠᶜ(i, j, k, grid, buoyancy, C) = ĝ_y(buoyancy) * ℑyᶜᶠᶜ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy.formulation, C)
@inline z_dot_g_bᶜᶜᶠ(i, j, k, grid, buoyancy, C) = ĝ_z(buoyancy) * ℑzᶜᶜᶠ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy.formulation, C)

const NZBF = BuoyancyForce{<:Any, NegativeZDirection}

@inline x_dot_g_bᶠᶜᶜ(i, j, k, grid, bf::NZBF, C) = 0
@inline y_dot_g_bᶜᶠᶜ(i, j, k, grid, bf::NZBF, C) = 0
@inline z_dot_g_bᶜᶜᶠ(i, j, k, grid, bf::NZBF, C) = ℑzᶜᶜᶠ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, bf.formulation, C)
