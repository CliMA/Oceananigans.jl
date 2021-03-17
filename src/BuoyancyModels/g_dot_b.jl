@inline x_dot_g_b(i, j, k, grid, buoyancy, C) = ĝ_x(buoyancy) * buoyancy_perturbation(i, j, k, grid, buoyancy.model, C)
@inline y_dot_g_b(i, j, k, grid, buoyancy, C) = ĝ_y(buoyancy) * buoyancy_perturbation(i, j, k, grid, buoyancy.model, C)
@inline z_dot_g_b(i, j, k, grid, buoyancy, C) = ĝ_z(buoyancy) * buoyancy_perturbation(i, j, k, grid, buoyancy.model, C)
