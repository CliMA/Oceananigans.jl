validate_buoyancy(::Nothing, tracers) = nothing

required_tracers(::Nothing) = ()

@inline buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, ::Nothing, C) = zero(grid)

@inline ∂x_b(i, j, k, grid, ::Nothing, C) = zero(grid)
@inline ∂y_b(i, j, k, grid, ::Nothing, C) = zero(grid)
@inline ∂z_b(i, j, k, grid, ::Nothing, C) = zero(grid)

@inline x_dot_g_bᶠᶜᶜ(i, j, k, grid, ::Nothing, C) = zero(grid)
@inline y_dot_g_bᶜᶠᶜ(i, j, k, grid, ::Nothing, C) = zero(grid)
@inline z_dot_g_bᶜᶜᶠ(i, j, k, grid, ::Nothing, C) = zero(grid)
