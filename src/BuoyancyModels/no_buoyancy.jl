validate_buoyancy(::Nothing, tracers) = nothing

required_tracers(::Nothing) = ()

@inline buoyancy_perturbation(i, j, k, grid::AbstractGrid{FT}, ::Nothing, C) where FT = zero(FT)

@inline ∂x_b(i, j, k, grid::AbstractGrid{FT}, ::Nothing, C) where FT = zero(FT)
@inline ∂y_b(i, j, k, grid::AbstractGrid{FT}, ::Nothing, C) where FT = zero(FT)
@inline ∂z_b(i, j, k, grid::AbstractGrid{FT}, ::Nothing, C) where FT = zero(FT)

@inline x_dot_g_b(i, j, k, grid, ::Nothing, C) = 0
@inline y_dot_g_b(i, j, k, grid, ::Nothing, C) = 0
@inline z_dot_g_b(i, j, k, grid, ::Nothing, C) = 0
