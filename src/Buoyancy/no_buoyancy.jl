using Oceananigans: AbstractGrid

required_tracers(::Nothing) = ()

@inline buoyancy_perturbation(i, j, k, grid::AbstractGrid{FT}, ::Nothing, C) where FT = zero(FT)

@inline ∂x_b(i, j, k, grid::AbstractGrid{FT}, ::Nothing, C) where FT = zero(FT)
@inline ∂y_b(i, j, k, grid::AbstractGrid{FT}, ::Nothing, C) where FT = zero(FT)
@inline ∂z_b(i, j, k, grid::AbstractGrid{FT}, ::Nothing, C) where FT = zero(FT)
