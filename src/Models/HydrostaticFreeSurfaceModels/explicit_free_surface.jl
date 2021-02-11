using Oceananigans.Grids: AbstractGrid
using Oceananigans.Operators: ∂xᶠᵃᵃ, ∂yᵃᶠᵃ
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions

using Adapt

struct ExplicitFreeSurface{E, T}
    η :: E
    gravitational_acceleration :: T
end

ExplicitFreeSurface(; gravitational_acceleration=g_Earth) =
    ExplicitFreeSurface(nothing, gravitational_acceleration)

Adapt.adapt_structure(to, free_surface::ExplicitFreeSurface) =
    ExplicitFreeSurface(Adapt.adapt(to, free_surface.η), gravitational_acceleration)

function FreeSurface(free_surface::ExplicitFreeSurface{Nothing}, arch, grid)
    η = CenterField(arch, grid, TracerBoundaryConditions(grid))
    g = convert(eltype(grid), free_surface.gravitational_acceleration)
    return ExplicitFreeSurface(η, g)
end

explicit_barotropic_pressure_x_gradient(i, j, k, grid, free_surface::ExplicitFreeSurface) =
    free_surface.gravitational_acceleration * ∂xᶠᵃᵃ(i, j, k, grid, free_surface.η)

explicit_barotropic_pressure_y_gradient(i, j, k, grid, free_surface::ExplicitFreeSurface) =
    free_surface.gravitational_acceleration * ∂yᵃᶠᵃ(i, j, k, grid, free_surface.η)
