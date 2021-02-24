using Oceananigans.Grids: AbstractGrid
using Oceananigans.Operators: ∂xᶠᵃᵃ, ∂yᵃᶠᵃ
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
using Oceananigans.Fields

using Adapt

struct ImplicitFreeSurface{E, G, B, I}
    η :: E
    gravitational_acceleration :: G
    barotropic_transport :: B
    implicit_step_solver :: I
end

# User interface to ImplicitFreeSurface
ImplicitFreeSurface(; gravitational_acceleration=g_Earth) =
    ImplicitFreeSurface(nothing, gravitational_acceleration, nothing, nothing)

### ExplicitFreeSurface(; gravitational_acceleration=g_Earth) =
###     ExplicitFreeSurface(nothing, gravitational_acceleration)

### Adapt.adapt_structure(to, free_surface::ExplicitFreeSurface) =
###     ExplicitFreeSurface(Adapt.adapt(to, free_surface.η), free_surface.gravitational_acceleration)

# Internal function for HydrostaticFreeSurfaceModel
function FreeSurface(free_surface::ImplicitFreeSurface{Nothing}, arch, grid)
    η = CenterField(arch, grid, TracerBoundaryConditions(grid))
    g = convert(eltype(grid), free_surface.gravitational_acceleration)

    implicit_step_solver = ImplicitFreeSurfaceSolver(arch, η)

    barotropic_u_transport = ReducedField(Face, Center, Nothing, arch, grid; dims=(1, 2), boundary_conditions=nothing)
    barotropic_v_transport = ReducedField(Center, Face, Nothing, arch, grid; dims=(1, 2), boundary_conditions=nothing)
    barotropic_transport = (u=barotropic_u_transport, v=barotropic_v_transport)

    return ImplicitFreeSurface(η, g, barotropic_transport, implicit_step_solver)
end

explicit_barotropic_pressure_x_gradient(i, j, k, grid, free_surface::ImplicitFreeSurface) =
    free_surface.gravitational_acceleration * ∂xᶠᵃᵃ(i, j, k, grid, free_surface.η)

explicit_barotropic_pressure_y_gradient(i, j, k, grid, free_surface::ImplicitFreeSurface) =
    free_surface.gravitational_acceleration * ∂yᵃᶠᵃ(i, j, k, grid, free_surface.η)

