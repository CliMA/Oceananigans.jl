using Oceananigans.Grids: AbstractGrid
using Oceananigans.Operators: ∂xᶠᵃᵃ, ∂yᵃᶠᵃ
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
using Oceananigans.Fields

using Adapt

struct ImplicitFreeSurface{E, G, B, VF, I}
    η :: E
    gravitational_acceleration :: G
    barotropic_transport :: B
    vertically_integrated_lateral_face_areas :: VF
    implicit_step_solver :: I
end

# User interface to ImplicitFreeSurface
ImplicitFreeSurface(; gravitational_acceleration=g_Earth) =
    ImplicitFreeSurface(nothing, gravitational_acceleration, nothing, nothing, nothing)

### ExplicitFreeSurface(; gravitational_acceleration=g_Earth) =
###     ExplicitFreeSurface(nothing, gravitational_acceleration)

### Adapt.adapt_structure(to, free_surface::ExplicitFreeSurface) =
###     ExplicitFreeSurface(Adapt.adapt(to, free_surface.η), free_surface.gravitational_acceleration)

# Internal function for HydrostaticFreeSurfaceModel
function FreeSurface(free_surface::ImplicitFreeSurface{Nothing}, arch, grid)
    η = CenterField(arch, grid, TracerBoundaryConditions(grid))
    g = convert(eltype(grid), free_surface.gravitational_acceleration)

    barotropic_u_transport = ReducedField(Face, Center, Nothing, arch, grid; dims=(3), boundary_conditions=nothing)
    barotropic_v_transport = ReducedField(Center, Face, Nothing, arch, grid; dims=(3), boundary_conditions=nothing)
    barotropic_transport = (u=barotropic_u_transport, v=barotropic_v_transport)

    Ax_zintegral = ReducedField(Face, Center, Nothing, arch, grid; dims=(3), boundary_conditions=nothing)
    Ay_zintegral = ReducedField(Face, Center, Nothing, arch, grid; dims=(3), boundary_conditions=nothing)
    vertically_integrated_lateral_face_areas = (Ax = Ax_zintegral, Ay=Ay_zintegral)
    compute_vertically_integrated_lateral_face_areas!(vertically_integrated_lateral_face_areas, grid, arch)

    implicit_step_solver = ImplicitFreeSurfaceSolver(arch, η, vertically_integrated_lateral_face_areas)

    return ImplicitFreeSurface(η, g, 
                               barotropic_transport, 
                               vertically_integrated_lateral_face_areas,
                               implicit_step_solver)
end

explicit_barotropic_pressure_x_gradient(i, j, k, grid, free_surface::ImplicitFreeSurface) =
    free_surface.gravitational_acceleration * ∂xᶠᵃᵃ(i, j, k, grid, free_surface.η)

explicit_barotropic_pressure_y_gradient(i, j, k, grid, free_surface::ImplicitFreeSurface) =
    free_surface.gravitational_acceleration * ∂yᵃᶠᵃ(i, j, k, grid, free_surface.η)

@kernel function _compute_vertically_integrated_lateral_face_areas!(grid, A )
    i, j = @index(Global, NTuple)
    # U.w[i, j, 1] = 0 is enforced via halo regions.
    A.Ax[i, j, 1] = 0.
    A.Ay[i, j, 1] = 0.
    @unroll for k in 1:grid.Nz
        #### @inbounds barotropic_transport.u[i, j, 1] += U.u[i, j, k-1]*Δyᶠᶜᵃ(i, j, k, grid)*Δzᵃᵃᶜ(i, j, k, grid)
        #### @inbounds barotropic_transport.v[i, j, 1] += U.v[i, j, k-1]*Δyᶠᶜᵃ(i, j, k, grid)*Δzᵃᵃᶜ(i, j, k, grid)
        @inbounds A.Ax[i, j, 1] += Δyᶠᶠᵃ(i, j, k, grid)*ΔzC(i, j, k, grid)
        @inbounds A.Ay[i, j, 1] += Δxᶠᶠᵃ(i, j, k, grid)*ΔzC(i, j, k, grid)
    end
end

function compute_vertically_integrated_lateral_face_areas!(vertically_integrated_lateral_face_areas, grid, arch)

    event = launch!(arch,
                    grid,
                    :xy,
                    _compute_vertically_integrated_lateral_face_areas!,
                    grid,
                    vertically_integrated_lateral_face_areas,
                    dependencies=Event(device(arch)))

    wait(device(arch), event)

    return
end
