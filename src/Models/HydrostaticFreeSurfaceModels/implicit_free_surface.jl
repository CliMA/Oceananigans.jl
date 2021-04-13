using Oceananigans.Grids: AbstractGrid
using Oceananigans.Architectures: device
using Oceananigans.Operators: ∂xᶠᵃᵃ, ∂yᵃᶠᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
using Oceananigans.Solvers: solve!
using Oceananigans.Fields

using Adapt

struct ImplicitFreeSurface{E, G, B, V, R, I, S}
    η :: E
    gravitational_acceleration :: G
    barotropic_volume_flux :: B
    vertically_integrated_lateral_face_areas :: V
    implicit_step_right_hand_side :: R
    implicit_step_solver :: I
    solver_settings :: S
end

# User interface to ImplicitFreeSurface
ImplicitFreeSurface(; gravitational_acceleration=g_Earth, solver_settings...) =
    ImplicitFreeSurface(nothing, gravitational_acceleration, nothing, nothing, nothing, nothing, solver_settings)

# Internal function for HydrostaticFreeSurfaceModel
function FreeSurface(free_surface::ImplicitFreeSurface{Nothing}, velocities, arch, grid)
    η = FreeSurfaceDisplacementField(velocities, arch, grid)
    g = convert(eltype(grid), free_surface.gravitational_acceleration)

    # Initialize barotropic volume fluxes
    barotropic_x_volume_flux = ReducedField(Face, Center, Nothing, arch, grid; dims=3)
    barotropic_y_volume_flux = ReducedField(Center, Face, Nothing, arch, grid; dims=3)
    barotropic_volume_flux = (u=barotropic_x_volume_flux, v=barotropic_y_volume_flux)

    # Initialize vertically integrated lateral face areas
    ∫ᶻ_Axᶠᶜᶜ = ReducedField(Face, Center, Nothing, arch, grid; dims=3)
    ∫ᶻ_Ayᶜᶠᶜ = ReducedField(Center, Face, Nothing, arch, grid; dims=3)

    vertically_integrated_lateral_face_areas = (xᶠᶜᶜ = ∫ᶻ_Axᶠᶜᶜ, yᶜᶠᶜ = ∫ᶻ_Ayᶜᶠᶜ)

    compute_vertically_integrated_lateral_face_areas!(vertically_integrated_lateral_face_areas, grid, arch)

    implicit_step_solver = PreconditionedConjugateGradientSolver(implicit_free_surface_linear_operation!,
                                                                 template_field = η,
                                                                 maximum_iterations = grid.Nx * grid.Ny,
                                                                 free_surface.solver_settings...)
    
    implicit_step_right_hand_side = ReducedField(Center, Center, Nothing, arch, grid; dims=3)

    return ImplicitFreeSurface(η,
                               g,
                               barotropic_volume_flux,
                               vertically_integrated_lateral_face_areas,
                               implicit_step_right_hand_side,
                               implicit_step_solver,
                               free_surface.solver_settings)
end

@inline explicit_barotropic_pressure_x_gradient(i, j, k, grid, ::ImplicitFreeSurface) = 0
@inline explicit_barotropic_pressure_y_gradient(i, j, k, grid, ::ImplicitFreeSurface) = 0

"""
Implicitly step forward η.
"""
ab2_step_free_surface!(free_surface::ImplicitFreeSurface, velocities_update, model, Δt, χ) =
    implicit_free_surface_step!(free_surface::ImplicitFreeSurface, velocities_update, model, Δt, χ)

function implicit_free_surface_step!(free_surface::ImplicitFreeSurface, velocities_update, model, Δt, χ)

    η = free_surface.η
    g = free_surface.gravitational_acceleration
    rhs = free_surface.implicit_step_right_hand_side
    ∫ᶻ_Q = free_surface.barotropic_volume_flux
    ∫ᶻ_A = free_surface.vertically_integrated_lateral_face_areas

    #=
    # Take an explicit step first to produce an improved initial guess for η for the iterative solver.
    event = explicit_ab2_step_free_surface!(free_surface, nothing, model, Δt, χ)
    wait(device(model.architecture), event)
    =#

    fill_halo_regions!(η, model.architecture)

    # Compute the vertically integrated volume flux
    compute_vertically_integrated_volume_flux!(∫ᶻ_Q, model, velocities_update)

    ## Compute volume scaled divergence of the barotropic transport and put into solver RHS
    compute_implicit_free_surface_right_hand_side!(rhs, model, g, Δt, ∫ᶻ_Q)

    # Subtract Azᵃᵃᵃ(i, j, 1, grid) * η[i, j, 1] / (g * Δt^2)
    event = add_previous_free_surface_contribution(free_surface, model, Δt)
    wait(device(model.architecture), event)

    fill_halo_regions!(rhs, model.architecture)

    # solve!(x, solver, b, args...) solves A*x = b for x.
    solve!(η, free_surface.implicit_step_solver, rhs, ∫ᶻ_A.xᶠᶜᶜ, ∫ᶻ_A.yᶜᶠᶜ, g, Δt)

    return nothing
end

function compute_implicit_free_surface_right_hand_side!(rhs, model, g, Δt, ∫ᶻ_U)

    event = launch!(model.architecture,
                    model.grid,
                    :xy,
                    implicit_free_surface_right_hand_side!,
                    rhs,
                    model.grid,
                    g,
                    Δt,
                    ∫ᶻ_U,
                    dependencies=Event(device(model.architecture)))


    wait(device(model.architecture), event)

    fill_halo_regions!(rhs, model.architecture)

    return nothing
end

""" Compute the divergence of fluxes Qu and Qv. """
@inline flux_div_xyᶜᶜᵃ(i, j, k, grid, Qu, Qv) = δxᶜᵃᵃ(i, j, k, grid, Qu) + δyᵃᶜᵃ(i, j, k, grid, Qv)

@kernel function implicit_free_surface_right_hand_side!(rhs, grid, g, Δt, ∫ᶻ_Q)
    i, j = @index(Global, NTuple)
    @inbounds rhs[i, j, 1] = flux_div_xyᶜᶜᵃ(i, j, 1, grid, ∫ᶻ_Q.u, ∫ᶻ_Q.v) / (g * Δt^2)
end

#=
@kernel function _compute_integrated_volume_flux_divergence!(divergence, grid, ∫ᶻ_U)
    # Here we use a integral form that has been multiplied through by volumes to be 
    # consistent with the symmetric "A" matrix.
    # The quantities differenced here are transports i.e. normal velocity vectors
    # integrated over an area.
    #
    i, j = @index(Global, NTuple)
    @inbounds divergence[i, j, 1] = δxᶜᵃᵃ(i, j, 1, grid, ∫ᶻ_Q.u) + δyᵃᶜᵃ(i, j, 1, grid, ∫ᶻ_Q.v)
end
=#

function add_previous_free_surface_contribution(free_surface, model, Δt)
   g = model.free_surface.gravitational_acceleration
   event = launch!(model.architecture,
                   model.grid,
                   :xy,
                   _add_previous_free_surface_contribution!,
                   free_surface.implicit_step_right_hand_side,
                   model.grid,
                   g,
                   Δt,
                   free_surface.η,
                   dependencies=Event(device(model.architecture)))
    return event
end

@kernel function _add_previous_free_surface_contribution!(RHS, grid, g, Δt, η)
    i, j = @index(Global, NTuple)
    @inbounds RHS[i, j, 1] -= Azᶜᶜᵃ(i, j, 1, grid) * η[i, j, 1] / (g * Δt^2)
end
