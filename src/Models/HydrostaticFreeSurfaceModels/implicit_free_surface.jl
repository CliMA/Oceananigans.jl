using Oceananigans.Grids: AbstractGrid
using Oceananigans.Architectures: device
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, Δzᶜᶜᶠ, Δzᶜᶜᶜ
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
using Oceananigans.Solvers: solve!
using Oceananigans.Fields
using Oceananigans.Utils: prettytime

using Adapt
using KernelAbstractions: NoneEvent

struct ImplicitFreeSurface{E, G, B, I, M, S}
    η :: E
    gravitational_acceleration :: G
    barotropic_volume_flux :: B
    implicit_step_solver :: I
    solver_method :: M
    solver_settings :: S
end

"""
    ImplicitFreeSurface(; solver_method=:Default, gravitational_acceleration=g_Earth, solver_settings...)

The implicit free-surface equation is

```math
\\left [ 𝛁_h ⋅ (H 𝛁_h) - \\frac{1}{g Δt^2} \\right ] η^{n+1} = \\frac{𝛁_h ⋅ 𝐐_⋆}{g Δt} - \\frac{η^{n}}{g Δt^2} ,
```

where ``η^n`` is the free-surface elevation at the ``n``-th time step, ``H`` is depth, ``g`` is
the gravitational acceleration, ``Δt`` is the time step, ``𝐐_⋆`` is the barotropic volume flux
associated with the predictor velocity field, and ``𝛁_h`` is the horizontal gradient operator.

This equation can be solved in general using the [`PreconditionedConjugateGradientSolver`](@ref).

In the case that ``H`` is constant, we divide through to obtain

```math
\\left ( ∇^2_h - \\frac{1}{g H Δt^2} \\right ) η^{n+1}  = \\frac{1}{g H Δt} \\left ( 𝛁_h ⋅ 𝐐_⋆ - \\frac{η^{n}}{Δt} \\right ) .
```

Thus, for constant ``H`` and on grids with regular spacing in ``x`` and ``y`` directions, the free
surface can be obtained using the `FFTImplicitFreeSurfaceSolver`.
"""
ImplicitFreeSurface(; solver_method=:Default, gravitational_acceleration=g_Earth, solver_settings...) =
    ImplicitFreeSurface(nothing, gravitational_acceleration, nothing, nothing, solver_method, solver_settings)

Adapt.adapt_structure(to, free_surface::ImplicitFreeSurface) =
    ImplicitFreeSurface(Adapt.adapt(to, free_surface.η), free_surface.gravitational_acceleration,
                        nothing, nothing, nothing, nothing)

# Internal function for HydrostaticFreeSurfaceModel
function FreeSurface(free_surface::ImplicitFreeSurface{Nothing}, velocities, grid)
    η = FreeSurfaceDisplacementField(velocities, free_surface, grid)
    gravitational_acceleration = convert(eltype(grid), free_surface.gravitational_acceleration)

    # Initialize barotropic volume fluxes
    barotropic_x_volume_flux = Field{Face, Center, Nothing}(grid)
    barotropic_y_volume_flux = Field{Center, Face, Nothing}(grid)
    barotropic_volume_flux = (u=barotropic_x_volume_flux, v=barotropic_y_volume_flux)

    solver_method = free_surface.solver_method   # could be = :Default

    solver = build_implicit_step_solver(Val(solver_method), grid, gravitational_acceleration, free_surface.solver_settings)
    
    actual_solver_method = typeof(solver).name.name

    actual_solver_method = actual_solver_method == :PCGImplicitFreeSurfaceSolver ? :PreconditionedConjugateGradientImplicitFreeSurfaceSolver : actual_solver_method

    return ImplicitFreeSurface(η, gravitational_acceleration,
                               barotropic_volume_flux,
                               solver,
                               actual_solver_method,
                               free_surface.solver_settings)
end

is_horizontally_regular(grid) = false
is_horizontally_regular(::RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:Number, <:Number}) = true

function build_implicit_step_solver(::Val{:Default}, grid, gravitational_acceleration, settings)
    default_method = is_horizontally_regular(grid) ? :FastFourierTransform : :PreconditionedConjugateGradient
    return build_implicit_step_solver(Val(default_method), grid, gravitational_acceleration, settings)
end

@inline explicit_barotropic_pressure_x_gradient(i, j, k, grid, ::ImplicitFreeSurface) = 0
@inline explicit_barotropic_pressure_y_gradient(i, j, k, grid, ::ImplicitFreeSurface) = 0

"""
Implicitly step forward η.
"""
ab2_step_free_surface!(free_surface::ImplicitFreeSurface, model, Δt, χ, velocities_update) =
    implicit_free_surface_step!(free_surface::ImplicitFreeSurface, model, Δt, χ, velocities_update)

function implicit_free_surface_step!(free_surface::ImplicitFreeSurface, model, Δt, χ, velocities_update)
    η = free_surface.η
    g = free_surface.gravitational_acceleration
    rhs = free_surface.implicit_step_solver.right_hand_side
    ∫ᶻQ = free_surface.barotropic_volume_flux
    solver = free_surface.implicit_step_solver
    arch = model.architecture

    # Wait for predictor velocity update step to complete.
    wait(device(arch), velocities_update)

    masking_events = Tuple(mask_immersed_field!(q) for q in model.velocities)
    wait(device(model.architecture), MultiEvent(masking_events))

    # Compute barotropic volume flux. Blocking.
    compute_vertically_integrated_volume_flux!(∫ᶻQ, model)

    # Compute right hand side of implicit free surface equation
    rhs_event = compute_implicit_free_surface_right_hand_side!(rhs, solver, g, Δt, ∫ᶻQ, η)
    wait(device(arch), rhs_event)

    # Solve for the free surface at tⁿ⁺¹
    start_time = time_ns()

    solve!(η, solver, rhs, g, Δt)

    @debug "Implicit step solve took $(prettytime((time_ns() - start_time) * 1e-9))."

    fill_halo_regions!(η, arch)
    
    return NoneEvent()
end

