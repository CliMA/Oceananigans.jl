using Oceananigans.Grids: AbstractGrid, XYRegularRG, static_column_depthᶜᶜᵃ
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
using Oceananigans.Solvers: solve!
using Oceananigans.Utils: prettytime, prettysummary
using Oceananigans.ImmersedBoundaries: MutableGridOfSomeKind

import Oceananigans: prognostic_state, restore_prognostic_state!
import Oceananigans.DistributedComputations: synchronize_communication!

using Adapt: Adapt

struct ImplicitFreeSurface{E, G, I, M, S} <: AbstractFreeSurface{E, G}
    displacement :: E
    gravitational_acceleration :: G
    implicit_step_solver :: I
    solver_method :: M
    solver_settings :: S
end

Base.show(io::IO, fs::ImplicitFreeSurface) =
    isnothing(fs.displacement) ?
    print(io, "ImplicitFreeSurface with ", fs.solver_method, "\n",
              "├─ gravitational_acceleration: ", prettysummary(fs.gravitational_acceleration), "\n",
              "├─ solver_method: ", fs.solver_method, "\n", # TODO: implement summary for solvers
              "└─ settings: ", isempty(fs.solver_settings) ? "Default" : fs.solver_settings) :
    print(io, "ImplicitFreeSurface with ", fs.solver_method, "\n",
              "├─ grid: ", summary(fs.displacement.grid), "\n",
              "├─ displacement: ", summary(fs.displacement), "\n",
              "├─ gravitational_acceleration: ", prettysummary(fs.gravitational_acceleration), "\n",
              "├─ implicit_step_solver: ", nameof(typeof(fs.implicit_step_solver)), "\n", # TODO: implement summary for solvers
              "└─ settings: ", fs.solver_settings)

"""
    ImplicitFreeSurface(; solver_method = :Default,
                        gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
                        solver_settings...)

Return an implicit free-surface solver. The implicit free-surface equation is

```math
\\left [ 𝛁_h ⋅ (H 𝛁_h) - \\frac{1}{g Δt^2} \\right ] η^{n+1} = \\frac{𝛁_h ⋅ 𝐐_⋆}{g Δt} - \\frac{η^{n}}{g Δt^2} ,
```

where ``η^n`` is the free-surface elevation at the ``n``-th time step, ``H`` is depth, ``g`` is
the gravitational acceleration, ``Δt`` is the time step, ``𝛁_h`` is the horizontal gradient operator,
and ``𝐐_⋆`` is the barotropic volume flux associated with the predictor velocity field ``𝐮_⋆``, i.e.,

```math
𝐐_⋆ = \\int_{-H}^0 𝐮_⋆ \\, 𝖽 z ,
```

where

```math
𝐮_⋆ = 𝐮^n + \\int_{t_n}^{t_{n+1}} 𝐆ᵤ \\, 𝖽t .
```

This equation can be solved, in general, using the [`ConjugateGradientSolver`](@ref) but
other solvers can be invoked in special cases.

If ``H`` is constant, we divide through out to obtain

```math
\\left ( ∇^2_h - \\frac{1}{g H Δt^2} \\right ) η^{n+1}  = \\frac{1}{g H Δt} \\left ( 𝛁_h ⋅ 𝐐_⋆ - \\frac{η^{n}}{Δt} \\right ) .
```

Thus, for constant ``H`` and on grids with regular spacing in ``x`` and ``y`` directions, the free
surface can be obtained using the [`FFTBasedPoissonSolver`](@ref).

`solver_method` can be either of:
* `:FastFourierTransform` for [`FFTBasedPoissonSolver`](@ref)
* `:PreconditionedConjugateGradient` for [`ConjugateGradientSolver`](@ref)

By default, if the grid has regular spacing in the horizontal directions then the `:FastFourierTransform` is chosen,
otherwise the `:PreconditionedConjugateGradient`.
"""
function ImplicitFreeSurface(;
    solver_method = :Default,
    gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
    solver_settings...)

    return ImplicitFreeSurface(nothing, gravitational_acceleration, nothing, solver_method, solver_settings)
end

Adapt.adapt_structure(to, free_surface::ImplicitFreeSurface) =
    ImplicitFreeSurface(Adapt.adapt(to, free_surface.displacement), free_surface.gravitational_acceleration,
                        nothing, nothing, nothing)

on_architecture(to, free_surface::ImplicitFreeSurface) =
    ImplicitFreeSurface(on_architecture(to, free_surface.displacement),
                        on_architecture(to, free_surface.gravitational_acceleration),
                        on_architecture(to, free_surface.implicit_step_solver),
                        on_architecture(to, free_surface.solver_methods),
                        on_architecture(to, free_surface.solver_settings))

# Internal function for HydrostaticFreeSurfaceModel
function materialize_free_surface(free_surface::ImplicitFreeSurface{Nothing}, velocities, grid)
    η = free_surface_displacement_field(velocities, free_surface, grid)
    gravitational_acceleration = convert(eltype(grid), free_surface.gravitational_acceleration)

    user_solver_method = free_surface.solver_method # could be = :Default
    solver = build_implicit_step_solver(Val(user_solver_method), grid, free_surface.solver_settings, gravitational_acceleration)
    solver_method = nameof(typeof(solver))

    return ImplicitFreeSurface(η,
                               gravitational_acceleration,
                               solver,
                               solver_method,
                               free_surface.solver_settings)
end

build_implicit_step_solver(::Val{:Default}, grid::XYRegularStaticRG, settings, gravitational_acceleration) =
    build_implicit_step_solver(Val(:FastFourierTransform), grid, settings, gravitational_acceleration)

build_implicit_step_solver(::Val{:Default}, grid, settings, gravitational_acceleration) =
    build_implicit_step_solver(Val(:PreconditionedConjugateGradient), grid, settings, gravitational_acceleration)

@inline explicit_barotropic_pressure_x_gradient(i, j, k, grid, ::ImplicitFreeSurface) = 0
@inline explicit_barotropic_pressure_y_gradient(i, j, k, grid, ::ImplicitFreeSurface) = 0

# No variables are asynchronously computed
synchronize_communication!(::ImplicitFreeSurface) = nothing

"""
Implicitly step forward η.
"""
function step_free_surface!(free_surface::ImplicitFreeSurface, model, timestepper, Δt)
    η       = free_surface.displacement
    g       = free_surface.gravitational_acceleration
    rhs     = free_surface.implicit_step_solver.right_hand_side
    solver  = free_surface.implicit_step_solver
    u, v, _ = model.velocities

    @apply_regionally begin
        mask_immersed_field!(u)
        mask_immersed_field!(v)
    end

    fill_halo_regions!((u, v), model.clock, fields(model))
    compute_implicit_free_surface_right_hand_side!(rhs, solver, g, Δt, model.velocities, η)

    # Solve for the free surface at tⁿ⁺¹
    start_time = time_ns()

    solve!(η, solver, rhs, g, Δt)

    @debug "Implicit step solve took $(prettytime((time_ns() - start_time) * 1e-9))."

    fill_halo_regions!(η)

    return nothing
end

function step_free_surface!(free_surface::ImplicitFreeSurface, model, timestepper::SplitRungeKuttaTimeStepper, Δt)
    parent(free_surface.displacement) .= parent(timestepper.Ψ⁻.η)
    step_free_surface!(free_surface, model, nothing, Δt)
    return nothing
end

#####
##### Compute transport velocities for RK discretization
#####

# Compute transport velocities for tracer advection
function compute_transport_velocities!(model, free_surface::ImplicitFreeSurface)
    grid = model.grid
    u, v, w = model.velocities
    ũ, ṽ, w̃ = model.transport_velocities

    launch!(architecture(grid), grid, volume_kernel_parameters(grid), _compute_implicit_transport_velocities!, ũ, ṽ, grid, u, v)
    update_vertical_velocities!(model.transport_velocities, model.grid, model)

    return nothing
end

@kernel function _compute_implicit_transport_velocities!(ũ, ṽ, grid, u, v)
    i, j = @index(Global, NTuple)
    Nz   = size(grid, 3)
    Hᶠᶜ  = column_depthᶠᶜᵃ(i, j, grid)
    Hᶜᶠ  = column_depthᶜᶠᵃ(i, j, grid)

    # Barotropic velocities
    Ũᵐ⁺¹ = barotropic_U(i, j, Nz, grid, u)
    Ṽᵐ⁺¹ = barotropic_V(i, j, Nz, grid, v)
    Ũ    = barotropic_U(i, j, Nz, grid, ũ)
    Ṽ    = barotropic_V(i, j, Nz, grid, ṽ)

    δuᵢ = ifelse(Hᶠᶜ == 0, zero(grid), (Ũᵐ⁺¹ - Ũ) / Hᶠᶜ)
    δvⱼ = ifelse(Hᶜᶠ == 0, zero(grid), (Ṽᵐ⁺¹ - Ṽ) / Hᶜᶠ)

    @inbounds for k in 1:Nz
        immersedᶠᶜᶜ = peripheral_node(i, j, k, grid, Face(), Center(), Center())
        immersedᶜᶠᶜ = peripheral_node(i, j, k, grid, Center(), Face(), Center())

        ũ[i, j, k] = ifelse(immersedᶠᶜᶜ, zero(grid), ũ[i, j, k] + δuᵢ)
        ṽ[i, j, k] = ifelse(immersedᶜᶠᶜ, zero(grid), ṽ[i, j, k] + δvⱼ)
    end
end

#####
##### Checkpointing
#####

function prognostic_state(fs::ImplicitFreeSurface)
    return (; displacement = prognostic_state(fs.displacement))
end

function restore_prognostic_state!(restored::ImplicitFreeSurface, from)
    restore_prognostic_state!(restored.displacement, from.displacement)
    return restored
end

restore_prognostic_state!(::ImplicitFreeSurface, ::Nothing) = nothing
