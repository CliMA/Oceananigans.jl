using Oceananigans.Advection: AdaptiveImplicitVerticalAdvection, FluxFormAdvection, update_advection!
using Oceananigans.Architectures: ReactantState
using Oceananigans.Biogeochemistry: update_biogeochemical_state!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: XFaceField, YFaceField, immersed_boundary_condition
using Oceananigans.Grids: inactive_node, column_depthᶜᶜᵃ, static_column_depthᶜᶜᵃ
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Models: surface_kernel_parameters
using Oceananigans.Operators: σⁿ
using Oceananigans.TimeSteppers: next_time
using Oceananigans.TurbulenceClosures: build_closure_fields, implicit_step!
using Oceananigans.Utils: tupleit

import Oceananigans: prognostic_state, restore_prognostic_state!
import Oceananigans.Biogeochemistry: biogeochemical_transition, biogeochemical_drift_velocity, biogeochemical_auxiliary_fields
import Oceananigans.TimeSteppers: step_split_tracers!

"""
    TracerTimeStepSplitting(; tracers, ratio, biogeochemistry_substeps = nothing)

Configure split tracer time stepping for `HydrostaticFreeSurfaceModel`. The `tracers` (a
`Symbol` or a tuple of `Symbol`s) form a "slow" group that is advanced once every `ratio`
dynamics time steps, with a single low-storage third-order Runge-Kutta step of length
`ratio * Δt` that uses the transports accumulated over the preceding `ratio` dynamics steps.
All other tracers, the momentum and the free surface are stepped every dynamics step.

The slow tracers stay in `model.tracers`, so they remain visible to the closure, the
biogeochemistry and any forcing. Between long steps they are frozen at the value they had
at the end of the previous long step: output of slow tracers is only meaningful at iterations
that are multiples of `ratio`, for example with `schedule = IterationInterval(ratio)`.

The long step is exactly conservative. Every dynamics step (on its final Runge-Kutta stage,
when the face stretching factor of the tracer update is still available) the model accumulates
the time-integrated, stretching-weighted horizontal transports

    Fᵘ = Σ Δt ũ σᶠᶜ,    Fᵛ = Σ Δt ṽ σᶜᶠ,

together with the time-integrated closure fields (for example, diffusivities). At every stage
of the long step the slow velocities are reconstructed as `ū = Fᵘ / (T σᶠᶜ)` where `T` is the
accumulated time and `σᶠᶜ` the stretching factor of that stage, and `w̄` is computed from
continuity. Under `ZStarCoordinate` the free surface is interpolated linearly between its
values at the beginning and at the end of the long step, so the column thickness evolves
consistently with the accumulated transports and a uniform tracer stays uniform.

Vertical diffusion with a vertically implicit closure uses the time-mean closure fields.

Slow tracers may use [`FluxFormSemiLagrangian`](@ref) advection, which remains stable at the
horizontal Courant numbers larger than one that a long step reaches. Their horizontal step is taken
once per long step, over the whole accumulated time `T`, with the long-step transports `ū, v̄` and the
cell volumes at the beginning of the long step. The `maximum_courant_number` of the scheme (and the
grid halo, which must be at least `maximum_courant_number + 3`) should cover the long-step Courant number,
and the net horizontal outflow of any cell during a long step must stay below its volume (see
[`FluxFormSemiLagrangian`](@ref)); reduce `ratio` if the long-step vertical Courant number approaches one.

Keyword arguments
=================

- `tracers` (required): names of the slow tracers.
- `ratio` (required): number of dynamics time steps per long tracer step.
- `biogeochemistry_substeps`: `nothing` (default) evaluates the biogeochemical sources of the slow
  tracers within the long Runge-Kutta step. An integer `M` instead Strang-splits the sources from
  the transport: half a long step of sources, the transport step, and another half step of sources,
  each half integrated pointwise with `M` Runge-Kutta sub-steps. This requires sources that only
  read tracers at the local grid point.

Restrictions: `ZStarCoordinate` requires a `SplitRungeKuttaTimeStepper`; the slow tracers cannot
feed the closure's prognostic tracers (e.g. CATKE's `e`); `PrescribedVelocityFields` and Reactant
are not supported. Slow tracers that feed buoyancy are frozen within each long step.

Example
=======

```jldoctest
using Oceananigans

splitting = TracerTimeStepSplitting(tracers = (:N, :P), ratio = 4, biogeochemistry_substeps = 2)

# output
TracerTimeStepSplitting
├── tracers: (:N, :P)
├── ratio: 4
└── biogeochemistry_substeps: 2
```
"""
mutable struct TracerTimeStepSplitting{N, M, S, F, U, V, K, E, G, P, C, T}
    tracer_names :: N
    ratio :: Int
    biogeochemistry_substeps :: M
    slow_tracers :: S
    fast_tracers :: F
    transport :: U
    velocities :: V
    closure_fields :: K
    initial_displacement :: E
    saved_grid_state :: G
    previous_tracers :: P
    clock :: C
    accumulated_steps :: Int
    accumulated_time :: T
end

function TracerTimeStepSplitting(; tracers, ratio, biogeochemistry_substeps = nothing)
    tracer_names = tupleit(tracers)

    if !(ratio isa Integer) || ratio < 1
        throw(ArgumentError("TracerTimeStepSplitting ratio must be a positive integer, got $ratio."))
    end

    if !isnothing(biogeochemistry_substeps) && (!(biogeochemistry_substeps isa Integer) || biogeochemistry_substeps < 1)
        throw(ArgumentError("biogeochemistry_substeps must be `nothing` or a positive integer, got $biogeochemistry_substeps."))
    end

    if isempty(tracer_names) || !all(name -> name isa Symbol, tracer_names)
        throw(ArgumentError("TracerTimeStepSplitting tracers must be a non-empty tuple of Symbols, got $tracers."))
    end

    return TracerTimeStepSplitting(tracer_names, Int(ratio), biogeochemistry_substeps,
                                   nothing, nothing, nothing, nothing, nothing, nothing,
                                   nothing, nothing, nothing, 0, 0.0)
end

Base.summary(::TracerTimeStepSplitting) = "TracerTimeStepSplitting"

function Base.show(io::IO, splitting::TracerTimeStepSplitting)
    print(io, "TracerTimeStepSplitting", '\n',
              "├── tracers: ", splitting.tracer_names, '\n',
              "├── ratio: ", splitting.ratio, '\n',
              "└── biogeochemistry_substeps: ", splitting.biogeochemistry_substeps)
end

#####
##### Construction
#####

materialize_tracer_time_step_splitting(::Nothing, args...) = nothing

function materialize_tracer_time_step_splitting(splitting::TracerTimeStepSplitting, grid, clock, tracers, velocities,
                                                closure, timestepper, vertical_coordinate, biogeochemistry,
                                                boundary_conditions)

    slow_names = splitting.tracer_names
    validate_tracer_time_step_splitting(slow_names, grid, tracers, velocities, closure, timestepper,
                                        vertical_coordinate, biogeochemistry, splitting.biogeochemistry_substeps)

    fast_names = Tuple(name for name in keys(tracers) if name ∉ slow_names)
    slow_tracers = NamedTuple{slow_names}(Tuple(tracers[name] for name in slow_names))
    fast_tracers = NamedTuple{fast_names}(Tuple(tracers[name] for name in fast_names))

    transport = (u = XFaceField(grid), v = YFaceField(grid))
    slow_velocities = transport_velocity_fields(velocities)
    time_integrated_closure_fields = build_closure_fields(nothing, grid, clock, keys(tracers), boundary_conditions, closure)

    initial_displacement, saved_grid_state = split_tracer_grid_storage(vertical_coordinate, grid)
    previous_tracers = previous_slow_tracer_fields(timestepper, slow_tracers)

    return TracerTimeStepSplitting(slow_names, splitting.ratio, splitting.biogeochemistry_substeps,
                                   slow_tracers, fast_tracers, transport, slow_velocities,
                                   time_integrated_closure_fields, initial_displacement, saved_grid_state,
                                   previous_tracers, deepcopy(clock), 0, zero(Float64))
end

function validate_tracer_time_step_splitting(slow_names, grid, tracers, velocities, closure, timestepper,
                                             vertical_coordinate, biogeochemistry, biogeochemistry_substeps)

    if architecture(grid) isa ReactantState
        throw(ArgumentError("TracerTimeStepSplitting is not supported with Reactant."))
    end

    if velocities isa PrescribedVelocityFields
        throw(ArgumentError("TracerTimeStepSplitting is not supported with PrescribedVelocityFields."))
    end

    for name in slow_names
        if name ∉ keys(tracers)
            throw(ArgumentError("The slow tracer $name is not one of the model tracers $(keys(tracers))."))
        end
    end

    closure_tracers = closure_required_tracers(closure)
    for name in slow_names
        if name ∈ closure_tracers
            throw(ArgumentError("The closure tracer $name cannot be a slow tracer in TracerTimeStepSplitting."))
        end
    end

    if vertical_coordinate isa ZStarCoordinate && !(timestepper isa SplitRungeKuttaTimeStepper)
        throw(ArgumentError("TracerTimeStepSplitting with ZStarCoordinate requires a SplitRungeKuttaTimeStepper."))
    end

    if !isnothing(biogeochemistry_substeps) && isnothing(biogeochemistry)
        throw(ArgumentError("TracerTimeStepSplitting biogeochemistry_substeps requires a biogeochemistry model."))
    end

    return nothing
end

split_tracer_grid_storage(vertical_coordinate, grid) = nothing, nothing

function split_tracer_grid_storage(::ZStarCoordinate, grid::MutableGridOfSomeKind)
    initial_displacement = similar(grid.z.ηⁿ)
    parent(initial_displacement) .= parent(grid.z.ηⁿ)

    saved_grid_state = (ηⁿ   = similar(grid.z.ηⁿ),
                        σᶜᶜ⁻ = similar(grid.z.σᶜᶜ⁻),
                        ∂t_σ = similar(grid.z.∂t_σ))

    return initial_displacement, saved_grid_state
end

# The split Runge-Kutta cache of the slow tracers is never used by the fast step, so the long step reuses it
previous_slow_tracer_fields(timestepper::SplitRungeKuttaTimeStepper, slow_tracers) =
    NamedTuple{keys(slow_tracers)}(Tuple(timestepper.Ψ⁻[name] for name in keys(slow_tracers)))

previous_slow_tracer_fields(timestepper, slow_tracers) = map(similar, slow_tracers)

#####
##### Queries used by the fast step
#####

@inline is_slow_tracer(::Nothing, name) = false
@inline is_slow_tracer(splitting::TracerTimeStepSplitting, ::Val{name}) where name = name ∈ keys(splitting.slow_tracers)

fast_tracers(model::HydrostaticFreeSurfaceModel) = fast_tracers(model.tracer_time_step_splitting, model.tracers)
fast_tracers(::Nothing, tracers) = tracers
fast_tracers(splitting::TracerTimeStepSplitting, tracers) = splitting.fast_tracers

mask_immersed_tracers!(model::HydrostaticFreeSurfaceModel) = mask_immersed_tracers!(model.tracer_time_step_splitting, model)
mask_immersed_tracers!(::Nothing, model) = mask_immersed_field!(model.tracers)

# Slow tracers only change during their long step, which masks them; they are masked again at the start of every cycle
# so that the state set before the first long step is masked too.
function mask_immersed_tracers!(splitting::TracerTimeStepSplitting, model)
    mask_immersed_field!(splitting.fast_tracers)
    splitting.accumulated_steps == 0 && mask_immersed_field!(splitting.slow_tracers)
    return nothing
end

#####
##### Accumulation of transports and closure fields
#####

# Split Runge-Kutta: accumulate the transport of the final stage, which moves the state from tⁿ to tⁿ⁺¹,
# while the stretching factor used by its tracer fluxes is still stored in the grid.
accumulate_split_tracer_transport!(::Nothing, model, timestepper) = nothing
accumulate_split_tracer_transport!(splitting, model, timestepper) = nothing

function accumulate_split_tracer_transport!(splitting::TracerTimeStepSplitting, model, timestepper::SplitRungeKuttaTimeStepper)
    model.clock.stage == timestepper.Nstages || return nothing
    accumulate_split_tracer_state!(splitting, model, model.clock.last_stage_Δt)
    return nothing
end

# Other time steppers (static vertical coordinate only) accumulate at the end of the time step.
accumulate_split_tracer_transport_at_end_of_step!(splitting, model, timestepper::SplitRungeKuttaTimeStepper, Δt) = nothing
accumulate_split_tracer_transport_at_end_of_step!(splitting, model, timestepper, Δt) =
    accumulate_split_tracer_state!(splitting, model, Δt)

function accumulate_split_tracer_state!(splitting, model, Δt)
    grid = model.grid
    arch = architecture(grid)
    FT = eltype(grid)
    Fᵘ, Fᵛ = splitting.transport
    ũ, ṽ, _ = model.transport_velocities

    launch!(arch, grid, size(Fᵘ), _accumulate_stretched_transport!, Fᵘ, grid, ũ, convert(FT, Δt), Face(), Center())
    launch!(arch, grid, size(Fᵛ), _accumulate_stretched_transport!, Fᵛ, grid, ṽ, convert(FT, Δt), Center(), Face())

    accumulate_fields!(splitting.closure_fields, model.closure_fields, convert(FT, Δt))

    splitting.accumulated_time += Δt

    return nothing
end

@kernel function _accumulate_stretched_transport!(F, grid, u, Δt, ℓx, ℓy)
    i, j, k = @index(Global, NTuple)
    @inbounds F[i, j, k] += Δt * u[i, j, k] * σⁿ(i, j, k, grid, ℓx, ℓy, Center())
end

@kernel function _reconstruct_slow_velocity!(u, grid, F, T⁻¹, ℓx, ℓy)
    i, j, k = @index(Global, NTuple)
    @inbounds u[i, j, k] = F[i, j, k] * T⁻¹ / σⁿ(i, j, k, grid, ℓx, ℓy, Center())
end

# Closure fields may alias each other (e.g. per-tracer diffusivity tuples), so every array is accumulated once.
accumulate_fields!(sum, fields, Δt) = accumulate_fields!(sum, fields, Δt, IdDict{Any, Nothing}())

accumulate_fields!(sum, fields, Δt, seen) = nothing

function accumulate_fields!(sum::Field, field::Field, Δt, seen)
    haskey(seen, sum) && return nothing
    seen[sum] = nothing
    parent(sum) .+= Δt .* parent(field)
    return nothing
end

function accumulate_fields!(sum::Union{Tuple, NamedTuple}, fields::Union{Tuple, NamedTuple}, Δt, seen)
    for (s, f) in zip(values(sum), values(fields))
        accumulate_fields!(s, f, Δt, seen)
    end
    return nothing
end

scale_fields!(fields, a) = scale_fields!(fields, a, IdDict{Any, Nothing}())

scale_fields!(fields, a, seen) = nothing

function scale_fields!(field::Field, a, seen)
    haskey(seen, field) && return nothing
    seen[field] = nothing
    parent(field) .*= a
    return nothing
end

function scale_fields!(fields::Union{Tuple, NamedTuple}, a, seen)
    for field in values(fields)
        scale_fields!(field, a, seen)
    end
    return nothing
end

#####
##### The long step
#####

step_split_tracers!(model::HydrostaticFreeSurfaceModel, Δt) =
    step_split_tracers!(model.tracer_time_step_splitting, model, Δt)

step_split_tracers!(::Nothing, model, Δt) = nothing

function step_split_tracers!(splitting::TracerTimeStepSplitting, model, Δt)
    accumulate_split_tracer_transport_at_end_of_step!(splitting, model, model.timestepper, Δt)
    splitting.accumulated_steps += 1

    if splitting.accumulated_steps ≥ splitting.ratio
        long_tracer_time_step!(splitting, model)
        reset_tracer_time_step_splitting!(splitting, model)
    end

    return nothing
end

function reset_tracer_time_step_splitting!(splitting, model)
    fill!(parent(splitting.transport.u), 0)
    fill!(parent(splitting.transport.v), 0)
    scale_fields!(splitting.closure_fields, 0)
    splitting.accumulated_steps = 0
    splitting.accumulated_time = 0
    store_initial_displacement!(splitting, model.vertical_coordinate, model.grid)
    return nothing
end

store_initial_displacement!(splitting, vertical_coordinate, grid) = nothing

function store_initial_displacement!(splitting, ::ZStarCoordinate, grid::MutableGridOfSomeKind)
    parent(splitting.initial_displacement) .= parent(grid.z.ηⁿ)
    return nothing
end

# Called when the model state is reconciled; a new cycle starts from the current free surface.
reconcile_tracer_time_step_splitting!(::Nothing, model) = nothing

function reconcile_tracer_time_step_splitting!(splitting::TracerTimeStepSplitting, model)
    splitting.accumulated_steps == 0 && store_initial_displacement!(splitting, model.vertical_coordinate, model.grid)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Advance the slow tracers of `splitting` over the accumulated time `T` with a
three-stage split Runge-Kutta step that uses the time-mean transports and closure fields.
"""
function long_tracer_time_step!(splitting, model)
    grid = model.grid
    T = splitting.accumulated_time
    clock = splitting.clock
    end_time = model.clock.time

    fill_halo_regions!(splitting.slow_tracers, model.clock, fields(model))
    update_biogeochemical_state!(model.biogeochemistry, model)

    scale_fields!(splitting.closure_fields, 1 / T)

    save_grid_state!(splitting, model.vertical_coordinate, grid)
    compute_long_step_grid_velocity!(splitting, model.vertical_coordinate, grid, T)
    set_long_step_grid!(splitting, model.vertical_coordinate, grid, 0)

    substeps = splitting.biogeochemistry_substeps
    biogeochemistry = long_step_biogeochemistry(model.biogeochemistry, substeps)

    start_time = next_time(model.clock, - T)
    step_biogeochemical_sources!(splitting, model, substeps, start_time, T / 2)

    cache_slow_tracers!(splitting, grid)

    stage_time = start_time

    stage_denominators = (3, 2, 1)
    Nstages = length(stage_denominators)

    for (stage, β) in enumerate(stage_denominators)
        Δτ = T / β
        clock.time = stage_time
        clock.stage = stage
        clock.last_stage_Δt = Δτ
        clock.last_Δt = T

        compute_long_step_velocities!(splitting, grid, T)
        set_slow_advection_timestep!(model.advection, splitting.slow_tracers, Δτ)
        compute_slow_tracer_tendencies!(splitting, model, biogeochemistry)

        # Horizontal flux-form semi-Lagrangian step of the slow tracers over the whole long step (on the final stage)
        slow_flux_form_semi_lagrangian_advection!(splitting, model, stage, Nstages, Δτ)

        advance_long_step_grid!(splitting, model.vertical_coordinate, grid, 1 / β)
        substep_slow_tracers!(splitting, model, Δτ)

        stage_time = β == 1 ? end_time : next_time(model.clock, Δτ - T)
    end

    step_biogeochemical_sources!(splitting, model, substeps, next_time(model.clock, - T / 2), T / 2)

    restore_grid_state!(splitting, model.vertical_coordinate, grid)

    # The fast tracers' adaptive vertical advection time step is refreshed as in `update_state!`
    update_advection!(model.advection, model)

    return nothing
end

#####
##### Grid evolution during the long step
#####

save_grid_state!(splitting, vertical_coordinate, grid) = nothing
restore_grid_state!(splitting, vertical_coordinate, grid) = nothing
compute_long_step_grid_velocity!(splitting, vertical_coordinate, grid, T) = nothing
set_long_step_grid!(splitting, vertical_coordinate, grid, fraction) = nothing
advance_long_step_grid!(splitting, vertical_coordinate, grid, fraction) = nothing

function save_grid_state!(splitting, ::ZStarCoordinate, grid::MutableGridOfSomeKind)
    saved = splitting.saved_grid_state
    parent(saved.ηⁿ)   .= parent(grid.z.ηⁿ)
    parent(saved.σᶜᶜ⁻) .= parent(grid.z.σᶜᶜ⁻)
    parent(saved.∂t_σ) .= parent(grid.z.∂t_σ)
    return nothing
end

# The final stage lands exactly on the saved free surface, so the scalings are recomputed bitwise;
# only the previous scaling and the grid velocity need to be restored.
function restore_grid_state!(splitting, ::ZStarCoordinate, grid::MutableGridOfSomeKind)
    saved = splitting.saved_grid_state
    parent(grid.z.ηⁿ)   .= parent(saved.ηⁿ)
    parent(grid.z.σᶜᶜ⁻) .= parent(saved.σᶜᶜ⁻)
    parent(grid.z.∂t_σ) .= parent(saved.∂t_σ)
    return nothing
end

function compute_long_step_grid_velocity!(splitting, ::ZStarCoordinate, grid::MutableGridOfSomeKind, T)
    FT = eltype(grid)
    η₀ = splitting.initial_displacement
    η₁ = splitting.saved_grid_state.ηⁿ
    launch!(architecture(grid), grid, surface_kernel_parameters(grid),
            _compute_long_step_grid_velocity!, grid.z.∂t_σ, grid, η₀, η₁, convert(FT, 1 / T))
    return nothing
end

# ∂t_σ = (σ(η₁) - σ(η₀)) / T with the same σ = H / h as `update_grid_scaling!`
@kernel function _compute_long_step_grid_velocity!(∂t_σ, grid, η₀, η₁, T⁻¹)
    i, j = @index(Global, NTuple)
    h  = static_column_depthᶜᶜᵃ(i, j, grid)
    H₀ = column_depthᶜᶜᵃ(i, j, 1, grid, η₀)
    H₁ = column_depthᶜᶜᵃ(i, j, 1, grid, η₁)
    σ₀ = ifelse(h == 0, one(grid), H₀ / h)
    σ₁ = ifelse(h == 0, one(grid), H₁ / h)
    @inbounds ∂t_σ[i, j, 1] = (σ₁ - σ₀) * T⁻¹
end

# Set the grid to the free surface `(1 - f) η₀ + f η₁`; `f = 1` recovers `η₁` exactly.
function set_long_step_grid!(splitting, ::ZStarCoordinate, grid::MutableGridOfSomeKind, fraction)
    FT = eltype(grid)
    f  = convert(FT, fraction)
    η₀ = parent(splitting.initial_displacement)
    η₁ = parent(splitting.saved_grid_state.ηⁿ)
    parent(grid.z.ηⁿ) .= (1 - f) .* η₀ .+ f .* η₁
    launch!(architecture(grid), grid, surface_kernel_parameters(grid), _update_grid_scaling!, grid)
    return nothing
end

function advance_long_step_grid!(splitting, vertical_coordinate::ZStarCoordinate, grid::MutableGridOfSomeKind, fraction)
    parent(grid.z.σᶜᶜ⁻) .= parent(grid.z.σᶜᶜⁿ)
    set_long_step_grid!(splitting, vertical_coordinate, grid, fraction)
    return nothing
end

#####
##### Velocities, tendencies and tracer update during the long step
#####

function compute_long_step_velocities!(splitting, grid, T)
    arch = architecture(grid)
    FT = eltype(grid)
    T⁻¹ = convert(FT, 1 / T)
    Fᵘ, Fᵛ = splitting.transport
    ū, v̄, _ = splitting.velocities

    launch!(arch, grid, size(ū), _reconstruct_slow_velocity!, ū, grid, Fᵘ, T⁻¹, Face(), Center())
    launch!(arch, grid, size(v̄), _reconstruct_slow_velocity!, v̄, grid, Fᵛ, T⁻¹, Center(), Face())
    fill_halo_regions!((ū, v̄))

    # `grid.z.∂t_σ` holds the long-step grid velocity, so `w̄` is consistent with the interpolated free surface
    compute_w_from_continuity!(splitting.velocities, grid)

    return nothing
end

set_slow_advection_timestep!(advection, slow_tracers, Δt) =
    foreach(name -> set_adaptive_advection_timestep!(advection[name], Δt), keys(slow_tracers))

set_adaptive_advection_timestep!(scheme, Δt) = nothing
set_adaptive_advection_timestep!(scheme::FluxFormAdvection, Δt) = set_adaptive_advection_timestep!(scheme.z, Δt)

function set_adaptive_advection_timestep!(scheme::AdaptiveImplicitVerticalAdvection, Δt)
    time_discretization(scheme).Δt[] = Δt
    return nothing
end

function cache_slow_tracers!(splitting, grid)
    for name in keys(splitting.slow_tracers)
        launch!(architecture(grid), grid, :xyz, _cache_tracer_fields!,
                splitting.previous_tracers[name], grid, splitting.slow_tracers[name])
    end
    return nothing
end

compute_slow_tracer_tendencies!(splitting, model, biogeochemistry) =
    compute_slow_tracer_tendencies!(splitting, model, biogeochemistry, Val(1), Val(keys(model.tracers)))

compute_slow_tracer_tendencies!(splitting, model, biogeochemistry, ::Val, ::Val{()}) = nothing

function compute_slow_tracer_tendencies!(splitting, model, biogeochemistry, ::Val{tracer_index}, ::Val{names}) where {tracer_index, names}
    tracer_name = first(names)

    if is_slow_tracer(splitting, Val(tracer_name))
        compute_slow_tracer_tendency!(splitting, model, biogeochemistry, Val(tracer_index), Val(tracer_name))
    end

    compute_slow_tracer_tendencies!(splitting, model, biogeochemistry, Val(tracer_index + 1), Val(Base.tail(names)))
    return nothing
end

function compute_slow_tracer_tendency!(splitting, model, biogeochemistry, ::Val{tracer_index}, ::Val{tracer_name}) where {tracer_index, tracer_name}
    grid = model.grid
    arch = architecture(grid)
    clock = splitting.clock

    Gⁿ = model.timestepper.Gⁿ[tracer_name]
    c = model.tracers[tracer_name]

    launch!(arch, grid, :xyz,
            compute_hydrostatic_free_surface_Gc!,
            Gⁿ,
            grid,
            Val(tracer_index),
            Val(tracer_name),
            model.advection[tracer_name],
            model.closure,
            immersed_boundary_condition(c),
            model.buoyancy,
            biogeochemistry,
            splitting.velocities,
            model.free_surface,
            model.tracers,
            splitting.closure_fields,
            model.auxiliary_fields,
            clock,
            model.forcing[tracer_name])

    args = (clock, fields(model), model.closure, model.buoyancy)
    compute_flux_bcs!(Gⁿ, c, arch, args)

    launch!(arch, grid, :xyz, _scale_by_stretching_factor!, Gⁿ, grid)

    return nothing
end

substep_slow_tracers!(splitting, model, Δτ) =
    substep_slow_tracers!(splitting, model, Δτ, Val(1), Val(keys(model.tracers)))

substep_slow_tracers!(splitting, model, Δτ, ::Val, ::Val{()}) = nothing

function substep_slow_tracers!(splitting, model, Δτ, ::Val{tracer_index}, ::Val{names}) where {tracer_index, names}
    tracer_name = first(names)

    if is_slow_tracer(splitting, Val(tracer_name))
        substep_slow_tracer!(splitting, model, Δτ, Val(tracer_index), Val(tracer_name))
    end

    substep_slow_tracers!(splitting, model, Δτ, Val(tracer_index + 1), Val(Base.tail(names)))
    return nothing
end

function substep_slow_tracer!(splitting, model, Δτ, ::Val{tracer_index}, ::Val{tracer_name}) where {tracer_index, tracer_name}
    grid = model.grid
    FT = eltype(grid)
    c = model.tracers[tracer_name]
    Gⁿ = model.timestepper.Gⁿ[tracer_name]
    σc⁻ = splitting.previous_tracers[tracer_name]

    launch!(architecture(grid), grid, :xyz, _rk_substep_tracer_field!, c, grid, convert(FT, Δτ), Gⁿ, σc⁻)

    c_velocities = tracer_advecting_velocities(splitting.velocities, model.biogeochemistry, model.closure,
                                               splitting.closure_fields, model.forcing[tracer_name], Val(tracer_name))

    implicit_step!(c,
                   model.timestepper.implicit_solver,
                   model.closure,
                   splitting.closure_fields,
                   Val(tracer_index),
                   splitting.clock,
                   fields(model),
                   Δτ,
                   model.advection[tracer_name],
                   c_velocities)

    mask_immersed_field!(c)
    fill_halo_regions!(c, splitting.clock, fields(model))

    return nothing
end

#####
##### Strang-split biogeochemical sources
#####

"""
    TransportOnlyBiogeochemistry(biogeochemistry)

Wrap `biogeochemistry` so that it contributes drift velocities and auxiliary fields to a tracer
tendency, but no biogeochemical sources. Used for the transport part of a Strang-split long step.
"""
struct TransportOnlyBiogeochemistry{B}
    biogeochemistry :: B
end

Adapt.adapt_structure(to, bgc::TransportOnlyBiogeochemistry) =
    TransportOnlyBiogeochemistry(Adapt.adapt(to, bgc.biogeochemistry))

@inline biogeochemical_transition(i, j, k, grid, ::TransportOnlyBiogeochemistry, val_tracer_name, clock, fields) = zero(grid)
@inline biogeochemical_drift_velocity(bgc::TransportOnlyBiogeochemistry, val_tracer_name) =
    biogeochemical_drift_velocity(bgc.biogeochemistry, val_tracer_name)
@inline biogeochemical_auxiliary_fields(bgc::TransportOnlyBiogeochemistry) =
    biogeochemical_auxiliary_fields(bgc.biogeochemistry)

long_step_biogeochemistry(biogeochemistry, ::Nothing) = biogeochemistry
long_step_biogeochemistry(biogeochemistry, substeps) = TransportOnlyBiogeochemistry(biogeochemistry)

step_biogeochemical_sources!(splitting, model, ::Nothing, start_time, Δt) = nothing

# Integrate `∂t c = S(c)` pointwise over `Δt` with `substeps` low-storage Runge-Kutta sub-steps.
function step_biogeochemical_sources!(splitting, model, substeps, start_time, Δt)
    grid = model.grid
    arch = architecture(grid)
    FT = eltype(grid)
    clock = splitting.clock
    δt = Δt / substeps
    slow_tracers = splitting.slow_tracers
    cached_tracers = splitting.previous_tracers
    val_tracer_names = map(Val, keys(slow_tracers))

    update_biogeochemical_state!(model.biogeochemistry, model)

    clock.time = start_time

    for substep in 1:substeps
        substep_start_time = clock.time

        for name in keys(slow_tracers)
            parent(cached_tracers[name]) .= parent(slow_tracers[name])
        end

        # Each stage restarts from the cached state and is evaluated at the time reached by the previous stage
        for β in (3, 2, 1)
            launch!(arch, grid, :xyz, _biogeochemical_source_substep!,
                    slow_tracers, grid, model.biogeochemistry, clock, fields(model),
                    cached_tracers, convert(FT, δt / β), val_tracer_names)

            clock.time = substep_start_time
            clock.time = next_time(clock, δt / β)
        end
    end

    fill_halo_regions!(slow_tracers, clock, fields(model))

    return nothing
end

@kernel function _biogeochemical_source_substep!(tracers, grid, biogeochemistry, clock, model_fields, cached_tracers, Δt, val_tracer_names)
    i, j, k = @index(Global, NTuple)

    N = length(val_tracer_names)
    sources = map(val_name -> biogeochemical_transition(i, j, k, grid, biogeochemistry, val_name, clock, model_fields), val_tracer_names)
    immersed = inactive_node(i, j, k, grid, Center(), Center(), Center())

    tracer_values = values(tracers)
    cached_values = values(cached_tracers)

    ntuple(Val(N)) do n
        Base.@_inline_meta
        @inbounds begin
            c⁰ = cached_values[n][i, j, k]
            tracer_values[n][i, j, k] = ifelse(immersed, c⁰, c⁰ + Δt * sources[n])
        end
    end
end

#####
##### Checkpointing
#####

function prognostic_state(splitting::TracerTimeStepSplitting)
    return (transport = prognostic_state(splitting.transport),
            closure_fields = prognostic_state(splitting.closure_fields),
            initial_displacement = prognostic_state(splitting.initial_displacement),
            accumulated_steps = splitting.accumulated_steps,
            accumulated_time = splitting.accumulated_time)
end

function restore_prognostic_state!(splitting::TracerTimeStepSplitting, from)
    restore_prognostic_state!(splitting.transport, from.transport)
    restore_prognostic_state!(splitting.closure_fields, from.closure_fields)
    restore_prognostic_state!(splitting.initial_displacement, from.initial_displacement)
    splitting.accumulated_steps = from.accumulated_steps
    splitting.accumulated_time = from.accumulated_time
    return splitting
end

restore_prognostic_state!(::TracerTimeStepSplitting, ::Nothing) = nothing
