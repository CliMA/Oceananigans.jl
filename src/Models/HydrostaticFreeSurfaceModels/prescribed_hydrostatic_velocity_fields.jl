#####
##### PrescribedVelocityFields
#####

using Oceananigans: location
using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: FunctionField, Field, field
using Oceananigans.TimeSteppers: tick!, step_lagrangian_particles!, Clock
using Oceananigans.BoundaryConditions: BoundaryConditions, fill_halo_regions!
using Oceananigans.OutputReaders: FieldTimeSeries, TimeSeriesInterpolation

import Oceananigans: prognostic_state, restore_prognostic_state!
import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.DistributedComputations: synchronize_communication!
import Oceananigans.Models: extract_boundary_conditions
import Oceananigans.Utils: datatuple, sum_of_velocities
import Oceananigans.TimeSteppers: time_step!

struct PrescribedVelocityFields{F, U, V, W, P}
    formulation :: F
    u :: U
    v :: V
    w :: W
    parameters :: P
end

const PVF = PrescribedVelocityFields

@inline Base.getindex(U::PrescribedVelocityFields, i) = getindex((u=U.u, v=U.v, w=U.w), i)

#####
##### DiagnosticVerticalVelocity
#####

"""
    DiagnosticVerticalVelocity()

A formulation type indicating that the vertical velocity `w` should be diagnosed from the
horizontal velocity fields `u` and `v` via the continuity equation, rather than
being prescribed.

When passed as the `formulation` argument to `PrescribedVelocityFields`, a
`Field{Center, Center, Face}` is allocated for `w` during model construction
and filled at each time step via `compute_w_from_continuity!`.

```jldoctest
julia> using Oceananigans

julia> DiagnosticVerticalVelocity()
DiagnosticVerticalVelocity()
```
"""
struct DiagnosticVerticalVelocity end

Base.show(io::IO, ::DiagnosticVerticalVelocity) = print(io, "DiagnosticVerticalVelocity()")

"""
    PrescribedVelocityFields(; u = ZeroField(),
                               v = ZeroField(),
                               w = ZeroField(),
                               formulation = nothing,
                               parameters = nothing)

Build `PrescribedVelocityFields` with prescribed `u`, `v`, and `w`.

Each of `u`, `v`, and `w` may be:

- A `Function` with signature `u(x, y, z, t)` (or `u(x, y, z, t, parameters)` when
  `parameters` is provided). Functions are wrapped in `FunctionField` during model
  construction and associated with the model's `grid` and `clock`.
- A `Field` with the appropriate staggering (`u` at `(Face, Center, Center)`,
  `v` at `(Center, Face, Center)`, `w` at `(Center, Center, Face)`).
- A `FieldTimeSeries`, which is wrapped in a `TimeSeriesInterpolation` that
  interpolates to the model clock time at each time step. The `FieldTimeSeries`
  must already have the correct staggered location.
- A `ZeroField()` (default).

`formulation` may be:

- `nothing` (default): `w` is prescribed directly.
- `DiagnosticVerticalVelocity()`: a `Field{Center, Center, Face}` is allocated for `w`
  during model construction and filled at each time step via `compute_w_from_continuity!`.
  The `w` keyword is ignored in this case.

```jldoctest
julia> using Oceananigans

julia> PrescribedVelocityFields().u
ZeroField{Int64}
```

Using `Field` arguments:

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> PrescribedVelocityFields(u = XFaceField(grid)).u
4×4×4 Field{Face, Center, Center} on RectilinearGrid on CPU
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
└── data: 10×10×10 OffsetArray(::Array{Float64, 3}, -2:7, -2:7, -2:7) with eltype Float64 with indices -2:7×-2:7×-2:7
    └── max=0.0, min=0.0, mean=0.0
```

Using a `FieldTimeSeries`:

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> u_fts = FieldTimeSeries{Face, Center, Center}(grid, 0:0.5:1);

julia> PrescribedVelocityFields(u = u_fts).u
4×4×4×3 FieldTimeSeries{InMemory} located at (Face, Center, Center) on CPU
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── indices: (:, :, :)
├── time_indexing: Clamp()
├── backend: InMemory()
└── data: 10×10×10×3 OffsetArray(::Array{Float64, 4}, -2:7, -2:7, -2:7, 1:3) with eltype Float64 with indices -2:7×-2:7×-2:7×1:3
    └── max=0.0, min=0.0, mean=0.0
```

Using `DiagnosticVerticalVelocity` as the formulation:

```jldoctest
julia> using Oceananigans

julia> pvf = PrescribedVelocityFields(formulation = DiagnosticVerticalVelocity());

julia> pvf.formulation
DiagnosticVerticalVelocity()
```
"""
function PrescribedVelocityFields(; u = ZeroField(),
                                    v = ZeroField(),
                                    w = ZeroField(),
                                    formulation = nothing,
                                    parameters = nothing)

    if formulation isa DiagnosticVerticalVelocity
        return PrescribedVelocityFields(formulation, u, v, nothing, parameters)
    else
        return PrescribedVelocityFields(formulation, u, v, w, parameters)
    end
end

materialize_prescribed_velocity(X, Y, Z, f::Function, grid; kwargs...) = FunctionField{X, Y, Z}(f, grid; kwargs...)

function materialize_prescribed_velocity(X, Y, Z, fts::FieldTimeSeries, grid; clock, kwargs...)
    fts_location = location(fts)
    requested_location = (X, Y, Z)
    if fts_location != requested_location
        throw(ArgumentError("FieldTimeSeries location $fts_location does not match " *
                            "the expected velocity location $requested_location"))
    end
    return TimeSeriesInterpolation(fts, grid; clock)
end

materialize_prescribed_velocity(X, Y, Z, f, grid; kwargs...) = field((X, Y, Z), f, grid)

function hydrostatic_velocity_fields(velocities::PrescribedVelocityFields{<:DiagnosticVerticalVelocity}, grid, clock, bcs)

    parameters = velocities.parameters
    u = materialize_prescribed_velocity(Face, Center, Center, velocities.u, grid; clock, parameters)
    v = materialize_prescribed_velocity(Center, Face, Center, velocities.v, grid; clock, parameters)
    w = Field{Center, Center, Face}(grid)

    fill_halo_regions!((u, v))
    fill_halo_regions!(w)

    return PrescribedVelocityFields(DiagnosticVerticalVelocity(), u, v, w, parameters)
end

function hydrostatic_velocity_fields(velocities::PrescribedVelocityFields, grid, clock, bcs)

    parameters = velocities.parameters
    u = materialize_prescribed_velocity(Face, Center, Center, velocities.u, grid; clock, parameters)
    v = materialize_prescribed_velocity(Center, Face, Center, velocities.v, grid; clock, parameters)
    w = materialize_prescribed_velocity(Center, Center, Face, velocities.w, grid; clock, parameters)

    fill_halo_regions!((u, v))
    fill_halo_regions!(w)

    return PrescribedVelocityFields(nothing, u, v, w, parameters)
end

# Allow u, v, w = velocities when velocities isa PrescribedVelocityFields
function Base.indexed_iterate(p::PrescribedVelocityFields, i::Int, state=1)
    if i == 1
        return p.u, 2
    elseif i == 2
        return p.v, 3
    else
        return p.w, 4
    end
end

hydrostatic_tendency_fields(::PrescribedVelocityFields, free_surface, grid, tracer_names, bcs) =
    merge((u=nothing, v=nothing), TracerFields(tracer_names, grid))

free_surface_names(free_surface, ::PrescribedVelocityFields, grid) = tuple()
free_surface_names(::SplitExplicitFreeSurface, ::PrescribedVelocityFields, grid) = tuple()

@inline BoundaryConditions.fill_halo_regions!(::PrescribedVelocityFields, args...; kwargs...) = nothing
@inline BoundaryConditions.fill_halo_regions!(::FunctionField, args...; kwargs...) = nothing
@inline BoundaryConditions.fill_halo_regions!(::TimeSeriesInterpolation, args...; kwargs...) = nothing

@inline datatuple(obj::PrescribedVelocityFields) = (; u = datatuple(obj.u), v = datatuple(obj.v), w = datatuple(obj.w))
@inline velocities(obj::PrescribedVelocityFields) = (u = obj.u, v = obj.v, w = obj.w)

# Extend sum_of_velocities for `PrescribedVelocityFields`
@inline sum_of_velocities(U1::PrescribedVelocityFields, U2) = sum_of_velocities(velocities(U1), U2)
@inline sum_of_velocities(U1, U2::PrescribedVelocityFields) = sum_of_velocities(U1, velocities(U2))

@inline sum_of_velocities(U1::PrescribedVelocityFields, U2, U3) = sum_of_velocities(velocities(U1), U2, U3)
@inline sum_of_velocities(U1, U2::PrescribedVelocityFields, U3) = sum_of_velocities(U1, velocities(U2), U3)
@inline sum_of_velocities(U1, U2, U3::PrescribedVelocityFields) = sum_of_velocities(U1, U2, velocities(U3))

ab2_step_velocities!(::PrescribedVelocityFields, args...) = nothing
rk_substep_velocities!(::PrescribedVelocityFields, args...) = nothing
step_free_surface!(::Nothing, model, timestepper, Δt) = nothing
compute_w_from_continuity!(::PrescribedVelocityFields, args...; kwargs...) = nothing

function compute_w_from_continuity!(velocities::PrescribedVelocityFields{<:DiagnosticVerticalVelocity},
                                    grid; parameters = surface_kernel_parameters(grid))
    w = velocities.w
    vels = (u=velocities.u, v=velocities.v, w=w)
    compute_w_from_continuity!(vels, grid; parameters)
end

mask_immersed_velocities!(::PrescribedVelocityFields) = nothing

# No need for extra velocities
transport_velocity_fields(velocities::PrescribedVelocityFields, free_surface) = velocities
transport_velocity_fields(velocities::PrescribedVelocityFields, ::ExplicitFreeSurface) = velocities
transport_velocity_fields(velocities::PrescribedVelocityFields, ::Nothing) = velocities

validate_velocity_boundary_conditions(grid, ::PrescribedVelocityFields) = nothing
extract_boundary_conditions(::PrescribedVelocityFields) = NamedTuple()

free_surface_displacement_field(::PrescribedVelocityFields, ::Nothing, grid) = nothing
HorizontalVelocityFields(::PrescribedVelocityFields, grid) = nothing, nothing

materialize_free_surface(::Nothing,                      ::PrescribedVelocityFields, grid, args...) = nothing
materialize_free_surface(::ExplicitFreeSurface{Nothing}, ::PrescribedVelocityFields, grid, args...) = nothing
materialize_free_surface(::ImplicitFreeSurface{Nothing}, ::PrescribedVelocityFields, grid, args...) = nothing
materialize_free_surface(::SplitExplicitFreeSurface,     ::PrescribedVelocityFields, grid, args...) = nothing

hydrostatic_prognostic_fields(::PrescribedVelocityFields, ::Nothing, tracers) = tracers
compute_hydrostatic_momentum_tendencies!(model, ::PrescribedVelocityFields, kernel_parameters; kwargs...) = nothing

compute_flux_bcs!(::Nothing, c, arch, clock, model_fields) = nothing

Adapt.adapt_structure(to, velocities::PrescribedVelocityFields) =
    PrescribedVelocityFields(velocities.formulation,
                             Adapt.adapt(to, velocities.u),
                             Adapt.adapt(to, velocities.v),
                             Adapt.adapt(to, velocities.w),
                             nothing) # Why are parameters not passed here? They probably should...

on_architecture(to, velocities::PrescribedVelocityFields) =
    PrescribedVelocityFields(velocities.formulation,
                             on_architecture(to, velocities.u),
                             on_architecture(to, velocities.v),
                             on_architecture(to, velocities.w),
                             on_architecture(to, velocities.parameters))

# If the model only tracks particles... do nothing but that!!!
const OnlyParticleTrackingModel = HydrostaticFreeSurfaceModel{TS, E, A, S, G, T, V, B, R, F, P, U, W, C} where
                 {TS, E, A, S, G, T, V, B, R, F, P<:AbstractLagrangianParticles, U<:PrescribedVelocityFields, W<:PrescribedVelocityFields, C<:NamedTuple{(), Tuple{}}}

function time_step!(model::OnlyParticleTrackingModel, Δt; callbacks = [], kwargs...)
    tick!(model.clock, Δt)
    step_lagrangian_particles!(model, Δt)
    update_state!(model, callbacks)
end

update_state!(model::OnlyParticleTrackingModel, callbacks) =
    [callback(model) for callback in callbacks if callback.callsite isa UpdateStateCallsite]

prognostic_state(::PrescribedVelocityFields) = nothing
restore_prognostic_state!(::PrescribedVelocityFields, ::Nothing) = nothing

#####
##### PrescribedFreeSurface
#####

struct PrescribedFreeSurface{E, G, P} <: AbstractFreeSurface{E, G}
    displacement :: E
    gravitational_acceleration :: G
    parameters :: P
end

"""
    PrescribedFreeSurface(; displacement,
                            gravitational_acceleration = defaults.gravitational_acceleration,
                            parameters = nothing)

Build a `PrescribedFreeSurface` with a prescribed `displacement` field.

`displacement` may be a `Function` with signature `η(x, y, z, t)` (or
`η(x, y, z, t, parameters)` if `parameters` is provided), or a `FieldTimeSeries`.

The displacement is used by the `ZStarCoordinate` vertical coordinate to update
grid scaling factors, but the free surface is never stepped forward in time.

This is useful when combining `PrescribedVelocityFields` with a
`MutableVerticalDiscretization` grid.
"""
PrescribedFreeSurface(; displacement,
                        gravitational_acceleration = defaults.gravitational_acceleration,
                        parameters = nothing) =
    PrescribedFreeSurface(displacement, gravitational_acceleration, parameters)

#####
##### PrescribedFreeSurface materialization
#####

materialize_prescribed_displacement(f::Function, grid; clock, parameters) =
    FunctionField{Center, Center, Face}(f, grid; clock, parameters)

function materialize_prescribed_displacement(fts::FieldTimeSeries, grid; clock, parameters=nothing)
    return TimeSeriesInterpolation(fts, grid; clock)
end

# Fallback: if already a field, just return it
materialize_prescribed_displacement(f, grid; kwargs...) = f

function materialize_free_surface(free_surface::PrescribedFreeSurface, velocities, grid, clock)
    # Create a separate clock so that step_free_surface! can advance it to tⁿ⁺¹
    # before the grid update, matching prognostic free surfaces. The model clock
    # must not be modified because other operations (e.g. tracer tendency forcings)
    # still need it at tⁿ.
    displacement_clock = Clock(time = clock.time)
    η = materialize_prescribed_displacement(free_surface.displacement, grid;
                                            clock = displacement_clock,
                                            parameters = free_surface.parameters)
    g = convert(eltype(grid), free_surface.gravitational_acceleration)
    return PrescribedFreeSurface(η, g, free_surface.parameters)
end

# PrescribedFreeSurface is NOT nullified by PrescribedVelocityFields — delegate to its own method
materialize_free_surface(fs::PrescribedFreeSurface, ::PrescribedVelocityFields, grid, clock) =
    materialize_free_surface(fs, nothing, grid, clock)

#####
##### PrescribedFreeSurface time stepping
#####

# For a constant (time-independent) plain Field displacement: no clock to advance.
step_free_surface!(fs::PrescribedFreeSurface{<:Field}, model, timestepper, Δt) = nothing

# Advance the displacement's clock to tⁿ⁺¹ so the FunctionField /
# TimeSeriesInterpolation evaluates η at the new time, consistent with how
# a prognostic free surface is stepped forward before the grid update.
function step_free_surface!(fs::PrescribedFreeSurface, model, timestepper, Δt)
    fs.displacement.clock.time = model.clock.time + Δt
    return nothing
end

compute_free_surface_tendency!(grid, model, ::PrescribedFreeSurface) = nothing
correct_barotropic_mode!(model, ::PrescribedFreeSurface, Δt; kwargs...) = nothing

@inline explicit_barotropic_pressure_x_gradient(i, j, k, grid, ::PrescribedFreeSurface) = zero(grid)
@inline explicit_barotropic_pressure_y_gradient(i, j, k, grid, ::PrescribedFreeSurface) = zero(grid)

barotropic_velocities(::PrescribedFreeSurface) = (nothing, nothing)
barotropic_transport(::PrescribedFreeSurface) = (nothing, nothing)

synchronize_communication!(::PrescribedFreeSurface) = nothing

transport_velocity_fields(velocities, ::PrescribedFreeSurface) = velocities
transport_velocity_fields(velocities::PrescribedVelocityFields, ::PrescribedFreeSurface) = velocities

#####
##### PrescribedFreeSurface field introspection
#####

@inline free_surface_fields(free_surface::PrescribedFreeSurface) = (; η = free_surface.displacement)
@inline free_surface_names(::PrescribedFreeSurface, velocities, grid) = tuple(:η)
@inline free_surface_names(::PrescribedFreeSurface, ::PrescribedVelocityFields, grid) = tuple(:η)

# The displacement is not prognostic — it is prescribed
hydrostatic_prognostic_fields(velocities, free_surface::PrescribedFreeSurface, tracers) =
    merge(horizontal_velocities(velocities), tracers)

hydrostatic_prognostic_fields(::PrescribedVelocityFields, ::PrescribedFreeSurface, tracers) = tracers

hydrostatic_tendency_fields(velocities, ::PrescribedFreeSurface, grid, tracer_names, bcs) =
    hydrostatic_tendency_fields(velocities, nothing, grid, tracer_names, bcs)

hydrostatic_tendency_fields(::PrescribedVelocityFields, ::PrescribedFreeSurface, grid, tracer_names, bcs) =
    merge((u=nothing, v=nothing), TracerFields(tracer_names, grid))

free_surface_displacement_field(velocities, ::PrescribedFreeSurface, grid) = nothing

# No initialization needed — the displacement is prescribed
initialize_free_surface!(::PrescribedFreeSurface, grid, velocities) = nothing

#####
##### PrescribedFreeSurface Adapt and on_architecture
#####

Adapt.adapt_structure(to, fs::PrescribedFreeSurface) =
    PrescribedFreeSurface(Adapt.adapt(to, fs.displacement),
                          fs.gravitational_acceleration,
                          nothing)

on_architecture(to, fs::PrescribedFreeSurface) =
    PrescribedFreeSurface(on_architecture(to, fs.displacement),
                          on_architecture(to, fs.gravitational_acceleration),
                          on_architecture(to, fs.parameters))

#####
##### PrescribedFreeSurface checkpointing
#####

prognostic_state(::PrescribedFreeSurface) = nothing
restore_prognostic_state!(::PrescribedFreeSurface, ::Nothing) = nothing

#####
##### PrescribedFreeSurface + ZStarCoordinate: ∂t_σ from prescribed displacement
#####

# For a constant (time-independent) plain Field displacement: ∂t_η = 0 so ∂t_σ = 0.
update_grid_vertical_velocity!(velocities, model, grid::MutableGridOfSomeKind,
                               ::ZStarCoordinate, fs::PrescribedFreeSurface{<:Field};
                               parameters = surface_kernel_parameters(grid)) = nothing

# When PrescribedFreeSurface is used with a mutable z-star grid, compute ∂t_σ as a
# forward finite difference of the prescribed displacement instead of the barotropic
# transport divergence. At the time this method is called, step_free_surface! has
# already advanced displacement.clock to t + Δt, while grid.z.ηⁿ still holds η(tⁿ).
function update_grid_vertical_velocity!(velocities, model, grid::MutableGridOfSomeKind,
                                        ::ZStarCoordinate, fs::PrescribedFreeSurface;
                                        parameters=surface_kernel_parameters(grid))
    η_new = fs.displacement
    Δt    = η_new.clock.time - model.clock.time
    iszero(Δt) && return nothing   # initialization: leave ∂t_σ = 0
    ∂t_σ  = grid.z.∂t_σ
    launch!(architecture(grid), grid, parameters, _update_prescribed_∂t_σ!, ∂t_σ, grid, η_new, Δt)
    return nothing
end

@kernel function _update_prescribed_∂t_σ!(∂t_σ, grid, η_new, Δt)
    i, j  = @index(Global, NTuple)
    hᶜᶜ   = static_column_depthᶜᶜᵃ(i, j, grid)
    η_old  = @inbounds grid.z.ηⁿ[i, j, 1]
    η_next = @inbounds η_new[i, j, grid.Nz+1]
    ∂t_η   = (η_next - η_old) / Δt
    @inbounds ∂t_σ[i, j, 1] = ifelse(hᶜᶜ == 0, zero(grid), ∂t_η / hᶜᶜ)
end
