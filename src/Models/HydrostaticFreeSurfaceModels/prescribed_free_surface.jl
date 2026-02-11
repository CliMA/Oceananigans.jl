using Adapt: Adapt
using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: FunctionField
using Oceananigans.OutputReaders: FieldTimeSeries, TimeSeriesInterpolation

import Oceananigans: prognostic_state, restore_prognostic_state!
import Oceananigans.DistributedComputations: synchronize_communication!

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
##### Materialization
#####

materialize_prescribed_displacement(f::Function, grid; clock, parameters) =
    FunctionField{Center, Center, Face}(f, grid; clock, parameters)

function materialize_prescribed_displacement(fts::FieldTimeSeries, grid; clock, parameters=nothing)
    return TimeSeriesInterpolation(fts, grid; clock)
end

# Fallback: if already a field, just return it
materialize_prescribed_displacement(f, grid; kwargs...) = f

function materialize_free_surface(free_surface::PrescribedFreeSurface, velocities, grid, clock)
    η = materialize_prescribed_displacement(free_surface.displacement, grid;
                                            clock,
                                            parameters = free_surface.parameters)
    g = convert(eltype(grid), free_surface.gravitational_acceleration)
    return PrescribedFreeSurface(η, g, free_surface.parameters)
end

#####
##### No-op methods for time stepping
#####

step_free_surface!(::PrescribedFreeSurface, model, timestepper, Δt) = nothing
compute_free_surface_tendency!(grid, model, ::PrescribedFreeSurface) = nothing
correct_barotropic_mode!(model, ::PrescribedFreeSurface, Δt; kwargs...) = nothing

@inline explicit_barotropic_pressure_x_gradient(i, j, k, grid, ::PrescribedFreeSurface) = zero(grid)
@inline explicit_barotropic_pressure_y_gradient(i, j, k, grid, ::PrescribedFreeSurface) = zero(grid)

barotropic_velocities(::PrescribedFreeSurface) = (nothing, nothing)
barotropic_transport(::PrescribedFreeSurface) = (nothing, nothing)

synchronize_communication!(::PrescribedFreeSurface) = nothing

transport_velocity_fields(velocities, ::PrescribedFreeSurface) = velocities

#####
##### Field introspection
#####

@inline free_surface_fields(free_surface::PrescribedFreeSurface) = (; η = free_surface.displacement)
@inline free_surface_names(::PrescribedFreeSurface, velocities, grid) = tuple(:η)

# The displacement is not prognostic — it is prescribed
hydrostatic_prognostic_fields(velocities, free_surface::PrescribedFreeSurface, tracers) =
    merge(horizontal_velocities(velocities), tracers)

hydrostatic_tendency_fields(velocities, ::PrescribedFreeSurface, grid, tracer_names, bcs) =
    hydrostatic_tendency_fields(velocities, nothing, grid, tracer_names, bcs)

free_surface_displacement_field(velocities, ::PrescribedFreeSurface, grid) = nothing

# No initialization needed — the displacement is prescribed
initialize_free_surface!(::PrescribedFreeSurface, grid, velocities) = nothing

#####
##### Adapt and on_architecture
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
##### Checkpointing
#####

prognostic_state(::PrescribedFreeSurface) = nothing
restore_prognostic_state!(::PrescribedFreeSurface, ::Nothing) = nothing
