using Oceananigans.Grids: Center
using Oceananigans.Fields: FunctionField, Field
using Oceananigans.OutputReaders: FieldTimeSeries, TimeSeriesInterpolation, GPUAdaptedTimeSeriesInterpolation

import Oceananigans.Models: extract_boundary_conditions

#####
##### PrescribedTracer
#####

struct PrescribedTracer{F, P}
    field :: F
    parameters :: P
end

"""
    PrescribedTracer(field; parameters=nothing)

Wrap a `FieldTimeSeries` or `Function` to indicate that a tracer should be prescribed
(not time-stepped) in the model. Prescribed tracers are available to turbulence closures
for buoyancy computation but are not advanced by the time-stepper.

Supported input types:
- `FieldTimeSeries`: interpolated to the current model time via `TimeSeriesInterpolation`
- `Function(x, y, z, t)` or `Function(x, y, z, t, parameters)`: wrapped in a `FunctionField`

Examples
========

Prescribe buoyancy with a `Function`:

```jldoctest prescribed
julia> using Oceananigans

julia> using Oceananigans.Models.HydrostaticFreeSurfaceModels: PrescribedTracer

julia> b(x, y, z, t) = 1e-7 * x + 1e-5 * z;

julia> PrescribedTracer(b)
PrescribedTracer wrapping b (generic function with 1 method)
```

Prescribe temperature and salinity with `FieldTimeSeries`:

```jldoctest prescribed
julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1e5, 1e5, 1e3));

julia> T = FieldTimeSeries{Center, Center, Center}(grid, [0.0, 86400.0]);

julia> S = FieldTimeSeries{Center, Center, Center}(grid, [0.0, 86400.0]);

julia> PrescribedTracer(T)
PrescribedTracer wrapping 4×4×4×2 FieldTimeSeries{InMemory} located at (Center, Center, Center) on CPU
```
"""
PrescribedTracer(field; parameters=nothing) = PrescribedTracer(field, parameters)

Base.summary(pt::PrescribedTracer) = "PrescribedTracer wrapping $(summary(pt.field))"
Base.show(io::IO, pt::PrescribedTracer) = print(io, summary(pt))

#####
##### Materialization: convert PrescribedTracer to a concrete field type
#####

materialize_prescribed_tracer(pt::PrescribedTracer{<:FieldTimeSeries}, grid; clock) =
    TimeSeriesInterpolation(pt.field, grid; clock)

materialize_prescribed_tracer(pt::PrescribedTracer{<:Function}, grid; clock) =
    FunctionField{Center, Center, Center}(pt.field, grid; clock, parameters=pt.parameters)

# Fallback: if already a Field or similar, just return it
materialize_prescribed_tracer(pt::PrescribedTracer, grid; clock) = pt.field

#####
##### Type predicates for prescribed tracers
#####

"""
    is_prescribed_tracer(field)

Return `true` if `field` is a prescribed (non-prognostic) tracer in `model.tracers`.
Prescribed tracers are `TimeSeriesInterpolation`, `FunctionField`, or
`GPUAdaptedTimeSeriesInterpolation` objects.
"""
is_prescribed_tracer(::TimeSeriesInterpolation) = true
is_prescribed_tracer(::FunctionField) = true
is_prescribed_tracer(::Any) = false

# GPU-adapted variant
is_prescribed_tracer(::GPUAdaptedTimeSeriesInterpolation) = true

#####
##### Helpers for separating prescribed and prognostic tracers
#####

"""
    prognostic_tracer_names(tracers::NamedTuple)

Return a tuple of tracer names that are prognostic (not prescribed).
"""
function prognostic_tracer_names(tracers::NamedTuple)
    names = propertynames(tracers)
    return Tuple(n for n in names if !is_prescribed_tracer(tracers[n]))
end

"""
    prognostic_tracers(tracers::NamedTuple)

Return a NamedTuple containing only prognostic (non-prescribed) tracer fields.
"""
function prognostic_tracers(tracers::NamedTuple)
    prog_names = prognostic_tracer_names(tracers)
    isempty(prog_names) && return NamedTuple()
    return NamedTuple{prog_names}(Tuple(tracers[n] for n in prog_names))
end

#####
##### Materialize tracer fields: handle mixed prescribed and prognostic tracers
#####

using Oceananigans.Fields: TracerFields

"""
    materialize_tracer_fields(tracers, grid, clock, boundary_conditions)

Create tracer fields from a tracer specification that may contain `PrescribedTracer` entries.
Regular tracers become `CenterField`s via `TracerFields`; prescribed tracers are materialized
into `TimeSeriesInterpolation` or `FunctionField` objects.
"""
function materialize_tracer_fields(tracers::NamedTuple, grid, clock, boundary_conditions)
    all_names = propertynames(tracers)

    # Check if any entries are PrescribedTracer
    has_prescribed = any(tracers[n] isa PrescribedTracer for n in all_names)
    has_prescribed || return TracerFields(tracers, grid, boundary_conditions)

    # Separate prescribed and prognostic tracer entries
    fields = map(all_names) do name
        entry = tracers[name]
        if entry isa PrescribedTracer
            materialize_prescribed_tracer(entry, grid; clock)
        else
            # entry is either a Field (user-provided) or a CenterField from closure tracers
            entry isa Field ? entry : CenterField(grid, boundary_conditions=boundary_conditions[name])
        end
    end

    return NamedTuple{all_names}(Tuple(fields))
end

# Fallback: when tracers is a Tuple of Symbols (no prescribed tracers possible)
materialize_tracer_fields(tracers::Tuple, grid, clock, boundary_conditions) =
    TracerFields(tracers, grid, boundary_conditions)

#####
##### Boundary condition handling for prescribed tracers
#####

extract_boundary_conditions(::PrescribedTracer) = FieldBoundaryConditions()
extract_boundary_conditions(::TimeSeriesInterpolation) = FieldBoundaryConditions()
extract_boundary_conditions(::FunctionField) = FieldBoundaryConditions()

# Note: fill_halo_regions! no-ops for TimeSeriesInterpolation and FunctionField
# are already defined in prescribed_hydrostatic_velocity_fields.jl
