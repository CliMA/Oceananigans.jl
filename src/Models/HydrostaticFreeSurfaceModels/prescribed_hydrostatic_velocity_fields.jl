#####
##### PrescribedVelocityFields
#####

using Oceananigans: location
using Oceananigans.Grids: Center, Face, halo_size
using Oceananigans.Fields: FunctionField, field, ZeroField, OneField, ConstantField
using Oceananigans.TimeSteppers: tick!, step_lagrangian_particles!
using Oceananigans.BoundaryConditions: BoundaryConditions, fill_halo_regions!
using Oceananigans.OutputReaders: FieldTimeSeries, TimeSeriesInterpolation

import Oceananigans: prognostic_state, restore_prognostic_state!
import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.Models: extract_boundary_conditions
import Oceananigans.Utils: datatuple, sum_of_velocities
import Oceananigans.TimeSteppers: time_step!

struct PrescribedVelocityFields{U, V, W, P}
    u :: U
    v :: V
    w :: W
    parameters :: P
end

@inline Base.getindex(U::PrescribedVelocityFields, i) = getindex((u=U.u, v=U.v, w=U.w), i)

"""
    PrescribedVelocityFields(; u = ZeroField(),
                               v = ZeroField(),
                               w = ZeroField(),
                               parameters = nothing)

Builds `PrescribedVelocityFields` with prescribed functions `u`, `v`, and `w`.

If `isnothing(parameters)`, then `u, v, w` are called with the signature

```
u(x, y, z, t) = # something interesting
```

If `!isnothing(parameters)`, then `u, v, w` are called with the signature

```
u(x, y, z, t, parameters) = # something parameterized and interesting
```

In the constructor for `HydrostaticFreeSurfaceModel`, the functions `u, v, w` are wrapped
in `FunctionField` and associated with the model's `grid` and `clock`.
"""
function PrescribedVelocityFields(; u = ZeroField(),
                                    v = ZeroField(),
                                    w = ZeroField(),
                                    parameters = nothing)

    return PrescribedVelocityFields(u, v, w, parameters)
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

materialize_prescribed_velocity(X, Y, Z, tsi::TimeSeriesInterpolation, grid; clock, kwargs...) =
    materialize_prescribed_velocity(X, Y, Z, tsi.time_series, grid; clock, kwargs...)

@inline prescribed_operation_contains_raw_field_time_series(::FieldTimeSeries) = true
@inline prescribed_operation_contains_raw_field_time_series(::TimeSeriesInterpolation) = false
@inline prescribed_operation_contains_raw_field_time_series(source::Tuple) =
    any(prescribed_operation_contains_raw_field_time_series(s) for s in source)
@inline prescribed_operation_contains_raw_field_time_series(source::NamedTuple) =
    any(prescribed_operation_contains_raw_field_time_series(s) for s in values(source))
@inline prescribed_operation_contains_raw_field_time_series(source::Oceananigans.AbstractOperations.AbstractOperation) =
    any(prescribed_operation_contains_raw_field_time_series(getproperty(source, property)) for property in propertynames(source))
@inline prescribed_operation_contains_raw_field_time_series(source) = false

@inline prescribed_operation_contains_any_time_series(::FieldTimeSeries) = true
@inline prescribed_operation_contains_any_time_series(::TimeSeriesInterpolation) = true
@inline prescribed_operation_contains_any_time_series(source::Tuple) =
    any(prescribed_operation_contains_any_time_series(s) for s in source)
@inline prescribed_operation_contains_any_time_series(source::NamedTuple) =
    any(prescribed_operation_contains_any_time_series(s) for s in values(source))
@inline prescribed_operation_contains_any_time_series(source::Oceananigans.AbstractOperations.AbstractOperation) =
    any(prescribed_operation_contains_any_time_series(getproperty(source, property)) for property in propertynames(source))
@inline prescribed_operation_contains_any_time_series(source) = false

@inline rebind_prescribed_operation_clock(source, clock) = source
@inline function rebind_prescribed_operation_clock(source::FieldTimeSeries, clock)
    isnothing(clock) && return source
    return TimeSeriesInterpolation(source, source.grid; clock)
end
@inline function rebind_prescribed_operation_clock(source::TimeSeriesInterpolation, clock)
    isnothing(clock) && return source
    source.clock === clock && return source
    return TimeSeriesInterpolation(source.time_series, source.grid; clock)
end
@inline rebind_prescribed_operation_clock(source::Tuple, clock) =
    map(s -> rebind_prescribed_operation_clock(s, clock), source)
@inline rebind_prescribed_operation_clock(source::NamedTuple, clock) =
    NamedTuple{keys(source)}(Tuple(rebind_prescribed_operation_clock(value, clock) for value in values(source)))
@inline function rebind_prescribed_operation_clock(source::Oceananigans.AbstractOperations.UnaryOperation{LX, LY, LZ}, clock) where {LX, LY, LZ}
    return Oceananigans.AbstractOperations.UnaryOperation{LX, LY, LZ}(source.op,
                                                                      rebind_prescribed_operation_clock(source.arg, clock),
                                                                      source.▶,
                                                                      source.grid,
                                                                      eltype(source))
end
@inline function rebind_prescribed_operation_clock(source::Oceananigans.AbstractOperations.BinaryOperation{LX, LY, LZ}, clock) where {LX, LY, LZ}
    return Oceananigans.AbstractOperations.BinaryOperation{LX, LY, LZ}(source.op,
                                                                       rebind_prescribed_operation_clock(source.a, clock),
                                                                       rebind_prescribed_operation_clock(source.b, clock),
                                                                       source.▶a,
                                                                       source.▶b,
                                                                       source.grid,
                                                                       eltype(source))
end
@inline function rebind_prescribed_operation_clock(source::Oceananigans.AbstractOperations.MultiaryOperation{LX, LY, LZ}, clock) where {LX, LY, LZ}
    return Oceananigans.AbstractOperations.MultiaryOperation{LX, LY, LZ}(source.op,
                                                                         rebind_prescribed_operation_clock(source.args, clock),
                                                                         source.▶,
                                                                         source.grid,
                                                                         eltype(source))
end
@inline function rebind_prescribed_operation_clock(source::Oceananigans.AbstractOperations.ConditionalOperation{LX, LY, LZ}, clock) where {LX, LY, LZ}
    return Oceananigans.AbstractOperations.ConditionalOperation{LX, LY, LZ}(rebind_prescribed_operation_clock(source.operand, clock),
                                                                            source.func,
                                                                            source.grid,
                                                                            rebind_prescribed_operation_clock(source.condition, clock),
                                                                            source.mask)
end

@inline function materialize_same_grid_prescribed_operation_velocity(candidate,
                                                                     source::Oceananigans.AbstractOperations.AbstractOperation,
                                                                     grid;
                                                                     clock = nothing)
    source.grid === grid ||
        throw(ArgumentError("Prescribed velocity AbstractOperations must be defined on the model grid."))
    location(source) == location(candidate) ||
        throw(ArgumentError("Prescribed velocity AbstractOperations must match the requested velocity location."))
    prescribed_operation_contains_raw_field_time_series(source) &&
    isnothing(clock) &&
        throw(ArgumentError("Prescribed velocity AbstractOperations containing raw FieldTimeSeries are unsupported. Wrap them in TimeSeriesInterpolation or materialize them first."))

    rebound_source = rebind_prescribed_operation_clock(source, clock)
    return Oceananigans.Fields.Field(rebound_source; boundary_conditions = candidate.boundary_conditions)
end

materialize_prescribed_velocity(::Type{Face}, ::Type{Center}, ::Type{Center},
                                source::Oceananigans.AbstractOperations.AbstractOperation, grid; clock = nothing, kwargs...) =
    materialize_same_grid_prescribed_operation_velocity(Oceananigans.Fields.XFaceField(grid), source, grid; clock)

materialize_prescribed_velocity(::Type{Center}, ::Type{Face}, ::Type{Center},
                                source::Oceananigans.AbstractOperations.AbstractOperation, grid; clock = nothing, kwargs...) =
    materialize_same_grid_prescribed_operation_velocity(Oceananigans.Fields.YFaceField(grid), source, grid; clock)

materialize_prescribed_velocity(::Type{Center}, ::Type{Center}, ::Type{Face},
                                source::Oceananigans.AbstractOperations.AbstractOperation, grid; clock = nothing, kwargs...) =
    materialize_same_grid_prescribed_operation_velocity(Oceananigans.Fields.ZFaceField(grid), source, grid; clock)

materialize_prescribed_velocity(X, Y, Z, f, grid; kwargs...) = field((X, Y, Z), f, grid)

@inline has_quadfolded_vector_boundary_conditions(field::Field) =
    Oceananigans.Fields.uses_quadfolded_vector_boundary_conditions(field)

@inline function has_quadfolded_vector_boundary_conditions(fts::FieldTimeSeries)
    bcs = fts.boundary_conditions
    return bcs.west isa Union{BoundaryConditions.QCovZBC, BoundaryConditions.QConZBC} ||
           bcs.east isa Union{BoundaryConditions.QCovZBC, BoundaryConditions.QConZBC} ||
           bcs.south isa Union{BoundaryConditions.QCovZBC, BoundaryConditions.QConZBC} ||
           bcs.north isa Union{BoundaryConditions.QCovZBC, BoundaryConditions.QConZBC}
end

@inline has_quadfolded_vector_boundary_conditions(tsi::TimeSeriesInterpolation) =
    has_quadfolded_vector_boundary_conditions(tsi.time_series)

@inline has_quadfolded_vector_boundary_conditions(::Any) = false

@inline prescribed_velocity_boundary_conditions(source::Field) = source.boundary_conditions
@inline prescribed_velocity_boundary_conditions(source::FieldTimeSeries) = source.boundary_conditions
@inline prescribed_velocity_boundary_conditions(source::TimeSeriesInterpolation) = source.time_series.boundary_conditions

@inline prescribed_velocity_source_grid(::Union{Oceananigans.Fields.ZeroField,
                                                Oceananigans.Fields.OneField,
                                                Oceananigans.Fields.ConstantField}, args...) =
    prescribed_velocity_source_grid(args...)
@inline prescribed_velocity_source_grid(source::Field, args...) = source.grid
@inline prescribed_velocity_source_grid(source::FieldTimeSeries, args...) = source.grid
@inline prescribed_velocity_source_grid(source::TimeSeriesInterpolation, args...) = source.time_series.grid
@inline prescribed_velocity_source_grid(_, args...) = prescribed_velocity_source_grid(args...)
prescribed_velocity_source_grid() =
    throw(ArgumentError("OctaHEALPix prescribed velocity materialization requires at least one grid-carrying source when inferring a source grid."))

@inline prescribed_velocity_time_series(source::FieldTimeSeries) = source
@inline prescribed_velocity_time_series(source::TimeSeriesInterpolation) = source.time_series
@inline prescribed_velocity_time_series(::Any) = nothing

@inline prescribed_velocity_snapshot(source::Field, n) = source
@inline prescribed_velocity_snapshot(source::FieldTimeSeries, n) = source[n]
@inline prescribed_velocity_snapshot(source::TimeSeriesInterpolation, n) = source.time_series[n]

@inline single_component_quadfolded_horizontal_time_series_source(source) =
    source isa Union{FieldTimeSeries, TimeSeriesInterpolation} &&
    has_quadfolded_vector_boundary_conditions(source)

@inline single_component_quadfolded_u_source(source) =
    source isa Union{Field{<:Face, <:Center, <:Center},
                     FieldTimeSeries{<:Face, <:Center, <:Center},
                     TimeSeriesInterpolation{<:Face, <:Center, <:Center}} &&
    has_quadfolded_vector_boundary_conditions(source)

@inline single_component_quadfolded_v_source(source) =
    source isa Union{Field{<:Center, <:Face, <:Center},
                     FieldTimeSeries{<:Center, <:Face, <:Center},
                     TimeSeriesInterpolation{<:Center, <:Face, <:Center}} &&
    has_quadfolded_vector_boundary_conditions(source)

@inline function supported_same_grid_prescribed_operation_source(candidate, source, grid; clock = nothing)
    if !(source isa Oceananigans.AbstractOperations.AbstractOperation) || source isa TimeSeriesInterpolation
        return false
    end

    source.grid === grid || return false
    location(source) == location(candidate) || return false
    return !prescribed_operation_contains_raw_field_time_series(source) || !isnothing(clock)
end

@inline static_prescribed_operation_source(source) =
    source isa Oceananigans.AbstractOperations.AbstractOperation &&
    !(source isa TimeSeriesInterpolation) &&
    !prescribed_operation_contains_raw_field_time_series(source) &&
    !prescribed_operation_contains_any_time_series(source)

@inline function unsupported_quadfolded_horizontal_operation_source(candidate, source, grid; clock = nothing)
    return source isa Oceananigans.AbstractOperations.AbstractOperation &&
           !(source isa TimeSeriesInterpolation) &&
           source.grid isa Oceananigans.Grids.SphericalShellGrid &&
           source.grid.connectivity isa Oceananigans.Grids.OctaHEALPixConnectivity &&
           !supported_same_grid_prescribed_operation_source(candidate, source, grid; clock)
end

@inline unsupported_adapted_time_series_source(source) =
    source isa Union{Oceananigans.OutputReaders.GPUAdaptedFieldTimeSeries,
                     Oceananigans.OutputReaders.GPUAdaptedTimeSeriesInterpolation}

@inline function equivalent_octahealpix_vector_grids(grid_u, grid_v)
    return typeof(grid_u) === typeof(grid_v) &&
           Oceananigans.Grids.architecture(grid_u) === Oceananigans.Grids.architecture(grid_v) &&
           grid_u.Nx == grid_v.Nx &&
           grid_u.Ny == grid_v.Ny &&
           grid_u.Nz == grid_v.Nz &&
           halo_size(grid_u) == halo_size(grid_v) &&
           typeof(grid_u.connectivity) === typeof(grid_v.connectivity)
end

@inline requires_prescribed_quadfolded_horizontal_remapping(candidate, source) =
    source isa Field &&
    has_quadfolded_vector_boundary_conditions(source) &&
    (source.grid !== candidate.grid || !Oceananigans.Fields.matching_field_discretization(candidate, source))

@inline function requires_prescribed_quadfolded_horizontal_remapping(candidate, source::FieldTimeSeries)
    source_snapshot = source[first(eachindex(source.times))]
    return has_quadfolded_vector_boundary_conditions(source) &&
           (source.grid !== candidate.grid || !Oceananigans.Fields.matching_field_discretization(candidate, source_snapshot))
end

@inline requires_prescribed_quadfolded_horizontal_remapping(candidate, source::TimeSeriesInterpolation) =
    requires_prescribed_quadfolded_horizontal_remapping(candidate, source.time_series)

@inline paired_quadfolded_horizontal_velocity_sources(source_u, source_v) =
    (source_u isa Union{Field{<:Face, <:Center, <:Center}, FieldTimeSeries{<:Face, <:Center, <:Center}, TimeSeriesInterpolation{<:Face, <:Center, <:Center}}) &&
    (source_v isa Union{Field{<:Center, <:Face, <:Center}, FieldTimeSeries{<:Center, <:Face, <:Center}, TimeSeriesInterpolation{<:Center, <:Face, <:Center}}) &&
    has_quadfolded_vector_boundary_conditions(source_u) &&
    has_quadfolded_vector_boundary_conditions(source_v)

function materialize_prescribed_paired_horizontal_velocity_time_series(source_u::Union{Field, FieldTimeSeries, TimeSeriesInterpolation},
                                                                       source_v::Union{Field, FieldTimeSeries, TimeSeriesInterpolation},
                                                                       grid; clock)
    source_u_grid = prescribed_velocity_source_grid(source_u)
    source_v_grid = prescribed_velocity_source_grid(source_v)
    (source_u_grid === source_v_grid || equivalent_octahealpix_vector_grids(source_u_grid, source_v_grid)) ||
        throw(ArgumentError("Paired OctaHEALPix time-dependent prescribed velocities must share a source grid."))

    source_u_fts = prescribed_velocity_time_series(source_u)
    source_v_fts = prescribed_velocity_time_series(source_v)
    source_times = isnothing(source_u_fts) ? source_v_fts.times : source_u_fts.times
    source_time_indexing = isnothing(source_u_fts) ? source_v_fts.time_indexing : source_u_fts.time_indexing

    if !isnothing(source_u_fts) && !isnothing(source_v_fts)
        source_u_fts.times == source_v_fts.times ||
            throw(ArgumentError("Paired OctaHEALPix FieldTimeSeries prescribed velocities must share the same times."))
        source_u_fts.time_indexing == source_v_fts.time_indexing ||
            throw(ArgumentError("Paired OctaHEALPix FieldTimeSeries prescribed velocities must share the same time indexing."))
    end

    target_u_fts = FieldTimeSeries{Face, Center, Center}(grid, source_times;
                                                         boundary_conditions = prescribed_velocity_boundary_conditions(source_u),
                                                         time_indexing = source_time_indexing)
    target_v_fts = FieldTimeSeries{Center, Face, Center}(grid, source_times;
                                                         boundary_conditions = prescribed_velocity_boundary_conditions(source_v),
                                                         time_indexing = source_time_indexing)

    for n in eachindex(source_times)
        target_u = target_u_fts[n]
        target_v = target_v_fts[n]
        source_u_n = prescribed_velocity_snapshot(source_u, n)
        source_v_n = prescribed_velocity_snapshot(source_v, n)

        Oceananigans.Fields.set_paired_quadfolded_vector_fields!(target_u, target_v, source_u_n, source_v_n)
        fill_halo_regions!((target_u, target_v))
    end

    return TimeSeriesInterpolation(target_u_fts, grid; clock),
           TimeSeriesInterpolation(target_v_fts, grid; clock)
end

@inline time_dependent_prescribed_operation_source(source) =
    source isa Oceananigans.AbstractOperations.AbstractOperation &&
    !(source isa TimeSeriesInterpolation) &&
    prescribed_operation_contains_any_time_series(source)

@inline function prescribed_operation_time_series_sources(sources...)
    extracted = Tuple(Oceananigans.OutputReaders.extract_field_time_series(source) for source in sources)
    flattened = Oceananigans.Fields.flattened_unique_values(extracted)
    return Tuple(fts for fts in flattened if !isnothing(fts))
end

function prescribed_operation_source_times_and_indexing(sources...)
    source_time_series = prescribed_operation_time_series_sources(sources...)
    isempty(source_time_series) &&
        throw(ArgumentError("Dynamic prescribed AbstractOperations require at least one TimeSeriesInterpolation source."))

    source_times = source_time_series[1].times
    source_time_indexing = source_time_series[1].time_indexing

    for source_fts in source_time_series
        source_fts.times == source_times ||
            throw(ArgumentError("Dynamic OctaHEALPix prescribed AbstractOperations must share the same times."))
        source_fts.time_indexing == source_time_indexing ||
            throw(ArgumentError("Dynamic OctaHEALPix prescribed AbstractOperations must share the same time indexing."))
    end

    return source_times, source_time_indexing
end

function materialize_prescribed_paired_horizontal_velocity_operation_time_series(source_u,
                                                                                 source_v,
                                                                                 source_u_grid,
                                                                                 source_v_grid,
                                                                                 grid; clock)
    (source_u_grid === source_v_grid || equivalent_octahealpix_vector_grids(source_u_grid, source_v_grid)) ||
        throw(ArgumentError("Dynamic OctaHEALPix prescribed AbstractOperations must share a source grid."))

    source_times, source_time_indexing = prescribed_operation_source_times_and_indexing(source_u, source_v)
    source_clock = Oceananigans.Clock(time = first(source_times))

    source_u_field = source_u isa Oceananigans.AbstractOperations.AbstractOperation ?
                     materialize_same_grid_prescribed_operation_velocity(Oceananigans.Fields.XFaceField(source_u_grid), source_u, source_u_grid; clock = source_clock) :
                     source_u
    source_v_field = source_v isa Oceananigans.AbstractOperations.AbstractOperation ?
                     materialize_same_grid_prescribed_operation_velocity(Oceananigans.Fields.YFaceField(source_v_grid), source_v, source_v_grid; clock = source_clock) :
                     source_v

    source_u_bcs = source_u isa Oceananigans.AbstractOperations.AbstractOperation ?
                   source_u_field.boundary_conditions :
                   prescribed_velocity_boundary_conditions(source_u)
    source_v_bcs = source_v isa Oceananigans.AbstractOperations.AbstractOperation ?
                   source_v_field.boundary_conditions :
                   prescribed_velocity_boundary_conditions(source_v)

    target_u_fts = FieldTimeSeries{Face, Center, Center}(grid, source_times;
                                                         boundary_conditions = source_u_bcs,
                                                         time_indexing = source_time_indexing)
    target_v_fts = FieldTimeSeries{Center, Face, Center}(grid, source_times;
                                                         boundary_conditions = source_v_bcs,
                                                         time_indexing = source_time_indexing)

    for n in eachindex(source_times)
        source_clock.time = source_times[n]

        target_u = target_u_fts[n]
        target_v = target_v_fts[n]

        source_u_n = source_u isa Oceananigans.AbstractOperations.AbstractOperation ? source_u_field : prescribed_velocity_snapshot(source_u, n)
        source_v_n = source_v isa Oceananigans.AbstractOperations.AbstractOperation ? source_v_field : prescribed_velocity_snapshot(source_v, n)

        update_prescribed_horizontal_velocity_field_operations!(source_u_n, source_v_n)
        Oceananigans.Fields.set_paired_quadfolded_vector_fields!(target_u, target_v, source_u_n, source_v_n)
        fill_halo_regions!((target_u, target_v))
    end

    return TimeSeriesInterpolation(target_u_fts, grid; clock),
           TimeSeriesInterpolation(target_v_fts, grid; clock)
end

function materialize_prescribed_vertical_operation_time_series(source_w, source_grid, grid; clock)
    source_times, source_time_indexing = prescribed_operation_source_times_and_indexing(source_w)
    source_clock = Oceananigans.Clock(time = first(source_times))
    source_w_field = materialize_same_grid_prescribed_operation_velocity(Oceananigans.Fields.ZFaceField(source_grid), source_w, source_grid; clock = source_clock)

    target_w_fts = FieldTimeSeries{Center, Center, Face}(grid, source_times;
                                                         boundary_conditions = source_w_field.boundary_conditions,
                                                         time_indexing = source_time_indexing)

    for n in eachindex(source_times)
        source_clock.time = source_times[n]
        Oceananigans.Fields.compute!(source_w_field)

        target_w = target_w_fts[n]
        Oceananigans.Fields.set!(target_w, source_w_field)
        fill_halo_regions!(target_w)
    end

    return TimeSeriesInterpolation(target_w_fts, grid; clock)
end

function materialize_prescribed_horizontal_velocities(velocities::PrescribedVelocityFields, grid; clock, parameters)
    source_u = velocities.u
    source_v = velocities.v
    octahealpix_grid =
        grid isa Oceananigans.Grids.SphericalShellGrid &&
        grid.connectivity isa Oceananigans.Grids.OctaHEALPixConnectivity

    if octahealpix_grid &&
       (unsupported_adapted_time_series_source(source_u) ||
        unsupported_adapted_time_series_source(source_v))
        msg = string("Prescribing OctaHEALPix horizontal velocities from adapted GPU time-series wrappers is unsupported.", '\n',
                     "Use unadapted FieldTimeSeries or TimeSeriesInterpolation sources when constructing the model.")
        throw(ArgumentError(msg))
    end

    candidate_u = Oceananigans.Fields.XFaceField(grid)
    candidate_v = Oceananigans.Fields.YFaceField(grid)

    same_grid_u_operation = supported_same_grid_prescribed_operation_source(candidate_u, source_u, grid; clock)
    same_grid_v_operation = supported_same_grid_prescribed_operation_source(candidate_v, source_v, grid; clock)
    static_u_operation = static_prescribed_operation_source(source_u)
    static_v_operation = static_prescribed_operation_source(source_v)
    dynamic_u_operation = time_dependent_prescribed_operation_source(source_u)
    dynamic_v_operation = time_dependent_prescribed_operation_source(source_v)
    paired_static_operation_sources =
        static_u_operation &&
        static_v_operation &&
        equivalent_octahealpix_vector_grids(source_u.grid, source_v.grid)
    paired_dynamic_operation_sources =
        dynamic_u_operation &&
        dynamic_v_operation &&
        equivalent_octahealpix_vector_grids(source_u.grid, source_v.grid)

    if paired_static_operation_sources &&
       (source_u.grid !== grid || source_v.grid !== grid)
        source_u_field = materialize_same_grid_prescribed_operation_velocity(Oceananigans.Fields.XFaceField(source_u.grid), source_u, source_u.grid)
        source_v_field = materialize_same_grid_prescribed_operation_velocity(Oceananigans.Fields.YFaceField(source_v.grid), source_v, source_v.grid)
        update_prescribed_horizontal_velocity_field_operations!(source_u_field, source_v_field)

        u = Oceananigans.Fields.XFaceField(grid; boundary_conditions = source_u_field.boundary_conditions)
        v = Oceananigans.Fields.YFaceField(grid; boundary_conditions = source_v_field.boundary_conditions)

        Oceananigans.Fields.set_paired_quadfolded_vector_fields!(u, v, source_u_field, source_v_field)
        fill_halo_regions!((u, v))

        return u, v
    end

    if paired_dynamic_operation_sources &&
       (source_u.grid !== grid || source_v.grid !== grid)
        return materialize_prescribed_paired_horizontal_velocity_operation_time_series(source_u,
                                                                                       source_v,
                                                                                       source_u.grid,
                                                                                       source_v.grid,
                                                                                       grid; clock)
    end

    if static_u_operation &&
       source_v isa Union{ZeroField, Oceananigans.Fields.OneField, Oceananigans.Fields.ConstantField} &&
       source_u.grid !== grid
        source_u_field = materialize_same_grid_prescribed_operation_velocity(Oceananigans.Fields.XFaceField(source_u.grid), source_u, source_u.grid)
        source_v_companion = Oceananigans.Fields.YFaceField(source_u.grid)
        set!(source_v_companion, source_v)
        update_prescribed_horizontal_velocity_field_operations!(source_u_field, source_v_companion)

        u = Oceananigans.Fields.XFaceField(grid; boundary_conditions = source_u_field.boundary_conditions)
        v = Oceananigans.Fields.YFaceField(grid; boundary_conditions = source_v_companion.boundary_conditions)

        Oceananigans.Fields.set_paired_quadfolded_vector_fields!(u, v, source_u_field, source_v_companion)
        fill_halo_regions!((u, v))

        return u, v
    end

    if dynamic_u_operation &&
       source_v isa Union{ZeroField, Oceananigans.Fields.OneField, Oceananigans.Fields.ConstantField} &&
       source_u.grid !== grid
        source_v_companion = Oceananigans.Fields.YFaceField(source_u.grid)
        set!(source_v_companion, source_v)
        return materialize_prescribed_paired_horizontal_velocity_operation_time_series(source_u,
                                                                                       source_v_companion,
                                                                                       source_u.grid,
                                                                                       source_u.grid,
                                                                                       grid; clock)
    end

    if static_v_operation &&
       source_u isa Union{ZeroField, Oceananigans.Fields.OneField, Oceananigans.Fields.ConstantField} &&
       source_v.grid !== grid
        source_u_companion = Oceananigans.Fields.XFaceField(source_v.grid)
        source_v_field = materialize_same_grid_prescribed_operation_velocity(Oceananigans.Fields.YFaceField(source_v.grid), source_v, source_v.grid)
        set!(source_u_companion, source_u)
        update_prescribed_horizontal_velocity_field_operations!(source_u_companion, source_v_field)

        u = Oceananigans.Fields.XFaceField(grid; boundary_conditions = source_u_companion.boundary_conditions)
        v = Oceananigans.Fields.YFaceField(grid; boundary_conditions = source_v_field.boundary_conditions)

        Oceananigans.Fields.set_paired_quadfolded_vector_fields!(u, v, source_u_companion, source_v_field)
        fill_halo_regions!((u, v))

        return u, v
    end

    if dynamic_v_operation &&
       source_u isa Union{ZeroField, Oceananigans.Fields.OneField, Oceananigans.Fields.ConstantField} &&
       source_v.grid !== grid
        source_u_companion = Oceananigans.Fields.XFaceField(source_v.grid)
        set!(source_u_companion, source_u)
        return materialize_prescribed_paired_horizontal_velocity_operation_time_series(source_u_companion,
                                                                                       source_v,
                                                                                       source_v.grid,
                                                                                       source_v.grid,
                                                                                       grid; clock)
    end

    if static_u_operation &&
       source_u.grid !== grid &&
       source_v isa Union{Field{<:Center, <:Face, <:Center},
                          FieldTimeSeries{<:Center, <:Face, <:Center},
                          TimeSeriesInterpolation{<:Center, <:Face, <:Center}} &&
       has_quadfolded_vector_boundary_conditions(source_v) &&
       equivalent_octahealpix_vector_grids(source_u.grid, prescribed_velocity_source_grid(source_v))
        source_u_field = materialize_same_grid_prescribed_operation_velocity(Oceananigans.Fields.XFaceField(source_u.grid), source_u, source_u.grid)

        if !isnothing(prescribed_velocity_time_series(source_v))
            return materialize_prescribed_paired_horizontal_velocity_operation_time_series(source_u_field,
                                                                                           source_v,
                                                                                           source_u.grid,
                                                                                           prescribed_velocity_source_grid(source_v),
                                                                                           grid; clock)
        end

        u = Oceananigans.Fields.XFaceField(grid; boundary_conditions = source_u_field.boundary_conditions)
        v = Oceananigans.Fields.YFaceField(grid; boundary_conditions = prescribed_velocity_boundary_conditions(source_v))
        update_prescribed_horizontal_velocity_field_operations!(source_u_field, source_v)

        Oceananigans.Fields.set_paired_quadfolded_vector_fields!(u, v, source_u_field, source_v)
        fill_halo_regions!((u, v))

        return u, v
    end

    if dynamic_u_operation &&
       source_u.grid !== grid &&
       source_v isa Union{Field{<:Center, <:Face, <:Center},
                          FieldTimeSeries{<:Center, <:Face, <:Center},
                          TimeSeriesInterpolation{<:Center, <:Face, <:Center}} &&
       has_quadfolded_vector_boundary_conditions(source_v) &&
       equivalent_octahealpix_vector_grids(source_u.grid, prescribed_velocity_source_grid(source_v))
        return materialize_prescribed_paired_horizontal_velocity_operation_time_series(source_u,
                                                                                       source_v,
                                                                                       source_u.grid,
                                                                                       prescribed_velocity_source_grid(source_v),
                                                                                       grid; clock)
    end

    if static_v_operation &&
       source_v.grid !== grid &&
       source_u isa Union{Field{<:Face, <:Center, <:Center},
                          FieldTimeSeries{<:Face, <:Center, <:Center},
                          TimeSeriesInterpolation{<:Face, <:Center, <:Center}} &&
       has_quadfolded_vector_boundary_conditions(source_u) &&
       equivalent_octahealpix_vector_grids(source_v.grid, prescribed_velocity_source_grid(source_u))
        source_v_field = materialize_same_grid_prescribed_operation_velocity(Oceananigans.Fields.YFaceField(source_v.grid), source_v, source_v.grid)

        if !isnothing(prescribed_velocity_time_series(source_u))
            return materialize_prescribed_paired_horizontal_velocity_operation_time_series(source_u,
                                                                                           source_v_field,
                                                                                           prescribed_velocity_source_grid(source_u),
                                                                                           source_v.grid,
                                                                                           grid; clock)
        end

        u = Oceananigans.Fields.XFaceField(grid; boundary_conditions = prescribed_velocity_boundary_conditions(source_u))
        v = Oceananigans.Fields.YFaceField(grid; boundary_conditions = source_v_field.boundary_conditions)
        update_prescribed_horizontal_velocity_field_operations!(source_u, source_v_field)

        Oceananigans.Fields.set_paired_quadfolded_vector_fields!(u, v, source_u, source_v_field)
        fill_halo_regions!((u, v))

        return u, v
    end

    if dynamic_v_operation &&
       source_v.grid !== grid &&
       source_u isa Union{Field{<:Face, <:Center, <:Center},
                          FieldTimeSeries{<:Face, <:Center, <:Center},
                          TimeSeriesInterpolation{<:Face, <:Center, <:Center}} &&
       has_quadfolded_vector_boundary_conditions(source_u) &&
       equivalent_octahealpix_vector_grids(source_v.grid, prescribed_velocity_source_grid(source_u))
        return materialize_prescribed_paired_horizontal_velocity_operation_time_series(source_u,
                                                                                       source_v,
                                                                                       prescribed_velocity_source_grid(source_u),
                                                                                       source_v.grid,
                                                                                       grid; clock)
    end

    if unsupported_quadfolded_horizontal_operation_source(candidate_u, source_u, grid; clock) ||
       unsupported_quadfolded_horizontal_operation_source(candidate_v, source_v, grid; clock)
        msg = string("Prescribing OctaHEALPix horizontal velocities from generic AbstractOperations is unsupported.", '\n',
                     "Use same-grid static operations, cross-grid static or dynamic paired operations, or model-clock TimeSeriesInterpolation operations, or materialize paired (u, v) as Fields, FieldTimeSeries, or TimeSeriesInterpolation instead.")
        throw(ArgumentError(msg))
    end

    u_requires_remapping = requires_prescribed_quadfolded_horizontal_remapping(candidate_u, source_u)
    v_requires_remapping = requires_prescribed_quadfolded_horizontal_remapping(candidate_v, source_v)
    paired_quadfolded_sources = paired_quadfolded_horizontal_velocity_sources(source_u, source_v)

    if same_grid_u_operation &&
       source_v isa Union{ZeroField, Oceananigans.Fields.OneField, Oceananigans.Fields.ConstantField} &&
       has_quadfolded_vector_boundary_conditions(candidate_u)
        u = materialize_same_grid_prescribed_operation_velocity(candidate_u, source_u, grid; clock)
        v = Oceananigans.Fields.YFaceField(grid; boundary_conditions = candidate_v.boundary_conditions)
        set!(v, source_v)
        return u, v
    end

    if same_grid_v_operation &&
       source_u isa Union{ZeroField, Oceananigans.Fields.OneField, Oceananigans.Fields.ConstantField} &&
       has_quadfolded_vector_boundary_conditions(candidate_v)
        u = Oceananigans.Fields.XFaceField(grid; boundary_conditions = candidate_u.boundary_conditions)
        set!(u, source_u)
        v = materialize_same_grid_prescribed_operation_velocity(candidate_v, source_v, grid; clock)
        return u, v
    end

    if single_component_quadfolded_u_source(source_u) &&
       source_v isa Union{ZeroField, Oceananigans.Fields.OneField, Oceananigans.Fields.ConstantField} &&
       (prescribed_velocity_source_grid(source_u) === grid ||
        requires_prescribed_quadfolded_horizontal_remapping(candidate_u, source_u) ||
        prescribed_velocity_time_series(source_u) !== nothing)
        source_v_companion = Oceananigans.Fields.YFaceField(prescribed_velocity_source_grid(source_u))
        set!(source_v_companion, source_v)

        if isnothing(prescribed_velocity_time_series(source_u))
            u = Oceananigans.Fields.XFaceField(grid; boundary_conditions = prescribed_velocity_boundary_conditions(source_u))
            v = Oceananigans.Fields.YFaceField(grid; boundary_conditions = source_v_companion.boundary_conditions)

            Oceananigans.Fields.set_paired_quadfolded_vector_fields!(u, v, source_u, source_v_companion)
            fill_halo_regions!((u, v))

            return u, v
        end

        return materialize_prescribed_paired_horizontal_velocity_time_series(source_u, source_v_companion, grid; clock)
    end

    if single_component_quadfolded_v_source(source_v) &&
       source_u isa Union{ZeroField, Oceananigans.Fields.OneField, Oceananigans.Fields.ConstantField} &&
       (prescribed_velocity_source_grid(source_v) === grid ||
        requires_prescribed_quadfolded_horizontal_remapping(candidate_v, source_v) ||
        prescribed_velocity_time_series(source_v) !== nothing)
        source_u_companion = Oceananigans.Fields.XFaceField(prescribed_velocity_source_grid(source_v))
        set!(source_u_companion, source_u)

        if isnothing(prescribed_velocity_time_series(source_v))
            u = Oceananigans.Fields.XFaceField(grid; boundary_conditions = source_u_companion.boundary_conditions)
            v = Oceananigans.Fields.YFaceField(grid; boundary_conditions = prescribed_velocity_boundary_conditions(source_v))

            Oceananigans.Fields.set_paired_quadfolded_vector_fields!(u, v, source_u_companion, source_v)
            fill_halo_regions!((u, v))

            return u, v
        end

        return materialize_prescribed_paired_horizontal_velocity_time_series(source_u_companion, source_v, grid; clock)
    end

    if (u_requires_remapping || v_requires_remapping) && !paired_quadfolded_sources
        msg = string("Interpolating OctaHEALPix prescribed velocities one component at a time is unsupported.", '\n',
                     "Prescribe paired (u, v) fields together instead.")
        throw(ArgumentError(msg))
    end

    paired_time_dependent_sources =
        paired_quadfolded_sources &&
        (prescribed_velocity_time_series(source_u) !== nothing ||
         prescribed_velocity_time_series(source_v) !== nothing)

    if paired_time_dependent_sources ||
       (paired_quadfolded_sources && (u_requires_remapping || v_requires_remapping))
        if paired_time_dependent_sources
            return materialize_prescribed_paired_horizontal_velocity_time_series(source_u, source_v, grid; clock)
        end

        u = Oceananigans.Fields.XFaceField(grid; boundary_conditions = prescribed_velocity_boundary_conditions(source_u))
        v = Oceananigans.Fields.YFaceField(grid; boundary_conditions = prescribed_velocity_boundary_conditions(source_v))
        Oceananigans.Fields.set_paired_quadfolded_vector_fields!(u, v, source_u, source_v)
        return u, v
    end

    u = same_grid_u_operation ? materialize_same_grid_prescribed_operation_velocity(candidate_u, source_u, grid; clock) :
                                materialize_prescribed_velocity(Face, Center, Center, source_u, grid; clock, parameters)
    v = same_grid_v_operation ? materialize_same_grid_prescribed_operation_velocity(candidate_v, source_v, grid; clock) :
                                materialize_prescribed_velocity(Center, Face, Center, source_v, grid; clock, parameters)

    return u, v
end

function materialize_prescribed_vertical_velocity(w::Field{<:Center, <:Center, <:Face}, grid; clock, parameters)
    materialized_w = Oceananigans.Fields.ZFaceField(grid; boundary_conditions = w.boundary_conditions)
    Oceananigans.Fields.set!(materialized_w, w)
    return materialized_w
end

function materialize_prescribed_vertical_velocity(w::FieldTimeSeries{<:Center, <:Center, <:Face}, grid; clock, parameters)
    candidate_w = Oceananigans.Fields.ZFaceField(grid; boundary_conditions = w.boundary_conditions)
    source_snapshot = w[first(eachindex(w.times))]
    octahealpix_scalar_halos =
        grid isa Oceananigans.Grids.SphericalShellGrid &&
        grid.connectivity isa Oceananigans.Grids.OctaHEALPixConnectivity

    if !octahealpix_scalar_halos &&
       w.grid === grid &&
       Oceananigans.Fields.matching_field_discretization(candidate_w, source_snapshot)
        return TimeSeriesInterpolation(w, grid; clock)
    end

    target_w_fts = FieldTimeSeries{Center, Center, Face}(grid, w.times;
                                                         boundary_conditions = w.boundary_conditions,
                                                         time_indexing = w.time_indexing)
    Oceananigans.Fields.interpolate!(target_w_fts, w)

    return TimeSeriesInterpolation(target_w_fts, grid; clock)
end

materialize_prescribed_vertical_velocity(w::TimeSeriesInterpolation{<:Center, <:Center, <:Face}, grid; clock, parameters) =
    materialize_prescribed_vertical_velocity(w.time_series, grid; clock, parameters)

function materialize_prescribed_vertical_velocity(w::Oceananigans.AbstractOperations.AbstractOperation, grid; clock, parameters)
    if w.grid isa Oceananigans.Grids.SphericalShellGrid &&
       w.grid.connectivity isa Oceananigans.Grids.OctaHEALPixConnectivity
        candidate_w = Oceananigans.Fields.ZFaceField(grid)
        if supported_same_grid_prescribed_operation_source(candidate_w, w, grid; clock)
            return materialize_same_grid_prescribed_operation_velocity(candidate_w, w, grid; clock)
        end

        if static_prescribed_operation_source(w) && w.grid !== grid
            source_w = materialize_same_grid_prescribed_operation_velocity(Oceananigans.Fields.ZFaceField(w.grid), w, w.grid)
            return materialize_prescribed_vertical_velocity(source_w, grid; clock, parameters)
        end

        if time_dependent_prescribed_operation_source(w) && w.grid !== grid
            return materialize_prescribed_vertical_operation_time_series(w, w.grid, grid; clock)
        end

        msg = string("Prescribing OctaHEALPix vertical velocity from generic AbstractOperations is unsupported.", '\n',
                     "Use a same-grid static operation, cross-grid static or dynamic operation, or model-clock TimeSeriesInterpolation operation, or materialize `w` as a Field, FieldTimeSeries, or TimeSeriesInterpolation instead.")
        throw(ArgumentError(msg))
    end

    return materialize_prescribed_velocity(Center, Center, Face, w, grid; clock, parameters)
end

@inline update_prescribed_velocity_field_operation!(source) = nothing

@inline function update_prescribed_velocity_field_operation!(field::Oceananigans.Fields.Field)
    isnothing(field.operand) || Oceananigans.Fields.compute!(field)
    return nothing
end

@inline prescribed_velocity_field_operation_is_live(::Any) = false

@inline prescribed_velocity_field_operation_is_live(field::Oceananigans.Fields.Field) =
    !isnothing(field.operand)

@inline refresh_prescribed_horizontal_velocity_field_halos!(u, v) = nothing

@inline function refresh_prescribed_horizontal_velocity_field_halos!(
    u::Oceananigans.Fields.Field{Face, Center, LZ},
    v::Oceananigans.Fields.Field{Center, Face, LZ},
) where LZ
    if prescribed_velocity_field_operation_is_live(u) || prescribed_velocity_field_operation_is_live(v)
        fill_halo_regions!((u, v))
    end

    return nothing
end

@inline function refresh_prescribed_horizontal_velocity_field_halos!(
    u::Oceananigans.Fields.Field{Face, Center, LZ},
    v::Union{ZeroField, OneField, ConstantField},
) where LZ
    if prescribed_velocity_field_operation_is_live(u)
        companion_v = Oceananigans.Fields.quadfolded_companion_field(u)
        set!(companion_v, v)
        fill_halo_regions!((u, companion_v))
    end

    return nothing
end

@inline function refresh_prescribed_horizontal_velocity_field_halos!(
    u::Union{ZeroField, OneField, ConstantField},
    v::Oceananigans.Fields.Field{Center, Face, LZ},
) where LZ
    if prescribed_velocity_field_operation_is_live(v)
        companion_u = Oceananigans.Fields.quadfolded_companion_field(v)
        set!(companion_u, u)
        fill_halo_regions!((companion_u, v))
    end

    return nothing
end

@inline function update_prescribed_horizontal_velocity_field_operations!(u, v)
    update_prescribed_velocity_field_operation!(u)
    update_prescribed_velocity_field_operation!(v)
    refresh_prescribed_horizontal_velocity_field_halos!(u, v)
    return nothing
end

function update_prescribed_horizontal_velocity_field_operations!(
    u::Oceananigans.AbstractOperations.ComputedField{Face, Center, LZ},
    v::Oceananigans.AbstractOperations.ComputedField{Center, Face, LZ},
) where LZ
    if !isnothing(u.operand) && !isnothing(v.operand)
        Oceananigans.Fields.compute!((u, v))
    else
        isnothing(u.operand) || Oceananigans.Fields.compute!(u)
        isnothing(v.operand) || Oceananigans.Fields.compute!(v)
        refresh_prescribed_horizontal_velocity_field_halos!(u, v)
    end

    return nothing
end

function update_prescribed_velocity_field_operations!(velocities::PrescribedVelocityFields)
    update_prescribed_horizontal_velocity_field_operations!(velocities.u, velocities.v)
    update_prescribed_velocity_field_operation!(velocities.w)
    return nothing
end

@inline update_prescribed_velocity_field_operations!(velocities) = nothing

function materialize_prescribed_vertical_velocity(w::Union{Oceananigans.OutputReaders.GPUAdaptedFieldTimeSeries,
                                                           Oceananigans.OutputReaders.GPUAdaptedTimeSeriesInterpolation},
                                                  grid; clock, parameters)
    if grid isa Oceananigans.Grids.SphericalShellGrid &&
       grid.connectivity isa Oceananigans.Grids.OctaHEALPixConnectivity
        msg = string("Prescribing OctaHEALPix vertical velocity from adapted GPU time-series wrappers is unsupported.", '\n',
                     "Use unadapted FieldTimeSeries or TimeSeriesInterpolation sources when constructing the model.")
        throw(ArgumentError(msg))
    end

    return materialize_prescribed_velocity(Center, Center, Face, w, grid; clock, parameters)
end

materialize_prescribed_vertical_velocity(w, grid; clock, parameters) =
    materialize_prescribed_velocity(Center, Center, Face, w, grid; clock, parameters)

function hydrostatic_velocity_fields(velocities::PrescribedVelocityFields, grid, clock, bcs)

    parameters = velocities.parameters
    u, v = materialize_prescribed_horizontal_velocities(velocities, grid; clock, parameters)
    w = materialize_prescribed_vertical_velocity(velocities.w, grid; clock, parameters)

    fill_halo_regions!((u, v))
    fill_halo_regions!(w)

    return PrescribedVelocityFields(u, v, w, parameters)
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

@inline BoundaryConditions.fill_halo_regions!(velocities::PrescribedVelocityFields, args...; kwargs...) =
    BoundaryConditions.fill_halo_regions!((velocities.u, velocities.v, velocities.w), args...; kwargs...)
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
mask_immersed_horizontal_velocities!(::PrescribedVelocityFields) = nothing

# Prescribed velocities need separate transport fields on non-orthogonal grids.
prescribed_transport_velocity_field(u::Field{<:Face, <:Center, <:Center}, grid, ::Val{:u}) =
    Oceananigans.Fields.XFaceField(grid;
                                   boundary_conditions = transport_velocity_boundary_conditions(u.boundary_conditions))
prescribed_transport_velocity_field(v::Field{<:Center, <:Face, <:Center}, grid, ::Val{:v}) =
    Oceananigans.Fields.YFaceField(grid;
                                   boundary_conditions = transport_velocity_boundary_conditions(v.boundary_conditions))
prescribed_transport_velocity_field(w::Field{<:Center, <:Center, <:Face}, grid, ::Val{:w}) =
    Oceananigans.Fields.ZFaceField(grid;
                                   boundary_conditions = w.boundary_conditions)
prescribed_transport_velocity_field(u::FieldTimeSeries{<:Face, <:Center, <:Center}, grid, ::Val{:u}) =
    Oceananigans.Fields.XFaceField(grid;
                                   boundary_conditions = transport_velocity_boundary_conditions(u.boundary_conditions))
prescribed_transport_velocity_field(v::FieldTimeSeries{<:Center, <:Face, <:Center}, grid, ::Val{:v}) =
    Oceananigans.Fields.YFaceField(grid;
                                   boundary_conditions = transport_velocity_boundary_conditions(v.boundary_conditions))
prescribed_transport_velocity_field(w::FieldTimeSeries{<:Center, <:Center, <:Face}, grid, ::Val{:w}) =
    Oceananigans.Fields.ZFaceField(grid;
                                   boundary_conditions = w.boundary_conditions)
prescribed_transport_velocity_field(u::TimeSeriesInterpolation{<:Face, <:Center, <:Center}, grid, ::Val{:u}) =
    Oceananigans.Fields.XFaceField(grid;
                                   boundary_conditions = transport_velocity_boundary_conditions(u.time_series.boundary_conditions))
prescribed_transport_velocity_field(v::TimeSeriesInterpolation{<:Center, <:Face, <:Center}, grid, ::Val{:v}) =
    Oceananigans.Fields.YFaceField(grid;
                                   boundary_conditions = transport_velocity_boundary_conditions(v.time_series.boundary_conditions))
prescribed_transport_velocity_field(w::TimeSeriesInterpolation{<:Center, <:Center, <:Face}, grid, ::Val{:w}) =
    Oceananigans.Fields.ZFaceField(grid;
                                   boundary_conditions = w.time_series.boundary_conditions)

prescribed_transport_velocity_field(u, grid, ::Val{:u}) =
    Oceananigans.Fields.XFaceField(grid;
                                   boundary_conditions = transport_velocity_boundary_conditions(
                                       Oceananigans.Fields.XFaceField(grid).boundary_conditions))
prescribed_transport_velocity_field(v, grid, ::Val{:v}) =
    Oceananigans.Fields.YFaceField(grid;
                                   boundary_conditions = transport_velocity_boundary_conditions(
                                       Oceananigans.Fields.YFaceField(grid).boundary_conditions))
prescribed_transport_velocity_field(w, grid, ::Val{:w}) = copy_velocity(Oceananigans.Fields.ZFaceField(grid))

prescribed_velocity_grid(::Union{Oceananigans.Fields.ZeroField,
                                 Oceananigans.Fields.OneField,
                                 Oceananigans.Fields.ConstantField}, args...) = prescribed_velocity_grid(args...)
prescribed_velocity_grid(u::Oceananigans.Fields.AbstractField, args...) = u.grid
prescribed_velocity_grid(u::TimeSeriesInterpolation, args...) = u.grid
prescribed_velocity_grid(u::Oceananigans.AbstractOperations.AbstractOperation, args...) = u.grid
prescribed_velocity_grid(_, args...) = prescribed_velocity_grid(args...)
prescribed_velocity_grid() =
    throw(ArgumentError("transport_velocity_fields(::PrescribedVelocityFields) requires at least one materialized field component. Use transport_velocity_fields(velocities, grid) for raw prescribed functions or constants."))

transport_velocity_fields(velocities::PrescribedVelocityFields) =
    transport_velocity_fields(velocities, prescribed_velocity_grid(velocities.u, velocities.v, velocities.w))

@inline octahealpix_prescribed_operation_transport_source(source) =
    source isa Oceananigans.AbstractOperations.AbstractOperation && !(source isa TimeSeriesInterpolation)

function octahealpix_prescribed_transport_materialization_clock(grid, sources...)
    source_time_series = prescribed_operation_time_series_sources(sources...)
    isempty(source_time_series) && return Oceananigans.Clock(time = zero(eltype(grid)))
    return Oceananigans.Clock(time = first(source_time_series[1].times))
end

function transport_velocity_fields(velocities::PrescribedVelocityFields, grid)
    octahealpix_grid =
        grid isa Oceananigans.Grids.SphericalShellGrid &&
        grid.connectivity isa Oceananigans.Grids.OctaHEALPixConnectivity

    if octahealpix_grid &&
       (unsupported_adapted_time_series_source(velocities.u) ||
        unsupported_adapted_time_series_source(velocities.v) ||
        unsupported_adapted_time_series_source(velocities.w))
        msg = string("Constructing OctaHEALPix transport velocities from adapted GPU time-series wrappers is unsupported.", '\n',
                     "Use unadapted Fields, FieldTimeSeries, or TimeSeriesInterpolation sources instead.")
        throw(ArgumentError(msg))
    end

    if octahealpix_grid &&
       (octahealpix_prescribed_operation_transport_source(velocities.u) ||
        octahealpix_prescribed_operation_transport_source(velocities.v) ||
        octahealpix_prescribed_operation_transport_source(velocities.w))
        materialization_clock = octahealpix_prescribed_transport_materialization_clock(grid,
                                                                                       velocities.u,
                                                                                       velocities.v,
                                                                                       velocities.w)
        materialized_velocities = hydrostatic_velocity_fields(velocities, grid, materialization_clock, NamedTuple())
        return transport_velocity_fields(materialized_velocities, grid)
    end

    return (u = prescribed_transport_velocity_field(velocities.u, grid, Val(:u)),
            v = prescribed_transport_velocity_field(velocities.v, grid, Val(:v)),
            w = prescribed_transport_velocity_field(velocities.w, grid, Val(:w)))
end

validate_velocity_boundary_conditions(grid, ::PrescribedVelocityFields) = nothing
extract_boundary_conditions(::PrescribedVelocityFields) = NamedTuple()

free_surface_displacement_field(::PrescribedVelocityFields, ::Nothing, grid) = nothing
HorizontalVelocityFields(::PrescribedVelocityFields, grid) = nothing, nothing

materialize_free_surface(::Nothing,                      ::PrescribedVelocityFields, grid) = nothing
materialize_free_surface(::ExplicitFreeSurface{Nothing}, ::PrescribedVelocityFields, grid) = nothing
materialize_free_surface(::ImplicitFreeSurface{Nothing}, ::PrescribedVelocityFields, grid) = nothing
materialize_free_surface(::SplitExplicitFreeSurface,     ::PrescribedVelocityFields, grid) = nothing

hydrostatic_prognostic_fields(::PrescribedVelocityFields, ::Nothing, tracers) = tracers
compute_hydrostatic_momentum_tendencies!(model, ::PrescribedVelocityFields, kernel_parameters; kwargs...) = nothing

compute_flux_bcs!(::Nothing, c, arch, clock, model_fields) = nothing

adapt_prescribed_velocity_source(to, source::FieldTimeSeries) = source
adapt_prescribed_velocity_source(to, source::TimeSeriesInterpolation) = source
adapt_prescribed_velocity_source(to, source) = Adapt.adapt(to, source)

Adapt.adapt_structure(to, velocities::PrescribedVelocityFields) =
    PrescribedVelocityFields(adapt_prescribed_velocity_source(to, velocities.u),
                             adapt_prescribed_velocity_source(to, velocities.v),
                             adapt_prescribed_velocity_source(to, velocities.w),
                             Adapt.adapt(to, velocities.parameters))

on_architecture(to, velocities::PrescribedVelocityFields) =
    PrescribedVelocityFields(on_architecture(to, velocities.u),
                             on_architecture(to, velocities.v),
                             on_architecture(to, velocities.w),
                             on_architecture(to, velocities.parameters))

# If the model only tracks particles... do nothing but that!!!
const OnlyParticleTrackingModel = HydrostaticFreeSurfaceModel{TS, E, A, S, RL, G, T, V, B, R, F, P, BGC, U, W, C} where
                 {TS, E, A, S, RL, G, T, V, B, R, F, P<:AbstractLagrangianParticles, BGC, U<:PrescribedVelocityFields, W, C<:NamedTuple{(), Tuple{}}}

function time_step!(model::OnlyParticleTrackingModel, Δt; callbacks = [], kwargs...)
    tick!(model.clock, Δt)
    step_lagrangian_particles!(model, Δt)
    update_state!(model, callbacks)
end

update_state!(model::OnlyParticleTrackingModel, callbacks) =
    [callback(model) for callback in callbacks if callback.callsite isa UpdateStateCallsite]

#####
##### Checkpointing
#####

prognostic_state(::PrescribedVelocityFields) = nothing
restore_prognostic_state!(::PrescribedVelocityFields, ::Nothing) = nothing
