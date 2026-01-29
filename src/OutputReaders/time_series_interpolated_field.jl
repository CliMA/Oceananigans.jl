using Adapt
using Oceananigans: location
using Oceananigans.Architectures: on_architecture
using Oceananigans.Fields: AbstractField, indices, show_location
using Oceananigans.Grids: size as grid_size
using Oceananigans.Units: Time

import Oceananigans.Fields: indices

#####
##### TimeSeriesInterpolatedField
#####
##### A wrapper around FieldTimeSeries that interpolates to the current clock time
#####

struct TimeSeriesInterpolatedField{LX, LY, LZ, FTS, C, G, T} <: AbstractField{LX, LY, LZ, G, T, 3}
    time_series :: FTS
    grid :: G
    clock :: C
end

"""
    TimeSeriesInterpolatedField(time_series, grid; clock)

Returns a `TimeSeriesInterpolatedField` that wraps a `FieldTimeSeries`
and interpolates to the current `clock.time` when indexed.

The location is obtained from the `FieldTimeSeries`.

When indexed at `i, j, k`, returns `time_series[i, j, k, Time(clock.time)]`, providing
time-interpolated values from the underlying `FieldTimeSeries`.
"""
function TimeSeriesInterpolatedField(time_series::FTS, grid::G; clock::C) where {FTS, G, C}
    LX, LY, LZ = location(time_series)
    T = eltype(grid)
    return TimeSeriesInterpolatedField{LX, LY, LZ, FTS, C, G, T}(time_series, grid, clock)
end

# Use indices from the underlying FieldTimeSeries
@inline indices(f::TimeSeriesInterpolatedField) = indices(f.time_series)

# Override size to account for reduced indices
@inline Base.size(f::TimeSeriesInterpolatedField) = grid_size(f.grid, location(f), indices(f))

#####
##### getindex: interpolate to current clock time
#####

@inline Base.getindex(f::TimeSeriesInterpolatedField, i, j, k) =
    @inbounds f.time_series[i, j, k, Time(f.clock.time)]

#####
##### GPU adaptation
#####
##### For GPU kernels, we adapt the underlying FieldTimeSeries to GPUAdaptedFieldTimeSeries
##### and store the clock time directly (since clock cannot be used on GPU).
##### We also store indices since GPUAdaptedFieldTimeSeries doesn't preserve them.
#####

struct GPUAdaptedTimeSeriesInterpolatedField{LX, LY, LZ, FTS, TT, I, T} <: AbstractField{LX, LY, LZ, Nothing, T, 3}
    time_series :: FTS  # GPUAdaptedFieldTimeSeries
    time :: TT          # Current clock time (scalar value)
    indices :: I        # Spatial indices from the original FieldTimeSeries

    function GPUAdaptedTimeSeriesInterpolatedField{LX, LY, LZ}(time_series::FTS,
                                                                time::TT,
                                                                indices::I) where {LX, LY, LZ, FTS, TT, I}
        T = eltype(time_series)
        return new{LX, LY, LZ, FTS, TT, I, T}(time_series, time, indices)
    end
end

@inline indices(f::GPUAdaptedTimeSeriesInterpolatedField) = f.indices

@inline Base.getindex(f::GPUAdaptedTimeSeriesInterpolatedField, i, j, k) =
    @inbounds f.time_series[i, j, k, Time(f.time)]

function Adapt.adapt_structure(to, f::TimeSeriesInterpolatedField{LX, LY, LZ}) where {LX, LY, LZ}
    adapted_time_series = Adapt.adapt(to, f.time_series)
    return GPUAdaptedTimeSeriesInterpolatedField{LX, LY, LZ}(adapted_time_series, f.clock.time, indices(f))
end

#####
##### on_architecture
#####

function on_architecture(to, f::TimeSeriesInterpolatedField)
    return TimeSeriesInterpolatedField(on_architecture(to, f.time_series),
                                       on_architecture(to, f.grid);
                                       clock = on_architecture(to, f.clock))
end

#####
##### Show method
#####

function Base.show(io::IO, field::TimeSeriesInterpolatedField)
    print(io, "TimeSeriesInterpolatedField located at ", show_location(field), "\n",
          "├── time_series: $(summary(field.time_series))", "\n",
          "├── grid: $(summary(field.grid))\n",
          "└── clock: $(summary(field.clock))")
end

function Base.show(io::IO, field::GPUAdaptedTimeSeriesInterpolatedField)
    print(io, "GPUAdaptedTimeSeriesInterpolatedField located at ", show_location(field), "\n",
          "├── time_series: $(summary(field.time_series))", "\n",
          "├── time: $(field.time)\n",
          "└── indices: $(field.indices)")
end
