using Adapt
using Oceananigans: location
using Oceananigans.Architectures: on_architecture
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Fields: indices, show_location
using Oceananigans.Grids: size as grid_size
using Oceananigans.Units: Time

import Oceananigans.Fields: indices

#####
##### TimeSeriesInterpolation
#####
##### An AbstractOperation that wraps a FieldTimeSeries and interpolates to the current clock time
#####

struct TimeSeriesInterpolation{LX, LY, LZ, FTS, C, G, T} <: AbstractOperation{LX, LY, LZ, G, T}
    time_series :: FTS
    grid :: G
    clock :: C
end

"""
    TimeSeriesInterpolation(time_series, grid; clock)

Returns a `TimeSeriesInterpolation` that wraps a `FieldTimeSeries`
and interpolates to the current `clock.time` when indexed.

The location is obtained from the `FieldTimeSeries`.

When indexed at `i, j, k`, returns `time_series[i, j, k, Time(clock.time)]`, providing
time-interpolated values from the underlying `FieldTimeSeries`.

Since `TimeSeriesInterpolation <: AbstractOperation`, it can be used with the `Field`
constructor to compute and store values for output.
"""
function TimeSeriesInterpolation(time_series::FTS, grid::G; clock::C) where {FTS, G, C}
    LX, LY, LZ = location(time_series)
    T = eltype(grid)
    return TimeSeriesInterpolation{LX, LY, LZ, FTS, C, G, T}(time_series, grid, clock)
end

# Use indices from the underlying FieldTimeSeries
@inline indices(f::TimeSeriesInterpolation) = indices(f.time_series)

# Override size to account for reduced indices
@inline Base.size(f::TimeSeriesInterpolation) = grid_size(f.grid, location(f), indices(f))

#####
##### getindex: interpolate to current clock time
#####

@inline Base.getindex(f::TimeSeriesInterpolation, i, j, k) =
    @inbounds f.time_series[i, j, k, Time(f.clock.time)]

#####
##### GPU adaptation
#####
##### For GPU kernels, we adapt the underlying FieldTimeSeries to GPUAdaptedFieldTimeSeries
##### and store the clock time directly (since clock cannot be used on GPU).
##### We also store indices since GPUAdaptedFieldTimeSeries doesn't preserve them.
#####

struct GPUAdaptedTimeSeriesInterpolation{LX, LY, LZ, FTS, TT, I, T} <: AbstractOperation{LX, LY, LZ, Nothing, T}
    time_series :: FTS  # GPUAdaptedFieldTimeSeries
    time :: TT          # Current clock time (scalar value)
    indices :: I        # Spatial indices from the original FieldTimeSeries
end

@inline indices(f::GPUAdaptedTimeSeriesInterpolation) = f.indices

@inline Base.getindex(f::GPUAdaptedTimeSeriesInterpolation, i, j, k) =
    @inbounds f.time_series[i, j, k, Time(f.time)]

function Adapt.adapt_structure(to, f::TimeSeriesInterpolation{LX, LY, LZ}) where {LX, LY, LZ}
    adapted_time_series = Adapt.adapt(to, f.time_series)
    T = eltype(f)
    return GPUAdaptedTimeSeriesInterpolation{LX, LY, LZ, typeof(adapted_time_series), typeof(f.clock.time), typeof(indices(f)), T}(
        adapted_time_series, f.clock.time, indices(f))
end

#####
##### on_architecture
#####

function on_architecture(to, f::TimeSeriesInterpolation)
    return TimeSeriesInterpolation(on_architecture(to, f.time_series),
                                   on_architecture(to, f.grid);
                                   clock = on_architecture(to, f.clock))
end

#####
##### Show method
#####

function Base.show(io::IO, op::TimeSeriesInterpolation)
    print(io, "TimeSeriesInterpolation located at ", show_location(op), "\n",
          "├── time_series: $(summary(op.time_series))", "\n",
          "├── grid: $(summary(op.grid))\n",
          "└── clock: $(summary(op.clock))")
end

function Base.show(io::IO, op::GPUAdaptedTimeSeriesInterpolation)
    print(io, "GPUAdaptedTimeSeriesInterpolation located at ", show_location(op), "\n",
          "├── time_series: $(summary(op.time_series))", "\n",
          "├── time: $(op.time)\n",
          "└── indices: $(op.indices)")
end
