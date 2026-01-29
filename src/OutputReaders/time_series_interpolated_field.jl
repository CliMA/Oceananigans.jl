using Adapt
using Oceananigans.Architectures: on_architecture
using Oceananigans.Fields: AbstractField, indices, show_location
using Oceananigans.Units: Time

#####
##### TimeSeriesInterpolatedField
#####
##### A wrapper around FieldTimeSeries that interpolates to the current clock time
#####

struct TimeSeriesInterpolatedField{LX, LY, LZ, FTS, C, G, T} <: AbstractField{LX, LY, LZ, G, T, 3}
    time_series :: FTS
    grid :: G
    clock :: C

    @doc """
        TimeSeriesInterpolatedField{LX, LY, LZ}(time_series, grid; clock) where {LX, LY, LZ}

    Returns a `TimeSeriesInterpolatedField` at location `LX, LY, LZ` that wraps a `FieldTimeSeries`
    and interpolates to the current `clock.time` when indexed.

    When indexed at `i, j, k`, returns `time_series[i, j, k, Time(clock.time)]`, providing
    time-interpolated values from the underlying `FieldTimeSeries`.
    """
    function TimeSeriesInterpolatedField{LX, LY, LZ}(time_series::FTS,
                                                      grid::G;
                                                      clock::C) where {LX, LY, LZ, FTS, G, C}
        T = eltype(grid)
        return new{LX, LY, LZ, FTS, C, G, T}(time_series, grid, clock)
    end
end

# Convenience constructor with location as positional tuple argument
@inline TimeSeriesInterpolatedField(L::Tuple, time_series, grid; clock) =
    TimeSeriesInterpolatedField{L[1], L[2], L[3]}(time_series, grid; clock)

@inline indices(::TimeSeriesInterpolatedField) = (:, :, :)

#####
##### getindex: interpolate to current clock time
#####

@inline function Base.getindex(f::TimeSeriesInterpolatedField, i, j, k)
    return @inbounds f.time_series[i, j, k, Time(f.clock.time)]
end

#####
##### GPU adaptation
#####
##### For GPU kernels, we adapt the underlying FieldTimeSeries to GPUAdaptedFieldTimeSeries
##### and store the clock time directly (since clock cannot be used on GPU).
#####

struct GPUAdaptedTimeSeriesInterpolatedField{LX, LY, LZ, FTS, TT, T} <: AbstractField{LX, LY, LZ, Nothing, T, 3}
    time_series :: FTS  # GPUAdaptedFieldTimeSeries
    time :: TT          # Current clock time (scalar value)

    function GPUAdaptedTimeSeriesInterpolatedField{LX, LY, LZ}(time_series::FTS,
                                                                time::TT) where {LX, LY, LZ, FTS, TT}
        T = eltype(time_series)
        return new{LX, LY, LZ, FTS, TT, T}(time_series, time)
    end
end

@inline indices(::GPUAdaptedTimeSeriesInterpolatedField) = (:, :, :)

@inline function Base.getindex(f::GPUAdaptedTimeSeriesInterpolatedField, i, j, k)
    return @inbounds f.time_series[i, j, k, Time(f.time)]
end

function Adapt.adapt_structure(to, f::TimeSeriesInterpolatedField{LX, LY, LZ}) where {LX, LY, LZ}
    adapted_time_series = Adapt.adapt(to, f.time_series)
    return GPUAdaptedTimeSeriesInterpolatedField{LX, LY, LZ}(adapted_time_series, f.clock.time)
end

#####
##### on_architecture
#####

function on_architecture(to, f::TimeSeriesInterpolatedField{LX, LY, LZ}) where {LX, LY, LZ}
    return TimeSeriesInterpolatedField{LX, LY, LZ}(on_architecture(to, f.time_series),
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
          "└── time: $(field.time)")
end
