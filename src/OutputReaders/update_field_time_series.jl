import Oceananigans.BoundaryConditions: BoundaryCondition, getbc
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Fields: AbstractField

# Termination (move all here when we switch the code up)
extract_field_timeseries(f::FieldTimeSeries) = f

const CPUFTSBC = BoundaryCondition{<:Any, <:FieldTimeSeries}
const GPUFTSBC = BoundaryCondition{<:Any, <:GPUAdaptedFieldTimeSeries}

const FTSBC = Union{CPUFTSBC, GPUFTSBC}

@inline getbc(bc::FTSBC, i::Int, j::Int, grid::AbstractGrid, clock, args...) = bc.condition[i, j, Time(clock.time)]

set!(::TotallyInMemoryFieldTimeSeries, index_range) = nothing

# Set a field with a range of time indices.
# We change the index range of the `FieldTimeSeries`
# and load the new data
function set!(fts::InMemoryFieldTimeSeries, index_range::UnitRange)
    # TODO: only load new data by comparing current and new index range?
    fts.backend.index_range = index_range
    set!(fts, fts.path, fts.name)
    return nothing
end

# fallback
update_field_time_series!(::Nothing, time) = nothing
update_field_time_series!(::TotallyInMemoryFieldTimeSeries, ::Int64) = nothing
update_field_time_series!(::TotallyInMemoryFieldTimeSeries, ::Time) = nothing

const CyclicalInMemoryFTS = InMemoryFieldTimeSeries{<:Any, <:Any, <:Any, <:Cyclical}
const LinearInMemoryFTS = InMemoryFieldTimeSeries{<:Any, <:Any, <:Any, <:Linear}
const ClampInMemoryFTS  = InMemoryFieldTimeSeries{<:Any, <:Any, <:Any, <:Clamp}

# Update the `fts` to contain the time `time_index.time`.
# Linear extrapolation, simple version
function update_field_time_series!(fts::InMemoryFieldTimeSeries, time_index::Time)
    time = time_index.time
    correct_time = corrected_time(time, fts.times[1], fts.times[end], fts.time_extrapolation)
    n₁, n₂ = index_binary_search(fts.times, correct_time, length(fts.times))
    update_field_time_series!(fts, n₂)
    return nothing
end

@inline corrected_time(t, t¹, tᴺ, ::Clamp) = 
    ifelse(t > tᴺ, tᴺ,  # Beyond last time: clamp to last
    ifelse(t < t¹, t¹,  # Before first time: clamp to first
           t))          # business as usual

@inline corrected_time(t, t¹, tᴺ, ::Linear) = t # Fallback for linear extrapolation (no need to handle boundaries)

@inline function corrected_time(t, t¹, tᴺ, ::Cyclical) 
    ΔT  = tᴺ - t¹ # time range
    Δt⁺ = t  - tᴺ # excess time
    Δt⁻ = t¹ - t  # time defect

    Δtᴺ = tᴺ - times[end-1]
    Δt¹ = times[2] - t¹

    # To interpolate inbetween tᴺ and t¹ we assume that:
    # - tᴺ corresponds to 2t¹ - t²
    # - t¹ corresponds to 2tᴺ - tᴺ⁻¹
    cycled_t = ifelse(t > tᴺ, t¹ - Δt¹ + mod(Δt⁺, ΔT), # Beyond last time: circle around
               ifelse(t < t¹, tᴺ + Δtᴺ - mod(Δt⁻, ΔT), # Before first time: circle around
                      t))                              # business as usual

    return cycled_t
end

# Update `fts` to contain the time index `n`.
# update rules are the following: 
# if `n` is 1, load the first `length(fts.backend.index_range)` time steps
# if `n` is within the last `length(fts.backend.index_range)` time steps, load the last `length(fts.backend.index_range)` time steps
# otherwise `n` will be placed at index `[:, :, :, 2]` of `fts.data`
function update_field_time_series!(fts::InMemoryFieldTimeSeries, n::Int)
    if !(n ∈ fts.backend.index_range)
        Nt = length(fts.times)
        Ni = length(fts.backend.index_range)
        if n == 1
            set!(fts, 1:Ni)
        elseif n > Nt - Ni
            set!(fts, Nt-Ni+1:Nt)
        else
            set!(fts, n-1:n+Ni-2)
        end
    end

    return nothing
end

# Utility used to extract field time series from a type through recursion
function extract_field_timeseries(t) 
    prop = propertynames(t)
    if isempty(prop)
        return nothing
    end

    return Tuple(extract_field_timeseries(getproperty(t, p)) for p in prop)
end

# For types that do not contain `FieldTimeSeries`, halt the recursion
NonFTS = [:Number, :AbstractArray]

for NonFTSType in NonFTS
    @eval extract_field_timeseries(::$NonFTSType) = nothing
end

# Special recursion rules for `Tuple` and `Field` types
extract_field_timeseries(t::AbstractField)     = Tuple(extract_field_timeseries(getproperty(t, p)) for p in propertynames(t))
extract_field_timeseries(t::AbstractOperation) = Tuple(extract_field_timeseries(getproperty(t, p)) for p in propertynames(t))
extract_field_timeseries(t::Tuple)             = Tuple(extract_field_timeseries(n) for n in t)
extract_field_timeseries(t::NamedTuple)        = Tuple(extract_field_timeseries(n) for n in t)
