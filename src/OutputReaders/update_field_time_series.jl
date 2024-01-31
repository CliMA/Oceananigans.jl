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
function set!(fts::InMemoryFieldTimeSeries, index_range::Tuple)
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
    t = time_index.time
    n, n₁, n₂ = interpolating_time_indices(fts, t)
    update_field_time_series!(fts, n₁, n₂)
    return nothing
end

# Update `fts` to contain the time index `n`.
# update rules are the following: 
# if `n` is 1, load the first `length(fts.backend.index_range)` time steps
# if `n` is within the last `length(fts.backend.index_range)` time steps, load the last `length(fts.backend.index_range)` time steps
# otherwise `n` will be placed at index `[:, :, :, 2]` of `fts.data`
function update_field_time_series!(fts::InMemoryFieldTimeSeries, n₁, n₂)

    in_range = n₁ ∈ fts.backend.index_range &&
               n₂ ∈ fts.backend.index_range

    if !in_range # out of range
        Nt = length(fts.times)
        Ni = length(fts.backend.index_range)

        te = fts.time_extrapolation
        nᴺ = n₂ + Ni - 2

        if te isa Cyclical
            possibly_wrapped_indices = Tuple(mod1(n, Nt) for n = n₂+1:nᴺ)
            index_range = tuple(n₁, n₂, possibly_wrapped_indices...)
        else
            nᴺ = min(nᴺ, Nt) # clip to Nt
            n₁ = nᴺ - Ni + 1 # recompute if clipped
            index_range = n₁:nᴺ
            index_range = tuple(index_range...)
        end

        set!(fts, index_range)
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

