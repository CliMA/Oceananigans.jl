import Oceananigans.BoundaryConditions: getbc
import Oceananigans.Models: update_time_series!

using Oceananigans.Fields: AbstractField
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.TimeSteppers: AbstractTimeStepper
using Oceananigans.Models: AbstractModel

const CPUFTSBC = BoundaryCondition{<:Any, <:FieldTimeSeries}
const GPUFTSBC = BoundaryCondition{<:Any, <:GPUAdaptedFieldTimeSeries}

const FTSBC = Union{CPUFTSBC, GPUFTSBC}

@inline getbc(bc::FTSBC, i::Int, j::Int, grid::AbstractGrid, clock::Clock, args...) = bc.condition[i, j, Time(clock.time)]

function update_time_series!(fts::InMemoryFieldTimeSeries, time_index::Time)
    time = time_index.time
    n₁, n₂ = index_binary_search(fts.times, time, length(fts.times))
    update_series!(fts, n₂)
    return nothing
end

function update_time_series!(fts::InMemoryFieldTimeSeries, n::Int)
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

function update_time_series!(model::AbstractModel, clock::Clock)

    time = Time(clock.time)
    time_series_tuple = extract_field_timeseries(model)
    time_series_tuple = flattened_unique_values(time_series_tuple)

    for fts in time_series_tuple
        update_time_series!(fts, time)
    end

    return nothing
end

# All Oceananigans' types
function extract_field_timeseries(t) 
    prop = propertynames(t)
    if isempty(prop)
        return ()
    end

    return Tuple(extract_field_timeseries(getproperty(t, p)) for p in prop)
end


NonFTS = [:Number, :AbstractArray, :AbstractTimeStepper, :AbstractGrid]

for NonFTSType in NonFTS
    @eval extract_field_timeseries(::$NonFTSType) = ()
end

extract_field_timeseries(t::AbstractField)     = Tuple(extract_field_timeseries(getproperty(t, p)) for p in propertynames(t))
extract_field_timeseries(t::AbstractOperation) = Tuple(extract_field_timeseries(getproperty(t, p)) for p in propertynames(t))
extract_field_timeseries(t::Tuple)             = Tuple(extract_field_timeseries(n) for n in t)
extract_field_timeseries(t::NamedTuple)        = Tuple(extract_field_timeseries(n) for n in t)
extract_field_timeseries(f::FieldTimeSeries)   = f