using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.BoundaryConditions: DiscreteBoundaryFunction, ContinuousBoundaryFunction
using Oceananigans.Fields: flattened_unique_values, Field
using Oceananigans.Grids: AbstractGrid

#####
##### Utility for "extracting" FieldTimeSeries from large nested objects (eg models)
#####

extract_field_time_series(t1, tn...) = extract_field_time_series(tuple(t1, tn...))

# Utility used to extract field time series from a type through recursion.
@inline function extract_field_time_series(t)
    N = fieldcount(typeof(t))
    N === 0 && return nothing
    extracted = ntuple(i -> extract_field_time_series(getfield(t, i)), Val(N))
    return flattened_unique_values(extracted)
end

# Termination (move all here when we switch the code up)
extract_field_time_series(f::FieldTimeSeries) = f

# Extract the underlying FieldTimeSeries from TimeSeriesInterpolation
extract_field_time_series(f::TimeSeriesInterpolation) = f.time_series

# For types that do not contain `FieldTimeSeries`, halt the recursion
CannotPossiblyContainFTS = (:Number, :AbstractArray, :AbstractGrid, :AbstractField, :Returns)

for T in CannotPossiblyContainFTS
    @eval extract_field_time_series(::$T) = nothing
end

# Special recursion rules for `Tuple` and `Field` types
extract_field_time_series(t::AbstractOperation) = Tuple(extract_field_time_series(getproperty(t, p)) for p in propertynames(t))
extract_field_time_series(t::Union{Tuple, NamedTuple}) = map(extract_field_time_series, t)

const CPUFTSBC = BoundaryCondition{<:Any, <:FieldTimeSeries}
const GPUFTSBC = BoundaryCondition{<:Any, <:GPUAdaptedFieldTimeSeries}
const DFBC     = BoundaryCondition{<:Any, <:DiscreteBoundaryFunction}
const CFBC     = BoundaryCondition{<:Any, <:ContinuousBoundaryFunction}
const FTSBC = Union{CPUFTSBC, GPUFTSBC, DFBC, CFBC}

const WFTSBCS = FieldBoundaryConditions{<:FTSBC}
const EFTSBCS = FieldBoundaryConditions{<:Any, <:FTSBC}
const SFTSBCS = FieldBoundaryConditions{<:Any, <:Any, <:FTSBC}
const NFTSBCS = FieldBoundaryConditions{<:Any, <:Any, <:Any, <:FTSBC}
const BFTSBCS = FieldBoundaryConditions{<:Any, <:Any, <:Any, <:Any, <:FTSBC}
const TFTSBCS = FieldBoundaryConditions{<:Any, <:Any, <:Any, <:Any, <:Any, <:FTSBC}
const IFTSBCS = FieldBoundaryConditions{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:FTSBC}

const FieldBCsFTS = Union{WFTSBCS, EFTSBCS, SFTSBCS, NFTSBCS, BFTSBCS, TFTSBCS, IFTSBCS}
const FieldFTS = Field{LX, LY, LZ, O, G, I, D, T, <:FieldBCsFTS} where {LX, LY, LZ, O, G, I, D, T}

extract_field_time_series(f::FieldFTS) = extract_field_time_series(f.boundary_conditions)

extract_field_time_series(bcs::FieldBoundaryConditions) = (extract_field_time_series(bcs.west),
                                                           extract_field_time_series(bcs.east),
                                                           extract_field_time_series(bcs.south),
                                                           extract_field_time_series(bcs.north),
                                                           extract_field_time_series(bcs.bottom),
                                                           extract_field_time_series(bcs.top),
                                                           extract_field_time_series(bcs.immersed))

extract_field_time_series(bc::BoundaryCondition) = extract_field_time_series(bc.condition)
extract_field_time_series(bc::DiscreteBoundaryFunction) = extract_field_time_series(bc.parameters)
extract_field_time_series(bc::ContinuousBoundaryFunction) = extract_field_time_series(bc.parameters)
