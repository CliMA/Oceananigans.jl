using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.BoundaryConditions: DiscreteBoundaryFunction, ContinuousBoundaryFunction
using Oceananigans.Fields: flattened_unique_values, Field
using Oceananigans.Grids: AbstractGrid

#####
##### Compile-time predicate: does a type (recursively) contain a `FieldTimeSeries`?
#####

function type_contains_field_time_series(T, seen)
    (T === Union{} || T in seen) && return false
    push!(seen, T)
    T isa Union && return type_contains_field_time_series(T.a, seen) ||
                          type_contains_field_time_series(T.b, seen)
    T <: FieldTimeSeries && return true
    T <: GPUAdaptedFieldTimeSeries && return true
    (T <: Number || T <: AbstractGrid) && return false
    (T <: AbstractArray && !(T <: AbstractField)) && return false
    isconcretetype(T) || return true # abstract / unresolved: assume an FTS may hide inside
    return any(ft -> type_contains_field_time_series(ft, seen), fieldtypes(T))
end

@generated function has_field_time_series(::Type{T}) where T
    return type_contains_field_time_series(T, Base.IdSet{Any}()) ? :true : :false
end

#####
##### Utility for "extracting" FieldTimeSeries from large nested objects (eg models)
#####

extract_field_time_series(t1, tn...) = extract_field_time_series(tuple(t1, tn...))

@inline concatenate_extracted(::Tuple{}) = ()
@inline concatenate_extracted(t::Tuple) = (first(t)..., concatenate_extracted(Base.tail(t))...)

# Utility used to extract field time series from a type through recursion over its fields.
@inline function extract_field_time_series(t)
    has_field_time_series(typeof(t)) || return ()
    N = fieldcount(typeof(t))
    N === 0 && return ()
    return concatenate_extracted(ntuple(i -> extract_field_time_series(getfield(t, i)), Val(N)))
end

# Terminations: a `FieldTimeSeries` (or the series underlying a `TimeSeriesInterpolation`) is collected.
extract_field_time_series(f::FieldTimeSeries) = (f,)
extract_field_time_series(f::TimeSeriesInterpolation) = (f.time_series,)

# Types that cannot contain a `FieldTimeSeries` halt the recursion with an empty tuple.
CannotPossiblyContainFTS = (:Number, :AbstractArray, :AbstractGrid, :AbstractField, :Returns, :Nothing)

for T in CannotPossiblyContainFTS
    @eval extract_field_time_series(::$T) = ()
end

# Special recursion rules for `AbstractOperation`, `Tuple` and `NamedTuple`
extract_field_time_series(t::AbstractOperation) =
    has_field_time_series(typeof(t)) ?
        concatenate_extracted(map(p -> extract_field_time_series(getproperty(t, p)), propertynames(t))) : ()

extract_field_time_series(t::Union{Tuple, NamedTuple}) =
    has_field_time_series(typeof(t)) ?
        concatenate_extracted(map(extract_field_time_series, values(t))) : ()

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

extract_field_time_series(bcs::FieldBoundaryConditions) =
    has_field_time_series(typeof(bcs)) ?
        concatenate_extracted((extract_field_time_series(bcs.west),
                               extract_field_time_series(bcs.east),
                               extract_field_time_series(bcs.south),
                               extract_field_time_series(bcs.north),
                               extract_field_time_series(bcs.bottom),
                               extract_field_time_series(bcs.top),
                               extract_field_time_series(bcs.immersed))) : ()

extract_field_time_series(bc::BoundaryCondition) = extract_field_time_series(bc.condition)
extract_field_time_series(bc::DiscreteBoundaryFunction) = extract_field_time_series(bc.parameters)
extract_field_time_series(bc::ContinuousBoundaryFunction) = extract_field_time_series(bc.parameters)
