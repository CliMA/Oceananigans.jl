using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Fields: flattened_unique_values

#####
##### Utility for "extracting" FieldTimeSeries from large nested objects (eg models)
#####

extract_field_time_series(t1, tn...) = extract_field_time_series(tuple(t1, tn...))

# Utility used to extract field time series from a type through recursion
function extract_field_time_series(t) 
    prop = propertynames(t)
    if isempty(prop)
        return nothing
    end

    extracted = Tuple(extract_field_time_series(getproperty(t, p)) for p in prop)
    flattened = flattened_unique_values(extracted)

    return flattened
end

# Termination (move all here when we switch the code up)
extract_field_time_series(f::FieldTimeSeries) = f

# For types that do not contain `FieldTimeSeries`, halt the recursion
CannotPossiblyContainFTS = (:Number, :AbstractArray)

for T in CannotPossiblyContainFTS
    @eval extract_field_time_series(::$T) = nothing
end

# Special recursion rules for `Tuple` and `Field` types
extract_field_time_series(t::AbstractField)     = Tuple(extract_field_time_series(getproperty(t, p)) for p in propertynames(t))
extract_field_time_series(t::AbstractOperation) = Tuple(extract_field_time_series(getproperty(t, p)) for p in propertynames(t))

extract_field_time_series(t::Union{Tuple, NamedTuple}) = map(extract_field_time_series, t)

