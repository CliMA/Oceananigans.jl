
#####
##### Helpers: location strings and indices strings
#####

# Location and indices as JSON-friendly String tuples.
location_strings(field::AbstractField) = map(loc -> loc === Nothing ? "Nothing" : string(loc),
                                             location(field))

indices_strings(field::AbstractField) = map(index_string, indices(field))
index_string(::Colon) = ":"
index_string(r::AbstractUnitRange) = string(first(r), ":", last(r))
index_string(i::Integer) = string(i)

#####
##### Conversion utilities
#####

# Using OrderedDict to preserve order of keys (important when saving positional arguments), and string(key) because that's what Zarr supports as global_attributes.
convert_for_zarr(dict::AbstractDict) = OrderedDict{String, Any}(string(k) => convert_for_zarr(v) for (k, v) in dict)
convert_for_zarr(x::Number)         = x
convert_for_zarr(x::Bool)           = string(x)
convert_for_zarr(x::NTuple{N, Number}) where N = collect(x)
convert_for_zarr(::CPU)             = "CPU()"
convert_for_zarr(::GPU)             = "GPU()"
# A Distributed arch is not serializable in a portable way; record a placeholder.
# The reader takes `architecture` as a kwarg and substitutes it in via the
# `args_ordered` override in `reconstruct_zarr_grid`.
convert_for_zarr(::Distributed)     = "CPU()"
convert_for_zarr(x)                 = string(x)

materialize_from_zarr(dict::AbstractDict) = OrderedDict{Symbol, Any}(Symbol(k) => materialize_from_zarr(v) for (k, v) in dict)
materialize_from_zarr(x::Number)          = x
materialize_from_zarr(x::AbstractArray)   = Tuple(x)
materialize_from_zarr(x::AbstractString)  = materialize_serialized_output(x)
materialize_from_zarr(x)                  = x

zarr_safe_dict(x::OrderedDict) = Dict{String, Any}(x)
zarr_safe_dict(x::AbstractDict) = Dict{String, Any}(x)
zarr_safe_dict(x) = x

#####
##### Extension utilities
#####

ext(::Type{ZarrWriter}) = ".zarr"
