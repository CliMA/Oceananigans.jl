#####
##### Utilities for general use
#####

# We collect to ensure we return an array which NCDatasets.jl needs
# instead of a range or offset array.
function collect_dim(ξ, ℓ, T, N, H, inds, with_halos)
    if with_halos
        return collect(ξ)
    else
        inds = validate_index(inds, ℓ, T, N, H)
        inds = restrict_to_interior(inds, ℓ, T, N)
        return collect(ξ[inds])
    end
end

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
materialize_from_zarr(x::AbstractString)  = @eval $(Meta.parse(x))
materialize_from_zarr(x)                  = x


#####
##### Extension utilities
#####

ext(::Type{ZarrWriter}) = ".zarr"
