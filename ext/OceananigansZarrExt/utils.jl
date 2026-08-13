#####
##### Utilities for general use
#####

# We collect to ensure we return an array which Zarr.jl needs
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
##### Helpers: field dim names, location strings, indices strings
#####

zarr_field_coordinates(field::AbstractField) = zarr_field_coordinates(field, field.grid)

function zarr_field_coordinates(field::AbstractField, grid::RectilinearGrid)
    LX, LY, LZ = location(field)
    name_x = LX === Nothing ? "" : trilocation_dim_name("x", grid, LX(), LY(), nothing, Val(:x))
    name_y = LY === Nothing ? "" : trilocation_dim_name("y", grid, LX(), LY(), nothing, Val(:y))
    name_z = LZ === Nothing ? "" : trilocation_dim_name("z", grid, nothing, nothing, LZ(), Val(:z))
    return (name_x, name_y, name_z)
end

function zarr_field_coordinates(field::AbstractField, grid::LatitudeLongitudeGrid)
    LΛ, LΦ, LZ = location(field)
    name_λ = LΛ === Nothing ? "" : trilocation_dim_name("λ", grid, LΛ(), LΦ(), nothing, Val(:x))
    name_φ = LΦ === Nothing ? "" : trilocation_dim_name("φ", grid, LΛ(), LΦ(), nothing, Val(:y))
    name_z = LZ === Nothing ? "" : trilocation_dim_name("z", grid, nothing, nothing, LZ(), Val(:z))
    return (name_λ, name_φ, name_z)
end

function zarr_field_coordinates(field::AbstractField, grid::OrthogonalSphericalShellGrid)
    LΛ, LΦ, LZ = location(field)
    name_λ = LΛ === Nothing ? "" : trilocation_dim_name("λ", grid, LΛ(), LΦ(), nothing, Val(:x))
    name_φ = LΦ === Nothing ? "" : trilocation_dim_name("φ", grid, LΛ(), LΦ(), nothing, Val(:y))
    name_z = LZ === Nothing ? "" : trilocation_dim_name("z", grid, nothing, nothing, LZ(), Val(:z))
    return (name_λ, name_φ, name_z)
end

zarr_field_coordinates(field::AbstractField, grid::ImmersedBoundaryGrid) =
    zarr_field_coordinates(field, grid.underlying_grid)

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
materialize_from_zarr(x::AbstractString)  = @eval $(Meta.parse(x))
materialize_from_zarr(x)                  = x

zarr_safe_dict(x::OrderedDict) = Dict{String, Any}(x)
zarr_safe_dict(x::AbstractDict) = Dict{String, Any}(x)
zarr_safe_dict(x) = x

#####
##### Extension utilities
#####

ext(::Type{ZarrWriter}) = ".zarr"
