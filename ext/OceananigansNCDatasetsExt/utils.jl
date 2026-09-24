#####
##### Utilities for general use
#####

dictify(outputs) = outputs
dictify(outputs::NamedTuple) = Dict(string(k) => dictify(v) for (k, v) in zip(keys(outputs), values(outputs)))

function create_time_dimension!(dataset; attrib=nothing, dimension_type=Float64)
    if "time" ∉ keys(dataset.dim)
        # Create an unlimited dimension "time"
        defDim(dataset, "time", Inf)
        defVar(dataset, "time", dimension_type, ("time",), attrib=attrib)
    end
end

#####
##### Conversion utilities
#####

# Using OrderedDict to preserve order of keys (important when saving positional arguments), and string(key) because that's what NetCDF supports as global_attributes.
convert_for_netcdf(dict::AbstractDict) = OrderedDict(string(key) => convert_for_netcdf(value) for (key, value) in dict)
convert_for_netcdf(x::Number) = x
convert_for_netcdf(x::Bool) = string(x)
convert_for_netcdf(x::NTuple{N, Number}) where N = collect(x)
convert_for_netcdf(x) = string(x)
convert_for_netcdf(::GPU) = "GPU()"
convert_for_netcdf(::CenterImmersedCondition) = "CenterImmersedCondition()"
convert_for_netcdf(::InterfaceImmersedCondition) = "InterfaceImmersedCondition()"

materialize_from_netcdf(dict::AbstractDict) = OrderedDict(Symbol(key) => materialize_from_netcdf(value) for (key, value) in dict)
materialize_from_netcdf(x::Number) = x
materialize_from_netcdf(x::Array) = Tuple(x)
materialize_from_netcdf(x::String) = materialize_serialized_output(x)
materialize_from_netcdf(x) = x

#####
##### Extension utilities
#####

ext(::Type{NetCDFWriter}) = ".nc"
