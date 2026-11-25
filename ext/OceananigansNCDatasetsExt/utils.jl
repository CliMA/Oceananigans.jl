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
OutputWriters.convert_for_netcdf(dict::AbstractDict) = OrderedDict(string(key) => convert_for_netcdf(value) for (key, value) in dict)
OutputWriters.convert_for_netcdf(x::Number) = x
OutputWriters.convert_for_netcdf(x::Bool) = string(x)
OutputWriters.convert_for_netcdf(x::NTuple{N, Number}) where N = collect(x)
OutputWriters.convert_for_netcdf(z::AbstractVerticalCoordinate) = serialize_vertical_coordinate(z)
OutputWriters.convert_for_netcdf(x) = string(x)
OutputWriters.convert_for_netcdf(::GPU) = "GPU()"
OutputWriters.convert_for_netcdf(::CenterImmersedCondition) = "CenterImmersedCondition()"
OutputWriters.convert_for_netcdf(::InterfaceImmersedCondition) = "InterfaceImmersedCondition()"

OutputWriters.materialize_from_netcdf(dict::AbstractDict) = OrderedDict(Symbol(key) => materialize_from_netcdf(value) for (key, value) in dict)
OutputWriters.materialize_from_netcdf(x::Number) = x
OutputWriters.materialize_from_netcdf(x::Array) = Tuple(x)
OutputWriters.materialize_from_netcdf(x::String) = materialize_serialized_output(x)
OutputWriters.materialize_from_netcdf(x) = x

#####
##### Extension utilities
#####

ext(::Type{NetCDFWriter}) = ".nc"
