using NCDatasets

using Oceananigans.Fields
using Oceananigans.Utils: validate_interval

"""
    netcdf_spatial_dimensions(::field)

Returns the NetCDF dimensions associated with a field.

Examples
========
julia> netcdf_spatial_dimensions(model.velocities.u)
("xF", "yC", "zC")

julia> netcdf_spatial_dimensions(model.tracers.T)
("xC", "yC", "zC")
"""
netcdf_spatial_dimensions(::Field{LX, LY, LZ}) where {LX, LY, LZ} = xdim(LX), ydim(LY), zdim(LZ)

xdim(::Type{Face}) = "xF"
xdim(::Type{Cell}) = "xC"
ydim(::Type{Face}) = "yF"
ydim(::Type{Cell}) = "yC"
zdim(::Type{Face}) = "zF"
zdim(::Type{Cell}) = "zC"

# This function converts an integer to a range, and nothing to a Colon
get_slice(n::Integer) = n:n
get_slice(n::UnitRange) = n
get_slice(n::Nothing) = Colon()

"""
    write_grid_and_attributes(model; filename="grid.nc", mode="c",
                              compression=0, attributes=Dict(), slice_kw...)

Writes grid and global `attributes` to `filename`. By default writes information
to a standalone `grid.nc` file.

Keyword arguments
=================
- `filename`  : File name to be saved under.
- `mode`: NetCDF file is opened in either clobber ("c") or append ("a") mode. Default: "c".
- `compression`: Defines the compression level of data from 0-9. Default: 0.
- `attributes`: Global attributes. Default: Dict().
"""
function write_grid_and_attributes(model; filename="grid.nc", mode="c",
                                   compression=0, attributes=Dict(), slice_kw...)

    dims = Dict(
        "xC" => collect(model.grid.xC),
        "yC" => collect(model.grid.yC),
        "zC" => collect(model.grid.zC),
        "xF" => collect(model.grid.xF)[1:end-1],
        "yF" => collect(model.grid.yF)[1:end-1],
        "zF" => collect(model.grid.zF)[1:end-1]
    )

    dim_attribs = Dict(
        "xC" => Dict("longname" => "Locations of the cell centers in the x-direction.", "units" => "m"),
        "yC" => Dict("longname" => "Locations of the cell centers in the y-direction.", "units" => "m"),
        "zC" => Dict("longname" => "Locations of the cell centers in the z-direction.", "units" => "m"),
        "xF" => Dict("longname" => "Locations of the cell faces in the x-direction.",   "units" => "m"),
        "yF" => Dict("longname" => "Locations of the cell faces in the y-direction.",   "units" => "m"),
        "zF" => Dict("longname" => "Locations of the cell faces in the z-direction.",   "units" => "m")
    )

    # Applies slices to the dimensions d
    for (d, slice) in slice_kw
        if String(d) in keys(dims)
            dims[String(d)] = dims[String(d)][get_slice(slice)]
        end
    end

    Dataset(filename, mode, attrib=attributes) do ds
        for (dim_name, dim_array) in dims
            defVar(ds, dim_name, dim_array, (dim_name,),
                   compression=compression, attrib=dim_attribs[dim_name])
        end
    end

    return nothing
end

const default_output_attributes =
    Dict("u" => Dict("longname" => "Velocity in the x-direction", "units" => "m/s"),
         "v" => Dict("longname" => "Velocity in the y-direction", "units" => "m/s"),
         "w" => Dict("longname" => "Velocity in the z-direction", "units" => "m/s"),
         "T" => Dict("longname" => "Temperature", "units" => "K"),
         "S" => Dict("longname" => "Salinity",    "units" => "g/kg"))

"""
    NetCDFOutputWriter <: AbstractOutputWriter

An output writer for writing to NetCDF files.
"""
mutable struct NetCDFOutputWriter{D, O, I, F, S} <: AbstractOutputWriter
             filename :: String
              dataset :: D
              outputs :: O
             interval :: I
            frequency :: F
              clobber :: Bool
               slices :: S
   len_time_dimension :: Int
             previous :: Float64
end

"""
    NetCDFOutputWriter(model, outputs; interval=nothing, frequency=nothing, filename=".",
                       clobber=true, global_attributes=Dict(), output_attributes=nothing,
                       slice_kw...)

Construct a `NetCDFOutputWriter` that writes `label, field` pairs in `outputs` (which should
be a `Dict`) to a NC file, where `label` is a symbol that labels the output and `field` is
a field from the model (e.g. `model.velocities.u`).

Keyword arguments
=================
- `filename`: Filepath to save output to. Default: "." (current working directory).
- `frequency`: Save output every `n` model iterations.
- `interval`: Save output every `t` units of model clock time.
- `clobber`: Remove existing files if their filenames conflict. Default: `true`.
- `compression`: Determines the compression level of data (0-9, default 0)
- `global_attributes`: Dict of model properties to save with every file (deafult: Dict())
- `output_attributes`: Dict of attributes to be saved with each field variable (reasonable
  defaults are provided for velocities, temperature, and salinity)
- `slice_kw`: `dimname = Union{OrdinalRange, Integer}` will slice the dimension `dimname`.
  All other keywords are ignored. E.g. `xC = 3:10` will only produce output along the dimension
  `xC` between indices 3 and 10 for all fields with `xC` as one of their dimensions. `xC = 1`
  is treated like `xC = 1:1`. Multiple dimensions can be sliced in one call. Not providing slices
  writes output over the entire domain.
"""

function NetCDFOutputWriter(model, outputs; interval=nothing, frequency=nothing, filename=".",
                            clobber=true, global_attributes=Dict(), output_attributes=Dict(),
                            compression=0, slice_kw...)

    validate_interval(interval, frequency)

    mode = clobber ? "c" : "a"

    # Initiates the output file with dimensions
    write_grid_and_attributes(model; filename=filename, compression=compression,
                              attributes=global_attributes, slice_kw...)

    # Opens the same output file for writing fields from the user-supplied variable outputs
    dataset = Dataset(filename, "a")

    # Creates an unliimited dimension "time"
    defDim(dataset, "time", Inf)
    sync(dataset)

    defVar(dataset, "time", Float64, ("time",))
    sync(dataset)

    len_time_dimension = 0 # Number of outputs so far

    # Ensure we have an attribute for every output. Use reasonable defaults if
    # none were specified by the user.
    for c in keys(outputs)
        if !haskey(output_attributes, c)
            output_attributes[c] = default_output_attributes[c]
        end
    end

    # Initiates empty variables for fields from the user-supplied variable outputs
    for (fieldname, field) in outputs
        FT = eltype(field.grid)
        defVar(dataset, fieldname, FT, (netcdf_spatial_dimensions(field)..., "time"),
               compression=compression, attrib=output_attributes[fieldname])
    end
    sync(dataset)

    # Stores slices for the dimensions of each output field
    slices = Dict{String, Vector{Union{OrdinalRange,Colon}}}()
    for (fieldname, field) in outputs
        slices[fieldname] = slice(field; slice_kw...)
    end

    return NetCDFOutputWriter(filename, dataset, outputs, interval, frequency,
                              clobber, slices, len_time_dimension, 0.0)
end


# Closes the outputwriter
Base.close(ow::NetCDFOutputWriter) = close(ow.dataset)

"""
    slice(field; slice_kw...)

For internal use only. Returns a slice for a field based on its dimensions and the
supplied slices in `slice_kw`.
"""
function slice(field; slice_kw...)
    slice_array = Vector{Union{AbstractRange,Colon}}()
    for dim in netcdf_spatial_dimensions(field)
        slice = get_slice(haskey(slice_kw, Symbol(dim)) ? slice_kw[Symbol(dim)] : nothing)
        push!(slice_array, slice)
    end
    return slice_array
end

"""
    write_output(model, OutputWriter)

For internal user only. Writes output to the netcdf file at specified intervals.
Increments the `time` dimension every time an output is written to the file.
"""
function write_output(model, ow::NetCDFOutputWriter)
    ow.len_time_dimension += 1
    ow.dataset["time"][ow.len_time_dimension] = model.clock.time
    for (fieldname, field) in ow.outputs
        ow.dataset[fieldname][:, :, :, ow.len_time_dimension] = view(interiorparent(field), ow.slices[fieldname]...)
    end
    sync(ow.dataset)
    return nothing
end
