"""
    netcdf_spatial_dimensions(::field)

Returns the dimensions associated with a field.

Examples
========

julia> netcdf_spatial_dimensions(model.velocities.u)
("xF", "yC", "zC")

julia> netcdf_spatial_dimensions(model.tracers.T)
("xC", "yC", "zC")
"""
function netcdf_spatial_dimensions(field::Field{LX, LY, LZ}) where {LX, LY, LZ}
    xdim(LX), ydim(LY), zdim(LZ)
end

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
    write_grid(model; filename="grid.nc", mode="c",
               compression=0, attributes=Dict(), slice_kw...)

Writes a grid.nc file that contains all the dimensions of the domain.

Keyword arguments
=================

    - `filename::String`  : File name to be saved under
    - `mode::String`      : Netcdf file is opened in either clobber ("c") or append ("a") mode (Default: "c" )
    - `compression::Int`  : Defines the compression level of data (0-9, default 0)
    - `attributes::Dict`  : Attributes (default: Dict())
"""
function write_grid(model; filename="./grid.nc", mode="c",
                    compression=0, attributes=Dict(), slice_kw...)

    dimensions = Dict(
        "xC" => collect(model.grid.xC),
        "yC" => collect(model.grid.yC),
        "zC" => collect(model.grid.zC),
        "xF" => collect(model.grid.xF)[1:end-1],
        "yF" => collect(model.grid.yF)[1:end-1],
        "zF" => collect(model.grid.zF)[1:end-1]
    )

    dim_attrib = Dict(
        "xC" => ["longname" => "Locations of the cell centers in the x-direction.", "units" => "m"],
        "yC" => ["longname" => "Locations of the cell centers in the y-direction.", "units" => "m"],
        "zC" => ["longname" => "Locations of the cell centers in the z-direction.", "units" => "m"],
        "xF" => ["longname" => "Locations of the cell faces in the x-direction.",   "units" => "m"],
        "yF" => ["longname" => "Locations of the cell faces in the y-direction.",   "units" => "m"],
        "zF" => ["longname" => "Locations of the cell faces in the z-direction.",   "units" => "m"]
    )

    # Applies slices to the dimensions d
    for (d, slice) in slice_kw
        if String(d) in keys(dimensions)
            dimensions[String(d)] = dimensions[String(d)][get_slice(slice)]
        end
    end

    # Writes the sliced dimensions to the specified netcdf file
    Dataset(filename, mode, attrib=attributes) do ds
        for (dimname, dimarray) in dimensions
            defVar(ds, dimname, dimarray, (dimname,),
                   compression=compression, attrib=dim_attrib[dimname])
        end
    end

    return nothing
end


"""
    NetCDFOutputWriter <: AbstractOutputWriter

An output writer for writing to NetCDF files.
"""
mutable struct NetCDFOutputWriter <: AbstractOutputWriter
             filename :: String
              dataset :: Any
              outputs :: Dict
             interval :: Union{Nothing, AbstractFloat}
            frequency :: Union{Nothing, Int}
              clobber :: Bool
               slices :: Dict
   len_time_dimension :: Int
             previous :: Float64
end

"""
    NetCDFOutputWriter(model, outputs; interval=nothing, frequency=nothing, filename=".",
                                   clobber=true, global_attributes=Dict(), output_attributes=nothing, slice_kw...)

Construct a `NetCDFOutputWriter` that writes `label, field` pairs in `outputs` (which can be a
`Dict` or `NamedTuple`) to a NC file, where `label` is a symbol that labels the output and
`field` is a field from the model (e.g. `model.velocities.u`).

Keyword arguments
=================

    - `filename::String`         : Directory to save output to. Default: "." (current working directory).
    - `frequency::Int`           : Save output every `n` model iterations.
    - `interval::Int`            : Save output every `t` units of model clock time.
    - `clobber::Bool`            : Remove existing files if their filenames conflict. Default: `false`.
    - `compression::Int`         : Determines the compression level of data (0-9, default 0)
    - `global_attributes::Dict`  : Dict of model properties to save with every file (deafult: Dict())
    - `output_attributes::Dict`  : Dict of attributes to be saved with each field variable (reasonable
                                   defaults are provided for velocities, temperature, and salinity)
    - `slice_kw`                 : `dimname = Union{OrdinalRange, Integer}` will slice the dimension `dimname`.
                                   All other keywords are ignored.
                                   e.g. `xC = 3:10` will only produce output along the dimension `xC` between
                                   indices 3 and 10 for all fields with `xC` as one of their dimensions.
                                   `xC = 1` is treated like `xC = 1:1`.
                                   Multiple dimensions can be sliced in one call. Not providing slices writes
                                   output over the entire domain.
"""

function NetCDFOutputWriter(model, outputs; interval=nothing, frequency=nothing, filename=".",
                        clobber=true, global_attributes=Dict(), output_attributes=nothing, compression=0, slice_kw...)

    validate_interval(interval, frequency)

    mode = clobber ? "c" : "a"

    # Initiates the output file with dimensions
    write_grid(model; filename=filename, mode=mode,
               compression=compression, attrib=global_attributes, slice_kw...)

    # Opens the same output file for writing fields from the user-supplied variable outputs
    dataset = Dataset(filename, "a")

    # Creates an unliimited dimension "Time"
    defDim(dataset, "Time", Inf); sync(dataset)
    defVar(dataset, "Time", Float64, ("Time",)); sync(dataset)
    len_time_dimension = 0 # Number of time-steps so far

    if isa(output_attributes, Nothing)
        output_attributes = Dict("u" => Dict("longname" => "Velocity in the x-direction", "units" => "m/s"),
                            "v" => Dict("longname" => "Velocity in the y-direction", "units" => "m/s"),
                            "w" => Dict("longname" => "Velocity in the z-direction", "units" => "m/s"),
                            "T" => Dict("longname" => "Temperature", "units" => "K"),
                            "S" => Dict("longname" => "Salinity", "units" => "g/kg"))
    end

    # Initiates empty Float32 arrays for fields from the user-supplied variable outputs
    for (fieldname, field) in outputs
        dtype = eltype(interiorparent(field))
        defVar(dataset, fieldname, dtype, (netcdf_spatial_dimensions(field)...,"Time"),
               compression=compression, attrib=output_attributes[fieldname])
    end
    sync(dataset)

    # Stores slices for the dimensions of each output field
    slices = Dict{String, Vector{Union{OrdinalRange,Colon}}}()
    for (fieldname, field) in outputs
        slices[fieldname] = slice(field; slice_kw...)
    end

    return NetCDFOutputWriter(filename, dataset, outputs, interval,
                          frequency, clobber, slices, len_time_dimension, 0.0)
end


# Closes the outputwriter
Base.close(fw::NetCDFOutputWriter) = close(fw.dataset)

function read_output(fw::NetCDFOutputWriter, fieldname)
    ds = Dataset(fw.filename,"r")
    field = ds[fieldname][:,:,:,end]
    close(ds)
    return field
end

"""
    slice(field; slice_kw...)

For internal use only. Returns a slice for a field based on its dimensions and the supplied slices in `slice_kw`.
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

For internal user only. Writes output to the netcdf file at specified intervals. Increments the `Time` dimension every time an output is written to the file.
"""
function write_output(model, fw::NetCDFOutputWriter)
    fw.len_time_dimension += 1
    fw.dataset["Time"][fw.len_time_dimension] = model.clock.time
    for (fieldname, field) in fw.outputs
        fw.dataset[fieldname][:, :, :, fw.len_time_dimension] = view(interiorparent(field), fw.slices[fieldname]...)
    end
    sync(fw.dataset)
    return nothing
end
