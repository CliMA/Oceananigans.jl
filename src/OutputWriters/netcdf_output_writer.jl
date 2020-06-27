using NCDatasets

using Oceananigans.Fields
using Oceananigans.Utils: validate_interval
using Oceananigans.Grids: topology, interior_x_indices, interior_y_indices, interior_z_indices
using Oceananigans.Fields: cpudata

# Possibly should
collect_face_nodes(topo, ξF) = collect(ξF)[1:end-1]
collect_face_nodes(::Bounded, ξF) = collect(ξF)

const default_output_attributes = Dict(
    "u" => Dict("longname" => "Velocity in the x-direction", "units" => "m/s"),
    "v" => Dict("longname" => "Velocity in the y-direction", "units" => "m/s"),
    "w" => Dict("longname" => "Velocity in the z-direction", "units" => "m/s"),
    "b" => Dict("longname" => "Buoyancy",                    "units" => "m/s²"),
    "T" => Dict("longname" => "Conservative temperature",    "units" => "°C"),
    "S" => Dict("longname" => "Absolute salinity",           "units" => "g/kg")
)

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
     previous :: Float64
end

"""
    NetCDFOutputWriter(model, outputs; frequency=nothing, interval=nothing, filename=".",
                       global_attributes=Dict(), output_attributes=Dict(), dimensions=Dict(),
                       clobber=true, compression=0, slice_kw...)

Construct a `NetCDFOutputWriter` that writes `(label, output)` pairs in `outputs` (which should
be a `Dict`) to a NetCDF file, where `label` is a string that labels the output and `output` is
either field from the model (e.g. `model.velocities.u`) or a function `f(model)` that returns
something to be written to disk.

Keyword arguments
=================
- `frequency`: Save output every `n` model iterations.
- `interval`: Save output every `t` units of model clock time.
- `filename`: Filepath to save output to. Default: "." (current working directory).
- `global_attributes`: Dict of model properties to save with every file (deafult: Dict())
- `output_attributes`: Dict of attributes to be saved with each field variable (reasonable
  defaults are provided for velocities, temperature, and salinity).
- `dimensions`: A Dict of dimensions to apply to outputs (useful for function outputs as
  field dimensions can be inferred).
- `clobber`: Remove existing files if their filenames conflict. Default: `true`.
- `compression`: Determines the compression level of data (0-9, default 0)
- `slice_kw`: `dimname = Union{OrdinalRange, Integer}` will slice the dimension `dimname`.
  All other keywords are ignored. E.g. `xC = 3:10` will only produce output along the dimension
  `xC` between indices 3 and 10 for all fields with `xC` as one of their dimensions. `xC = 1`
  is treated like `xC = 1:1`. Multiple dimensions can be sliced in one call. Not providing slices
  writes output over the entire domain.
"""

function NetCDFOutputWriter(model, outputs; filename,
                                     interval = nothing,
                                    frequency = nothing,
                            global_attributes = Dict(),
                            output_attributes = Dict(),
                                   dimensions = Dict(),
                                      clobber = true,
                                  compression = 0,
                                           xC = interior_x_indices(Cell, model.grid),
                                           xF = interior_x_indices(Face, model.grid),
                                           yC = interior_y_indices(Cell, model.grid),
                                           yF = interior_y_indices(Face, model.grid),
                                           zC = interior_z_indices(Cell, model.grid),
                                           zF = interior_z_indices(Face, model.grid)
                           )


    mode = clobber ? "c" : "a"
    validate_interval(interval, frequency)

    # Generates a dictionary with keys "xC", "xF", etc, whose values give the slices to be saved.
    slice_keywords = Dict(name => a for (name, a) in zip(("xC", "yC", "zC", "xF", "yF", "zF"),
                                                         ( xC,   yC,   zC,   xF,   yF,   zF )))

    # Initiates the output file with dimensions
    write_grid_and_attributes(model; filename=filename, compression=compression,
                              attributes=global_attributes, mode=mode,
                              xC=xC, yC=yC, zC=zC, xF=xF, yF=yF, zF=zF)

    # Opens the same output file for writing fields from the user-supplied variable outputs
    dataset = Dataset(filename, "a")
    println("hello")
    # Creates an unliimited dimension "time"
    defDim(dataset, "time", Inf)
    defVar(dataset, "time", Float64, ("time",))
    sync(dataset)

    # Ensure we have an attribute for every output. Use reasonable defaults if
    # none were specified by the user.
    for c in keys(outputs)
        if !haskey(output_attributes, c)
            output_attributes[c] = default_output_attributes[c]
        end
    end

    # Initiates empty variables for fields from the user-supplied variable outputs
    for (name, output) in outputs
        if output isa Field
            FT = eltype(output.grid)
            defVar(dataset, name, FT, (netcdf_spatial_dimensions(output)..., "time"),
                   compression=compression, attrib=output_attributes[name])
        else
            defVar(dataset, name, Float64, (dimensions[name]..., "time"),
                   compression=compression, attrib=output_attributes[name])
        end
    end
    sync(dataset)

    field_outputs = filter(o -> o.second isa Field, outputs) # extract outputs whose values are Fields

    # Store a slice specification for each field.
    slices = Dict(name => slice_indices(field; xC=xC, yC=yC, zC=zC, xF=xF, yF=yF, zF=zF)
                  for (name, field) in field_outputs)

    return NetCDFOutputWriter(filename, dataset, outputs, interval, frequency,
                              clobber, slices, 0.0)
end

Base.open(ow::NetCDFOutputWriter) = Dataset(ow.filename, "a")
Base.close(ow::NetCDFOutputWriter) = close(ow.dataset)

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
ydim(::Type{Face}) = "yF"
zdim(::Type{Face}) = "zF"

xdim(::Type{Cell}) = "xC"
ydim(::Type{Cell}) = "yC"
zdim(::Type{Cell}) = "zC"

# This function allows users to specify slices with integers; eg xC=3.
# Note: size(a[3:3, :, :]) = (1, Ny, Nz) versus size(a[3, :, :]) = (Ny, Nz)
get_slice(n::Integer) = n:n
get_slice(n::UnitRange) = n

"""
    slice_indices(field; slice_specs...)

Returns an array of indices that specify a view over a field's data.
"""
slice_indices(field; slice_specs...) =
    [get_slice(slice_specs[Symbol(dim)]) for dim in netcdf_spatial_dimensions(field)]

"""
    write_output(model, OutputWriter)

Writes output to the netcdf file at specified intervals. Increments the `time` dimension
every time an output is written to the file.
"""
function write_output(model, ow::NetCDFOutputWriter)
    ds = ow.dataset

    time_index = length(ds["time"]) + 1
    ds["time"][time_index] = model.clock.time

    for (name, output) in ow.outputs

        if output isa Field
            data = cpudata(output) # Transfer data to CPU if parent(output) is a CuArray
            ds[name][:, :, :, time_index] = view(data, ow.slices[name]...)
        else
            data = output(model)
            colons = Tuple(Colon() for _ in 1:ndims(data))
            ds[name][colons..., time_index] = data
        end

    end

    sync(ow.dataset)

    return nothing
end

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
function write_grid_and_attributes(model;
                                      filename = "grid.nc",
                                          mode = "c",
                                         units = "m",
                                   compression = 0,
                                    attributes = Dict(),
                                    slice_keywords...)

    dims = Dict(
                "xC" => collect(model.grid.xC[:, 1, 1]),
                "yC" => collect(model.grid.yC[1, :, 1]),
                "zC" => collect(model.grid.zC[1, 1, :]),

                "xF" => collect(model.grid.xF[:, 1, 1]),
                "yF" => collect(model.grid.yF[1, :, 1]),
                "zF" => collect(model.grid.zF[1, 1, :]),
               )

    dim_attribs = Dict(
                       "xC" => Dict("longname" => "Locations of the cell centers in the x-direction.", "units" => units),
                       "yC" => Dict("longname" => "Locations of the cell centers in the y-direction.", "units" => units),
                       "zC" => Dict("longname" => "Locations of the cell centers in the z-direction.", "units" => units),
                       "xF" => Dict("longname" => "Locations of the cell faces in the x-direction.",   "units" => units),
                       "yF" => Dict("longname" => "Locations of the cell faces in the y-direction.",   "units" => units),
                       "zF" => Dict("longname" => "Locations of the cell faces in the z-direction.",   "units" => units)
                      )

    # Slice coordinate arrays stored in the dims dict
    for (dim, indices) in slice_keywords
        dim = string(dim) # convert symbol to string
        dims[dim] = dims[dim][get_slice(indices)] # overwrite entries in dims Dict
    end

    Dataset(filename, mode, attrib=attributes) do ds
        for (dim_name, dim_array) in dims
            defVar(ds, dim_name, dim_array, (dim_name,),
                   compression=compression, attrib=dim_attribs[dim_name])
        end
    end

    return nothing
end
