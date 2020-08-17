using Dates: now
using NCDatasets
using Oceananigans.Fields
using Oceananigans.Fields: cpudata
using Oceananigans.Utils: validate_intervals, versioninfo_with_gpu, oceananigans_versioninfo
using Oceananigans.Grids: topology, interior_x_indices, interior_y_indices, interior_z_indices,
                          all_x_indices, all_y_indices, all_z_indices

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
    NetCDFOutputWriter{D, O, I, T, S} <: AbstractOutputWriter

An output writer for writing to NetCDF files.
"""
mutable struct NetCDFOutputWriter{D, O, I, T, S} <: AbstractOutputWriter
              filename :: String
               dataset :: D
               outputs :: O
    iteration_interval :: I
         time_interval :: T
               clobber :: Bool
                slices :: S
              previous :: Float64
               verbose :: Bool
end

"""
    NetCDFOutputWriter(model, outputs; filename, iteration_interval=nothing, time_interval=nothing,
                       global_attributes=Dict(), output_attributes=Dict(), dimensions=Dict(),
                       clobber=true, compression=0, with_halos=false, verbose=false, slice_kwargs...)

Construct a `NetCDFOutputWriter` that writes `(label, output)` pairs in `outputs` (which should
be a `Dict`) to a NetCDF file, where `label` is a string that labels the output and `output` is
either a field from the model (e.g. `model.velocities.u`) or a function `f(model)` that returns
something to be written to disk. Custom output requires the spatial `dimensions` to be manually
specified.

Keyword arguments
=================
- `iteration_interval`: Save output every `n` model iterations.
- `time_interval`: Save output every `t` units of model clock time.
- `filename`: Filepath to save output to.
- `global_attributes`: Dict of model properties to save with every file (deafult: `Dict()`)
- `output_attributes`: Dict of attributes to be saved with each field variable (reasonable
  defaults are provided for velocities, buoyancy, temperature, and salinity).
- `dimensions`: A `Dict` of dimension tuples to apply to outputs (useful for function outputs
  as field dimensions can be inferred).
- `with_halos`: Include the halo regions in the grid coordinates and output fields
  (default: `false`).
- `clobber`: Remove existing files if their filenames conflict. Default: `true`.
- `compression`: Determines the compression level of data (0-9, default 0)
- `slice_kwargs`: `dimname = Union{OrdinalRange, Integer}` will slice the dimension `dimname`.
  All other keywords are ignored. E.g. `xC = 3:10` will only produce output along the dimension
  `xC` between indices 3 and 10 for all fields with `xC` as one of their dimensions. `xC = 1`
  is treated like `xC = 1:1`. Multiple dimensions can be sliced in one call. Not providing slices
  writes output over the entire domain (including halo regions if `with_halos=true`).

Examples
========
Saving the u velocity field and temperature fields, the full 3D fields and surface 2D slices
to separate NetCDF files:

```jldoctest netcdf1
using Oceananigans, Oceananigans.OutputWriters

grid = RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1));

model = IncompressibleModel(grid=grid);

simulation = Simulation(model, Δt=12, stop_time=3600);

fields = Dict("u" => model.velocities.u, "T" => model.tracers.T);

simulation.output_writers[:field_writer] =
    NetCDFOutputWriter(model, fields, filename="fields.nc", time_interval=60)

# output
NetCDFOutputWriter (time_interval=60): fields.nc
├── dimensions: zC(16), zF(17), xC(16), yF(16), xF(16), yC(16), time(0)
└── 2 outputs: ["T", "u"]
```

```jldoctest netcdf1
simulation.output_writers[:surface_slice_writer] =
    NetCDFOutputWriter(model, fields, filename="surface_xy_slice.nc",
                       time_interval=60, zC=grid.Nz, zF=grid.Nz+1)

# output
NetCDFOutputWriter (time_interval=60): surface_xy_slice.nc
├── dimensions: zC(1), zF(1), xC(16), yF(16), xF(16), yC(16), time(0)
└── 2 outputs: ["T", "u"]
```

Writing a scalar, profile, and slice to NetCDF:

```jldoctest
using Oceananigans, Oceananigans.OutputWriters

grid = RegularCartesianGrid(size=(16, 16, 16), extent=(1, 2, 3));

model = IncompressibleModel(grid=grid);

simulation = Simulation(model, Δt=1.25, stop_iteration=3);

f(model) = model.clock.time^2; # scalar output

g(model) = model.clock.time .* exp.(znodes(Cell, grid)); # vector/profile output

h(model) = model.clock.time .* (   sin.(xnodes(Cell, grid, reshape=true)[:, :, 1])
                            .*     cos.(ynodes(Face, grid, reshape=true)[:, :, 1])); # xy slice output

outputs = Dict("scalar" => f, "profile" => g, "slice" => h);

dims = Dict("scalar" => (), "profile" => ("zC",), "slice" => ("xC", "yC"));

output_attributes = Dict(
    "scalar"  => Dict("longname" => "Some scalar", "units" => "bananas"),
    "profile" => Dict("longname" => "Some vertical profile", "units" => "watermelons"),
    "slice"   => Dict("longname" => "Some slice", "units" => "mushrooms")
);

global_attributes = Dict("location" => "Bay of Fundy", "onions" => 7);

simulation.output_writers[:things] =
    NetCDFOutputWriter(model, outputs,
                       iteration_interval=1, filename="things.nc", dimensions=dims, verbose=true,
                       global_attributes=global_attributes, output_attributes=output_attributes)

# output
NetCDFOutputWriter (iteration_interval=1): things.nc
├── dimensions: zC(16), zF(17), xC(16), yF(16), xF(16), yC(16), time(0)
└── 3 outputs: ["profile", "slice", "scalar"]
```
"""
function NetCDFOutputWriter(model, outputs; filename,
    iteration_interval = nothing,
         time_interval = nothing,
     global_attributes = Dict(),
     output_attributes = Dict(),
            dimensions = Dict(),
               clobber = true,
           compression = 0,
            with_halos = false,
               verbose = false,
    xC = with_halos ? all_x_indices(Cell, model.grid) : interior_x_indices(Cell, model.grid),
    xF = with_halos ? all_x_indices(Face, model.grid) : interior_x_indices(Face, model.grid),
    yC = with_halos ? all_y_indices(Cell, model.grid) : interior_y_indices(Cell, model.grid),
    yF = with_halos ? all_y_indices(Face, model.grid) : interior_y_indices(Face, model.grid),
    zC = with_halos ? all_z_indices(Cell, model.grid) : interior_z_indices(Cell, model.grid),
    zF = with_halos ? all_z_indices(Face, model.grid) : interior_z_indices(Face, model.grid)
    )

    mode = clobber ? "c" : "a"
    validate_intervals(iteration_interval, time_interval)

    # Ensure we can add metadata to the global attributes later by converting to pairs of type {Any, Any}.
    global_attributes = Dict{Any, Any}(k => v for (k, v) in global_attributes)

    # Generates a dictionary with keys "xC", "xF", etc, whose values give the slices to be saved.
    slice_keywords = Dict(name => a for (name, a) in zip(("xC", "yC", "zC", "xF", "yF", "zF"),
                                                         ( xC,   yC,   zC,   xF,   yF,   zF )))

    # Initializes the output file with dimensions.
    write_grid_and_attributes(model; filename=filename, compression=compression,
                              with_halos=with_halos, attributes=global_attributes, mode=mode,
                              xC=xC, yC=yC, zC=zC, xF=xF, yF=yF, zF=zF)

    # Opens the same output file for writing fields from the user-supplied variable outputs
    dataset = Dataset(filename, "a")

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

    # extract outputs whose values are Fields
    field_outputs = filter(o -> o.second isa Field, outputs)

    # Store a slice specification for each field.
    slices = Dict(name => slice_indices(field; xC=xC, yC=yC, zC=zC, xF=xF, yF=yF, zF=zF)
                  for (name, field) in field_outputs)

    return NetCDFOutputWriter(filename, dataset, outputs, iteration_interval, time_interval,
                              clobber, slices, 0.0, verbose)
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

function save_output_to_netcdf!(nc, model, f::Field, name, time_index)
    data = cpudata(f) # Transfer data to CPU if parent(output) is a CuArray
    nc.dataset[name][:, :, :, time_index] = view(data, nc.slices[name]...)
end

function save_output_to_netcdf!(nc, model, output, name, time_index)
    data = output(model)
    colons = Tuple(Colon() for _ in 1:ndims(data))
    nc.dataset[name][colons..., time_index] = data
end

"""
    write_output(model, OutputWriter)

Writes output to the netcdf file at specified intervals. Increments the `time` dimension
every time an output is written to the file.
"""
function write_output(model, ow::NetCDFOutputWriter)
    ds, verbose, filepath = ow.dataset, ow.verbose, ow.filename

    time_index = length(ds["time"]) + 1
    ds["time"][time_index] = model.clock.time

    if verbose
        @info "Writing to NetCDF: $filepath..."
        @info "Computing NetCDF outputs for time index $(time_index): $(keys(ow.outputs))..."
    end

    if verbose
        # Time and file size before computing any outputs.
        t0, sz0 = time_ns(), filesize(filepath)
    end

    for (name, output) in ow.outputs
        # Time before computing this output.
        verbose && (t0′ = time_ns())

        save_output_to_netcdf!(ow, model, output, name, time_index)

        if verbose
            # Time after computing this output.
            t1′ = time_ns()
            @info "Computing $name done: time=$(prettytime((t1′-t0′)/1e9))"
        end
    end

    sync(ow.dataset)

    if verbose
        # Time and file size after computing and writing all outputs.
        t1, sz1 = time_ns(), filesize(filepath)
        verbose && @info begin
            @sprintf("Writing done: time=%s, size=%s, Δsize=%s",
                    prettytime((t1-t0)/1e9), pretty_filesize(sz1), pretty_filesize(sz1-sz0))
        end
    end

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
    with_halos = false,
       attributes = Dict(),
    slice_keywords...)

    grid = model.grid

    # Allow values to be of Any type as OffsetArrays may get modified below.
    dims = Dict{String,Any}(
        "xC" => with_halos ? grid.xC : collect(xnodes(Cell, grid)),
        "xF" => with_halos ? grid.xF : collect(xnodes(Face, grid)),
        "yC" => with_halos ? grid.yC : collect(ynodes(Cell, grid)),
        "yF" => with_halos ? grid.yF : collect(ynodes(Face, grid)),
        "zC" => with_halos ? grid.zC : collect(znodes(Cell, grid)),
        "zF" => with_halos ? grid.zF : collect(znodes(Face, grid))
    )

    dim_attribs = Dict(
        "xC" => Dict("longname" => "Locations of the cell centers in the x-direction.", "units" => units),
        "xF" => Dict("longname" => "Locations of the cell faces in the x-direction.",   "units" => units),
        "yC" => Dict("longname" => "Locations of the cell centers in the y-direction.", "units" => units),
        "yF" => Dict("longname" => "Locations of the cell faces in the y-direction.",   "units" => units),
        "zC" => Dict("longname" => "Locations of the cell centers in the z-direction.", "units" => units),
        "zF" => Dict("longname" => "Locations of the cell faces in the z-direction.",   "units" => units)
    )

    # Add useful metadata as global attributes
    attributes["date"] = "This file was generated on $(now())."
    attributes["Julia"] = "This file was generated using " * versioninfo_with_gpu()
    attributes["Oceananigans"] = "This file was generated using " * oceananigans_versioninfo()

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

function Base.show(io::IO, ow::NetCDFOutputWriter)
    freq_int = isnothing(ow.iteration_interval) && isnothing(ow.time_interval) ? "" :
               isnothing(ow.iteration_interval) ? "(time_interval=$(ow.time_interval))" :
               isnothing(ow.time_interval)      ? "(iteration_interval=$(ow.iteration_interval))" :
               "(iteration_interval=$(ow.iteration_interval), time_interval=$(ow.time_interval))"

    dims = join([dim * "(" * string(length(ow.dataset[dim])) * "), "
                 for dim in keys(ow.dataset.dim)])[1:end-2]

    print(io, "NetCDFOutputWriter $freq_int: $(ow.filename)\n",
        "├── dimensions: $dims\n",
        "└── $(length(ow.outputs)) outputs: $(keys(ow.outputs))")
end
