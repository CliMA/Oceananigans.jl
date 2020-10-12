using NCDatasets

using Oceananigans.Fields

using Dates: now
using Oceananigans.Grids: topology, halo_size
using Oceananigans.Utils: validate_intervals, versioninfo_with_gpu, oceananigans_versioninfo

xdim(::Type{Face}) = "xF"
ydim(::Type{Face}) = "yF"
zdim(::Type{Face}) = "zF"

xdim(::Type{Cell}) = "xC"
ydim(::Type{Cell}) = "yC"
zdim(::Type{Cell}) = "zC"

netcdf_spatial_dimensions(::Field{LX, LY, LZ}) where {LX, LY, LZ} = xdim(LX), ydim(LY), zdim(LZ)

const default_dimension_attributes = Dict(
    "xC" => Dict("longname" => "Locations of the cell centers in the x-direction.", "units" => "m"),
    "xF" => Dict("longname" => "Locations of the cell faces in the x-direction.",   "units" => "m"),
    "yC" => Dict("longname" => "Locations of the cell centers in the y-direction.", "units" => "m"),
    "yF" => Dict("longname" => "Locations of the cell faces in the y-direction.",   "units" => "m"),
    "zC" => Dict("longname" => "Locations of the cell centers in the z-direction.", "units" => "m"),
    "zF" => Dict("longname" => "Locations of the cell faces in the z-direction.",   "units" => "m")
)

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
mutable struct NetCDFOutputWriter{D, O, I, T, S, A} <: AbstractOutputWriter
              filepath :: String
               dataset :: D
               outputs :: O
    iteration_interval :: I
         time_interval :: T
                  mode :: String
          field_slicer :: S
            array_type :: A
              previous :: Float64
               verbose :: Bool
end

"""
    NetCDFOutputWriter(model, outputs; filepath, iteration_interval=nothing, time_interval=nothing,
                       global_attributes=Dict(), output_attributes=Dict(), dimensions=Dict(),
                       mode="c", compression=0, with_halos=false, verbose=false, slice_kwargs...)

Construct a `NetCDFOutputWriter` that writes `(label, output)` pairs in `outputs` (which should
be a `Dict`) to a NetCDF file, where `label` is a string that labels the output and `output` is
either a field from the model (e.g. `model.velocities.u`) or a function `f(model)` that returns
something to be written to disk. Custom output requires the spatial `dimensions` to be manually
specified.

Keyword arguments
=================
- `iteration_interval`: Save output every `n` model iterations.

- `time_interval`: Save output every `t` units of model clock time.

- `filepath`: Filepath to save output to.

- `global_attributes`: Dict of model properties to save with every file (deafult: `Dict()`)

- `output_attributes`: Dict of attributes to be saved with each field variable (reasonable
  defaults are provided for velocities, buoyancy, temperature, and salinity).

- `dimensions`: A `Dict` of dimension tuples to apply to outputs (useful for function outputs
  as field dimensions can be inferred).

- `with_halos`: Include the halo regions in the grid coordinates and output fields
  (default: `false`).

- `mode`: "a" (for append) and "c" (for clobber or create). Default: "c". See NCDatasets.jl
  documentation for more information on the `mode` option.

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
    NetCDFOutputWriter(model, fields, filepath="fields.nc", time_interval=60)

# output
NetCDFOutputWriter (time_interval=60): fields.nc
├── dimensions: zC(16), zF(17), xC(16), yF(16), xF(16), yC(16), time(0)
└── 2 outputs: ["T", "u"]
```

```jldoctest netcdf1
simulation.output_writers[:surface_slice_writer] =
    NetCDFOutputWriter(model, fields, filepath="surface_xy_slice.nc",
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
                       iteration_interval=1, filepath="things.nc", dimensions=dims, verbose=true,
                       global_attributes=global_attributes, output_attributes=output_attributes)

# output
NetCDFOutputWriter (iteration_interval=1): things.nc
├── dimensions: zC(16), zF(17), xC(16), yF(16), xF(16), yC(16), time(0)
└── 3 outputs: ["profile", "slice", "scalar"]
```
"""
function NetCDFOutputWriter(model, outputs; filepath,
                               iteration_interval = nothing,
                                    time_interval = nothing,
                            time_averaging_window = nothing,
                            time_averaging_stride = 1,
                                       array_type = Array{Float32},
                                     field_slicer = FieldSlicer(),
                                global_attributes = Dict(),
                                output_attributes = Dict(),
                                       dimensions = Dict(),
                                             mode = "c",
                                      compression = 0,
                                          verbose = false)

    validate_intervals(iteration_interval, time_interval)

    # Convert each output to WindowedTimeAverage if time_averaging_window is specified
    if !isnothing(time_averaging_window)

        !isnothing(iteration_interval) && error("Cannot specify iteration_interval with time_averaging_window.")

        outputs = Dict(name => WindowedTimeAverage(outputs[name], model, time_interval = time_interval,
                                               time_window = time_averaging_window, stride = time_averaging_stride,
                                               field_slicer = field_slicer)
                   for name in keys(outputs))
    end

    # Ensure we can add any kind of metadata to the global attributes later by converting to pairs of type {Any, Any}.
    global_attributes = Dict{Any, Any}(k => v for (k, v) in global_attributes)

    # Add useful metadata
    global_attributes["date"] = "This file was generated on $(now())."
    global_attributes["Julia"] = "This file was generated using " * versioninfo_with_gpu()
    global_attributes["Oceananigans"] = "This file was generated using " * oceananigans_versioninfo()

    # Add useful metadata about intervals and time averaging
    if isnothing(time_averaging_window)
        if !isnothing(iteration_interval)
            global_attributes["iteration_interval"] = iteration_interval
            global_attributes["output iteration interval"] =
                "Output was saved every $iteration_interval iteration(s)."
        end

        if !isnothing(time_interval)
            global_attributes["time_interval"] = time_interval
            global_attributes["output time interval"] =
                "Output was saved every $(prettytime(time_interval))."
        end
    else
        global_attributes["time_interval"] = time_interval
        global_attributes["output time interval"] =
            "Output was time averaged and saved every $(prettytime(time_interval))."

        global_attributes["time_averaging_window"] = time_averaging_window
        global_attributes["time averaging window"] =
            "Output was time averaged with a window size of $(prettytime(time_averaging_window))"

        if time_averaging_stride != 1
            global_attributes["time_averaging_stride"] = time_averaging_stride
            global_attributes["time averaging stride"] =
                "Output was time averaged with a stride of $time_averaging_stride iteration(s) within the time averaging window."	
        end
    end

    grid = model.grid
    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)
    TX, TY, TZ = topology(grid)

    dims = Dict(
        "xC" => grid.xC.parent[parent_slice_indices(Cell, TX, Nx, Hx, field_slicer.i, field_slicer.with_halos)],
        "xF" => grid.xF.parent[parent_slice_indices(Face, TX, Nx, Hx, field_slicer.i, field_slicer.with_halos)],
        "yC" => grid.yC.parent[parent_slice_indices(Cell, TY, Ny, Hy, field_slicer.j, field_slicer.with_halos)],
        "yF" => grid.yF.parent[parent_slice_indices(Face, TY, Ny, Hy, field_slicer.j, field_slicer.with_halos)],
        "zC" => grid.zC.parent[parent_slice_indices(Cell, TZ, Nz, Hz, field_slicer.k, field_slicer.with_halos)],
        "zF" => grid.zF.parent[parent_slice_indices(Face, TZ, Nz, Hz, field_slicer.k, field_slicer.with_halos)]
    )

    # Open the NetCDF dataset file
    dataset = Dataset(filepath, mode, attrib=global_attributes)

    # Define variables for each dimension and attributes if this is a new file.
    if mode == "c"
        for (dim_name, dim_array) in dims
            defVar(dataset, dim_name, dim_array, (dim_name,),
                   compression=compression, attrib=default_dimension_attributes[dim_name])
        end

        # Creates an unlimited dimension "time"
        defDim(dataset, "time", Inf)
        defVar(dataset, "time", typeof(model.clock.time), ("time",))

        # Use default output attributes for known outputs if the user has not specified any.
        # Unknown outputs get an empty tuple (no output attributes).
        for c in keys(outputs)
            if !haskey(output_attributes, c)
                output_attributes[c] = c in keys(default_output_attributes) ? default_output_attributes[c] : ()
            end
        end

        # Initiates empty variables for fields from the user-supplied `outputs`.
        for (name, output) in outputs
            if output isa Field
                defVar(dataset, name, eltype(array_type), (netcdf_spatial_dimensions(output)..., "time"),
                       compression=compression, attrib=output_attributes[name])
            elseif output isa WindowedTimeAverage && output.operand isa Field
                defVar(dataset, name, eltype(array_type), (netcdf_spatial_dimensions(output.operand)..., "time"),
                       compression=compression, attrib=output_attributes[name])
            else
                name ∉ keys(dimensions) && error("Custom output $name needs dimensions!")

                defVar(dataset, name, eltype(array_type), (dimensions[name]..., "time"),
                       compression=compression, attrib=output_attributes[name])
            end
        end

        sync(dataset)
    end

    return NetCDFOutputWriter(filepath, dataset, outputs, iteration_interval, time_interval,
                              mode, field_slicer, array_type, 0.0, verbose)
end

Base.open(ow::NetCDFOutputWriter) = Dataset(ow.filepath, "a")
Base.close(ow::NetCDFOutputWriter) = close(ow.dataset)

"""
    write_output!(output_writer, model)

Writes output to netcdf file `output_writer.filepath` at specified intervals. Increments the `time` dimension
every time an output is written to the file.
"""
function write_output!(ow::NetCDFOutputWriter, model)
    ds, verbose, filepath = ow.dataset, ow.verbose, ow.filepath

    time_index = length(ds["time"]) + 1
    ds["time"][time_index] = model.clock.time

    if verbose
        @info "Writing to NetCDF: $filepath..."
        @info "Computing NetCDF outputs for time index $(time_index): $(keys(ow.outputs))..."

        # Time and file size before computing any outputs.
        t0, sz0 = time_ns(), filesize(filepath)
    end

    for (name, output) in ow.outputs
        # Time before computing this output.
        verbose && (t0′ = time_ns())

        data = fetch_and_convert_output(output, model, ow)

        if output isa AveragedField
            data = dropdims(data, dims=output.dims)
        elseif output isa WindowedTimeAverage && output.operand isa AveragedField
            data = dropdims(data, dims=output.operand.dims)
        end

        colons = Tuple(Colon() for _ in 1:ndims(data))
        ds[name][colons..., time_index] = data

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

function Base.show(io::IO, ow::NetCDFOutputWriter)
    freq_int = isnothing(ow.iteration_interval) && isnothing(ow.time_interval) ? "" :
               isnothing(ow.iteration_interval) ? "(time_interval=$(ow.time_interval))" :
               isnothing(ow.time_interval)      ? "(iteration_interval=$(ow.iteration_interval))" :
               "(iteration_interval=$(ow.iteration_interval), time_interval=$(ow.time_interval))"

    dims = join([dim * "(" * string(length(ow.dataset[dim])) * "), "
                 for dim in keys(ow.dataset.dim)])[1:end-2]

    print(io, "NetCDFOutputWriter $freq_int: $(ow.filepath)\n",
        "├── dimensions: $dims\n",
        "└── $(length(ow.outputs)) outputs: $(keys(ow.outputs))")
end
