using NCDatasets

using Dates: AbstractTime, now

using Oceananigans.Fields

using Oceananigans.Grids: topology, halo_size, all_x_nodes, all_y_nodes, all_z_nodes, parent_index_range
using Oceananigans.Utils: versioninfo_with_gpu, oceananigans_versioninfo, prettykeys
using Oceananigans.TimeSteppers: float_or_date_time
using Oceananigans.Fields: reduced_dimensions, reduced_location, location, validate_indices

dictify(outputs) = outputs
dictify(outputs::NamedTuple) = Dict(string(k) => dictify(v) for (k, v) in zip(keys(outputs), values(outputs)))

xdim(::Type{Face}) = ("xF",)
ydim(::Type{Face}) = ("yF",)
zdim(::Type{Face}) = ("zF",)

xdim(::Type{Center}) = ("xC",)
ydim(::Type{Center}) = ("yC",)
zdim(::Type{Center}) = ("zC",)

xdim(::Type{Nothing}) = ()
ydim(::Type{Nothing}) = ()
zdim(::Type{Nothing}) = ()

netcdf_spatial_dimensions(::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} =
    tuple(xdim(LX)..., ydim(LY)..., zdim(LZ)...)

function default_dimensions(output, grid, indices, with_halos)
    Hx, Hy, Hz = halo_size(grid)
    TX, TY, TZ = topo = topology(grid)

    locs = Dict(
                "xC" => (Center, Center, Center),
                "xF" => (  Face, Center, Center),
                "yC" => (Center, Center, Center),
                "yF" => (Center,   Face, Center),
                "zC" => (Center, Center, Center),
                "zF" => (Center, Center,   Face),
               )

    indices = Dict(name => validate_indices(indices, locs[name], grid) for name in keys(locs))

    if !with_halos
        indices = Dict(name => restrict_to_interior.(indices[name], locs[name], topo, size(grid))
                       for name in keys(locs))
    end

    dims = Dict("xC" => parent(all_x_nodes(Center, grid))[parent_index_range(indices["xC"][1], Center, TX, Hx)],
                "xF" => parent(all_x_nodes(Face,   grid))[parent_index_range(indices["xF"][1],   Face, TX, Hx)],
                "yC" => parent(all_y_nodes(Center, grid))[parent_index_range(indices["yC"][2], Center, TY, Hy)],
                "yF" => parent(all_y_nodes(Face,   grid))[parent_index_range(indices["yF"][2],   Face, TY, Hy)],
                "zC" => parent(all_z_nodes(Center, grid))[parent_index_range(indices["zC"][3], Center, TZ, Hz)],
                "zF" => parent(all_z_nodes(Face,   grid))[parent_index_range(indices["zF"][3],   Face, TZ, Hz)])

    return dims
end


const default_dimension_attributes = Dict(
    "xC"          => Dict("longname" => "Locations of the cell centers in the x-direction.", "units" => "m"),
    "xF"          => Dict("longname" => "Locations of the cell faces in the x-direction.",   "units" => "m"),
    "yC"          => Dict("longname" => "Locations of the cell centers in the y-direction.", "units" => "m"),
    "yF"          => Dict("longname" => "Locations of the cell faces in the y-direction.",   "units" => "m"),
    "zC"          => Dict("longname" => "Locations of the cell centers in the z-direction.", "units" => "m"),
    "zF"          => Dict("longname" => "Locations of the cell faces in the z-direction.",   "units" => "m"),
    "time"        => Dict("longname" => "Time", "units" => "s"),
    "particle_id" => Dict("longname" => "Particle ID")
)

const default_output_attributes = Dict(
    "u" => Dict("longname" => "Velocity in the x-direction", "units" => "m/s"),
    "v" => Dict("longname" => "Velocity in the y-direction", "units" => "m/s"),
    "w" => Dict("longname" => "Velocity in the z-direction", "units" => "m/s"),
    "b" => Dict("longname" => "Buoyancy",                    "units" => "m/s²"),
    "T" => Dict("longname" => "Conservative temperature",    "units" => "°C"),
    "S" => Dict("longname" => "Absolute salinity",           "units" => "g/kg")
)

add_schedule_metadata!(attributes, schedule) = nothing

function add_schedule_metadata!(global_attributes, schedule::IterationInterval)
    global_attributes["schedule"] = "IterationInterval"
    global_attributes["interval"] = schedule.interval
    global_attributes["output iteration interval"] =
        "Output was saved every $(schedule.interval) iteration(s)."

    return nothing
end

function add_schedule_metadata!(global_attributes, schedule::TimeInterval)
    global_attributes["schedule"] = "TimeInterval"
    global_attributes["interval"] = schedule.interval
    global_attributes["output time interval"] =
        "Output was saved every $(prettytime(schedule.interval))."

    return nothing
end

function add_schedule_metadata!(global_attributes, schedule::WallTimeInterval)
    global_attributes["schedule"] = "WallTimeInterval"
    global_attributes["interval"] = schedule.interval
    global_attributes["output time interval"] =
        "Output was saved every $(prettytime(schedule.interval))."

    return nothing
end

function add_schedule_metadata!(global_attributes, schedule::AveragedTimeInterval)
    add_schedule_metadata!(global_attributes, TimeInterval(schedule))

    global_attributes["time_averaging_window"] = schedule.window
    global_attributes["time averaging window"] =
        "Output was time averaged with a window size of $(prettytime(schedule.window))"

    global_attributes["time_averaging_stride"] = schedule.stride
    global_attributes["time averaging stride"] =
        "Output was time averaged with a stride of $(schedule.stride) iteration(s) within the time averaging window."

    return nothing
end

"""
    NetCDFOutputWriter{D, O, I, T, A} <: AbstractOutputWriter

An output writer for writing to NetCDF files.
"""
mutable struct NetCDFOutputWriter{D, O, T, A} <: AbstractOutputWriter
    filepath :: String
    dataset :: D
    outputs :: O
    schedule :: T
    overwrite_existing :: Bool
    array_type :: A
    previous :: Float64
    verbose :: Bool
end

"""
    NetCDFOutputWriter(model, outputs; filename, schedule
                                          dir = ".",
                                   array_type = Array{Float32},
                                      indices = nothing,
                            global_attributes = Dict(),
                            output_attributes = Dict(),
                                   dimensions = Dict(),
                           overwrite_existing = false,
                                  compression = 0,
                                      verbose = false)

Construct a `NetCDFOutputWriter` that writes `(label, output)` pairs in `outputs` (which should
be a `Dict`) to a NetCDF file, where `label` is a string that labels the output and `output` is
either a `Field` (e.g. `model.velocities.u`) or a function `f(model)` that
returns something to be written to disk. Custom output requires the spatial `dimensions` (a
`Dict`) to be manually specified (see examples).

Keyword arguments
=================
- `filename` (required): Descriptive filename. ".nc" is appended to `filename` if ".nc" is not detected.

- `schedule` (required): `AbstractSchedule` that determines when output is saved.

- `dir`: Directory to save output to.

- `array_type`: The array type to which output arrays are converted to prior to saving.
                Default: Array{Float32}.

- `indices`: Tuple of indices of the output variables to include. Default is `(:, :, :)`, which
             includes the full fields.

- `with_halos`: Boolean defining whether or not to include halos in the outputs.

- `global_attributes`: Dict of model properties to save with every file (deafult: `Dict()`)

- `output_attributes`: Dict of attributes to be saved with each field variable (reasonable
                       defaults are provided for velocities, buoyancy, temperature, and salinity;
                       otherwise `output_attributes` *must* be user-provided).

- `dimensions`: A `Dict` of dimension tuples to apply to outputs (required for function outputs)

- `overwrite_existing`: If false, NetCDFOutputWriter will be set to append to `filepath`. If true, NetCDFOutputWriter 
                        will overwrite `filepath` if it exists or create it if it does not. 
                        Default: false. See NCDatasets.jl documentation for more information about its `mode` option.

- `compression`: Determines the compression level of data (0-9, default 0)

Examples
========
Saving the u velocity field and temperature fields, the full 3D fields and surface 2D slices
to separate NetCDF files:

```jldoctest netcdf1
using Oceananigans

grid = RectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1))

model = NonhydrostaticModel(grid=grid, tracers=:c)

simulation = Simulation(model, Δt=12, stop_time=3600)

fields = Dict("u" => model.velocities.u, "c" => model.tracers.c)

simulation.output_writers[:field_writer] =
    NetCDFOutputWriter(model, fields, filename="fields.nc", schedule=TimeInterval(60))

# output
NetCDFOutputWriter scheduled on TimeInterval(1 minute):
├── filepath: ./fields.nc
├── dimensions: zC(16), zF(17), xC(16), yF(16), xF(16), yC(16), time(0)
├── 2 outputs: (c, u)
└── array type: Array{Float32}
```

```jldoctest netcdf1
simulation.output_writers[:surface_slice_writer] =
    NetCDFOutputWriter(model, fields, filename="surface_xy_slice.nc",
                       schedule=TimeInterval(60), indices=(:, :, grid.Nz))

# output
NetCDFOutputWriter scheduled on TimeInterval(1 minute):
├── filepath: ./surface_xy_slice.nc
├── dimensions: zC(1), zF(1), xC(16), yF(16), xF(16), yC(16), time(0)
├── 2 outputs: (c, u)
└── array type: Array{Float32}
```

```jldoctest netcdf1
simulation.output_writers[:averaged_profile_writer] =
    NetCDFOutputWriter(model, fields,
                       filename = "averaged_z_profile.nc",
                       schedule = AveragedTimeInterval(60, window=20),
                       indices = (1, 1, :))

# output
NetCDFOutputWriter scheduled on TimeInterval(1 minute):
├── filepath: ./averaged_z_profile.nc
├── dimensions: zC(16), zF(17), xC(1), yF(1), xF(1), yC(1), time(0)
├── 2 outputs: (c, u) averaged on AveragedTimeInterval(window=20 seconds, stride=1, interval=1 minute)
└── array type: Array{Float32}
```

`NetCDFOutputWriter` also accepts output functions that write scalars and arrays to disk,
provided that their `dimensions` are provided:

```jldoctest
using Oceananigans

grid = RectilinearGrid(size=(16, 16, 16), extent=(1, 2, 3))

model = NonhydrostaticModel(grid=grid)

simulation = Simulation(model, Δt=1.25, stop_iteration=3)

f(model) = model.clock.time^2; # scalar output

g(model) = model.clock.time .* exp.(znodes(Center, grid)) # vector/profile output

h(model) = model.clock.time .* (   sin.(xnodes(Center, grid, reshape=true)[:, :, 1])
                            .*     cos.(ynodes(Face, grid, reshape=true)[:, :, 1])) # xy slice output

outputs = Dict("scalar" => f, "profile" => g, "slice" => h)

dims = Dict("scalar" => (), "profile" => ("zC",), "slice" => ("xC", "yC"))

output_attributes = Dict(
    "scalar"  => Dict("longname" => "Some scalar", "units" => "bananas"),
    "profile" => Dict("longname" => "Some vertical profile", "units" => "watermelons"),
    "slice"   => Dict("longname" => "Some slice", "units" => "mushrooms")
);

global_attributes = Dict("location" => "Bay of Fundy", "onions" => 7)

simulation.output_writers[:things] =
    NetCDFOutputWriter(model, outputs,
                       schedule=IterationInterval(1), filename="things.nc", dimensions=dims, verbose=true,
                       global_attributes=global_attributes, output_attributes=output_attributes)

# output
NetCDFOutputWriter scheduled on IterationInterval(1):
├── filepath: ./things.nc
├── dimensions: zC(16), zF(17), xC(16), yF(16), xF(16), yC(16), time(0)
├── 3 outputs: (profile, slice, scalar)
└── array type: Array{Float32}
```
"""
function NetCDFOutputWriter(model, outputs; filename, schedule,
                                          dir = ".",
                                   array_type = Array{Float32},
                                      indices = (:, :, :),
                                   with_halos = false,
                            global_attributes = Dict(),
                            output_attributes = Dict(),
                                   dimensions = Dict(),
                           overwrite_existing = nothing,
                                  compression = 0,
                                      verbose = false)

    mkpath(dir)
    filename = auto_extension(filename, "nc")
    filepath = joinpath(dir, filename)

    if isnothing(overwrite_existing)
        if isfile(filepath)
            overwrite_existing = false
        else
            overwrite_existing = true
        end
    else

        if isfile(filepath) && !overwrite_existing
            @warn "$filepath already exists and `overwrite_existing = false`. Mode will be set to append to existing file. " *
                  "You might experience errors when writing output if the existing file belonged to a different simulation!"

        elseif isfile(filepath) && overwrite_existing
            @warn "Overwriting existing $filepath."

        end
    end

    mode = overwrite_existing ? "c" : "a"

    # TODO: This call to dictify is only necessary because "dictify" is hacked to help
    # with LagrangianParticles output (see the end of the file).
    # We shouldn't support this in the future; we should require users to 'name' LagrangianParticles output.
    outputs = dictify(outputs)
    outputs = Dict(string(name) => construct_output(outputs[name], model.grid, indices, with_halos) for name in keys(outputs))
    output_attributes = dictify(output_attributes)
    global_attributes = dictify(global_attributes)
    dimensions = dictify(dimensions)

    # Ensure we can add any kind of metadata to the global attributes later by converting to Dict{Any, Any}.
    global_attributes = Dict{Any, Any}(global_attributes)

    # Add useful metadata
    global_attributes["date"] = "This file was generated on $(now())."
    global_attributes["Julia"] = "This file was generated using " * versioninfo_with_gpu()
    global_attributes["Oceananigans"] = "This file was generated using " * oceananigans_versioninfo()

    add_schedule_metadata!(global_attributes, schedule)

    # Convert schedule to TimeInterval and each output to WindowedTimeAverage if
    # schedule::AveragedTimeInterval
    schedule, outputs = time_average_outputs(schedule, outputs, model)

    dims = default_dimensions(outputs, model.grid, indices, with_halos)

    # Open the NetCDF dataset file
    dataset = NCDataset(filepath, mode, attrib=global_attributes)

    # Define variables for each dimension and attributes if this is a new file.
    if mode == "c"
        for (dim_name, dim_array) in dims
            defVar(dataset, dim_name, array_type(dim_array), (dim_name,),
                   compression=compression, attrib=default_dimension_attributes[dim_name])
        end

        # DateTime and TimeDate are both <: AbstractTime
        time_attrib = model.clock.time isa AbstractTime ?
            Dict("longname" => "Time", "units" => "seconds since 2000-01-01 00:00:00") :
            Dict("longname" => "Time", "units" => "seconds")

        # Creates an unlimited dimension "time"
        defDim(dataset, "time", Inf)
        defVar(dataset, "time", eltype(model.grid), ("time",), attrib=time_attrib)

        # Use default output attributes for known outputs if the user has not specified any.
        # Unknown outputs get an empty tuple (no output attributes).
        for c in keys(outputs)
            if !haskey(output_attributes, c)
                output_attributes[c] = c in keys(default_output_attributes) ? default_output_attributes[c] : ()
            end
        end

        for (name, output) in outputs
            attributes = try output_attributes[name]; catch; Dict(); end
            define_output_variable!(dataset, output, name, array_type, compression, attributes, dimensions)
        end

        sync(dataset)
    end

    close(dataset)

    return NetCDFOutputWriter(filepath, dataset, outputs, schedule, overwrite_existing, array_type, 0.0, verbose)
end

#####
##### Variable definition
#####

""" Defines empty variables for 'custom' user-supplied `output`. """
function define_output_variable!(dataset, output, name, array_type, compression, output_attributes, dimensions)
    name ∉ keys(dimensions) && error("Custom output $name needs dimensions!")

    defVar(dataset, name, eltype(array_type), (dimensions[name]..., "time"),
           compression=compression, attrib=output_attributes)

    return nothing
end


""" Defines empty field variable. """
define_output_variable!(dataset, output::AbstractField, name, array_type, compression, output_attributes, dimensions) =
    defVar(dataset, name, eltype(array_type),
           (netcdf_spatial_dimensions(output)..., "time"),
           compression=compression, attrib=output_attributes)

""" Defines empty field variable for `WindowedTimeAverage`s over fields. """
define_output_variable!(dataset, output::WindowedTimeAverage{<:AbstractField}, args...) =
    define_output_variable!(dataset, output.operand, args...)


#####
##### Write output
#####

Base.open(nc::NetCDFOutputWriter) = NCDataset(nc.filepath, "a")
Base.close(nc::NetCDFOutputWriter) = close(nc.dataset)

function save_output!(ds, output, model, ow, time_index, name)
    data = fetch_and_convert_output(output, model, ow)
    data = drop_output_dims(output, data)
    colons = Tuple(Colon() for _ in 1:ndims(data))
    ds[name][colons..., time_index] = data
    return nothing
end

function save_output!(ds, output::LagrangianParticles, model, ow, time_index, name)
    data = fetch_and_convert_output(output, model, ow)
    for (particle_field, vals) in pairs(data)
        ds[string(particle_field)][:, time_index] = vals
    end

    return nothing
end

"""
    write_output!(output_writer, model)

Writes output to netcdf file `output_writer.filepath` at specified intervals. Increments the `time` dimension
every time an output is written to the file.
"""
function write_output!(ow::NetCDFOutputWriter, model)
    ow.dataset = open(ow)

    ds, verbose, filepath = ow.dataset, ow.verbose, ow.filepath

    time_index = length(ds["time"]) + 1
    ds["time"][time_index] = float_or_date_time(model.clock.time)

    if verbose
        @info "Writing to NetCDF: $filepath..."
        @info "Computing NetCDF outputs for time index $(time_index): $(keys(ow.outputs))..."

        # Time and file size before computing any outputs.
        t0, sz0 = time_ns(), filesize(filepath)
    end

    for (name, output) in ow.outputs
        # Time before computing this output.
        verbose && (t0′ = time_ns())

        save_output!(ds, output, model, ow, time_index, name)

        if verbose
            # Time after computing this output.
            t1′ = time_ns()
            @info "Computing $name done: time=$(prettytime((t1′-t0′) / 1e9))"
        end
    end

    if verbose
        # Time and file size after computing and writing all outputs.
        t1, sz1 = time_ns(), filesize(filepath)
        verbose && @info begin
            @sprintf("Writing done: time=%s, size=%s, Δsize=%s",
                    prettytime((t1-t0)/1e9), pretty_filesize(sz1), pretty_filesize(sz1-sz0))
        end
    end

    sync(ds)
    close(ow)

    return nothing
end

drop_output_dims(output, data) = data # fallback
drop_output_dims(output::Field, data) = dropdims(data, dims=reduced_dimensions(output))
drop_output_dims(output::WindowedTimeAverage{<:Field}, data) = dropdims(data, dims=reduced_dimensions(output.operand))

#####
##### Show
#####

Base.summary(ow::NetCDFOutputWriter) =
    string("NetCDFOutputWriter writing ", prettykeys(ow.outputs), " to ", ow.filepath, " on ", summary(ow.schedule))

function Base.show(io::IO, ow::NetCDFOutputWriter)
    dims = NCDataset(ow.filepath, "r") do ds
        join([dim * "(" * string(length(ds[dim])) * "), "
              for dim in keys(ds.dim)])[1:end-2]
    end

    averaging_schedule = output_averaging_schedule(ow)
    Noutputs = length(ow.outputs)

    print(io, "NetCDFOutputWriter scheduled on $(summary(ow.schedule)):", '\n',
              "├── filepath: ", ow.filepath, '\n',
              "├── dimensions: $dims", '\n',
              "├── $Noutputs outputs: ", prettykeys(ow.outputs), show_averaging_schedule(averaging_schedule), '\n',
              "└── array type: ", show_array_type(ow.array_type))
end

#####
##### Support / hacks for Lagrangian particles output
#####

""" Defines empty variable for particle trackting. """
function define_output_variable!(dataset, output::LagrangianParticles, name, array_type, compression, output_attributes, dimensions)
    particle_fields = eltype(output.properties) |> fieldnames .|> string
    for particle_field in particle_fields
        defVar(dataset, particle_field, eltype(array_type),
               ("particle_id", "time"), compression=compression)
    end
end

dictify(outputs::LagrangianParticles) = Dict("particles" => outputs)

default_dimensions(outputs::Dict{String,<:LagrangianParticles}, grid, indices, with_halos) =
    Dict("particle_id" => collect(1:length(outputs["particles"])))
