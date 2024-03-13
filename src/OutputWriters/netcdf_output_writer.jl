using NCDatasets

using Dates: AbstractTime, now

using Oceananigans.Fields

using Oceananigans.Grids: AbstractCurvilinearGrid, RectilinearGrid, topology, halo_size, parent_index_range
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Utils: versioninfo_with_gpu, oceananigans_versioninfo, prettykeys
using Oceananigans.TimeSteppers: float_or_date_time
using Oceananigans.Fields: reduced_dimensions, reduced_location, location, validate_indices

mutable struct NetCDFOutputWriter{D, O, T, A} <: AbstractOutputWriter
    filepath :: String
    dataset :: D
    outputs :: O
    schedule :: T
    array_type :: A
    indices :: Tuple
    with_halos :: Bool
    global_attributes :: Dict
    output_attributes :: Dict
    dimensions :: Dict
    overwrite_existing :: Bool
    deflatelevel :: Int
    part :: Int
    max_filesize :: Float64
    verbose :: Bool
end

ext(::Type{NetCDFOutputWriter}) = ".nc"

dictify(outputs) = outputs
dictify(outputs::NamedTuple) = Dict(string(k) => dictify(v) for (k, v) in zip(keys(outputs), values(outputs)))

xdim(::Face) = tuple("xF")
ydim(::Face) = tuple("yF")
zdim(::Face) = tuple("zF")

xdim(::Center) = tuple("xC")
ydim(::Center) = tuple("yC")
zdim(::Center) = tuple("zC")

xdim(::Nothing) = tuple()
ydim(::Nothing) = tuple()
zdim(::Nothing) = tuple()

netcdf_spatial_dimensions(::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} =
    tuple(xdim(instantiate(LX))..., ydim(instantiate(LY))..., zdim(instantiate(LZ))...)

native_dimensions_for_netcdf_output(grid, indices, TX, TY, TZ, Hx, Hy, Hz) =
    Dict("xC" => parent(xnodes(grid, Center(); with_halos=true))[parent_index_range(indices["xC"][1], Center(), TX(), Hx)],
         "xF" => parent(xnodes(grid, Face();   with_halos=true))[parent_index_range(indices["xF"][1],   Face(), TX(), Hx)],
         "yC" => parent(ynodes(grid, Center(); with_halos=true))[parent_index_range(indices["yC"][2], Center(), TY(), Hy)],
         "yF" => parent(ynodes(grid, Face();   with_halos=true))[parent_index_range(indices["yF"][2],   Face(), TY(), Hy)],
         "zC" => parent(znodes(grid, Center(); with_halos=true))[parent_index_range(indices["zC"][3], Center(), TZ(), Hz)],
         "zF" => parent(znodes(grid, Face();   with_halos=true))[parent_index_range(indices["zF"][3],   Face(), TZ(), Hz)])

native_dimensions_for_netcdf_output(grid::AbstractCurvilinearGrid, indices, TX, TY, TZ, Hx, Hy, Hz) =
    Dict("xC" => parent(λnodes(grid, Center(); with_halos=true))[parent_index_range(indices["xC"][1], Center(), TX(), Hx)],
         "xF" => parent(λnodes(grid, Face();   with_halos=true))[parent_index_range(indices["xF"][1],   Face(), TX(), Hx)],
         "yC" => parent(φnodes(grid, Center(); with_halos=true))[parent_index_range(indices["yC"][2], Center(), TY(), Hy)],
         "yF" => parent(φnodes(grid, Face();   with_halos=true))[parent_index_range(indices["yF"][2],   Face(), TY(), Hy)],
         "zC" => parent(znodes(grid, Center(); with_halos=true))[parent_index_range(indices["zC"][3], Center(), TZ(), Hz)],
         "zF" => parent(znodes(grid, Face();   with_halos=true))[parent_index_range(indices["zF"][3],   Face(), TZ(), Hz)])

function default_dimensions(output, grid, indices, with_halos)
    Hx, Hy, Hz = halo_size(grid)
    TX, TY, TZ = topo = topology(grid)

    locs = Dict("xC" => (Center(), Center(), Center()),
                "xF" => (Face(),   Center(), Center()),
                "yC" => (Center(), Center(), Center()),
                "yF" => (Center(), Face(),   Center()),
                "zC" => (Center(), Center(), Center()),
                "zF" => (Center(), Center(), Face()  ))

    topo = map(instantiate, topology(grid))

    indices = Dict(name => validate_indices(indices, locs[name], grid) for name in keys(locs))

    if !with_halos
        indices = Dict(name => restrict_to_interior.(indices[name], locs[name], topo, size(grid))
                       for name in keys(locs))
    end

    return native_dimensions_for_netcdf_output(grid, indices, TX, TY, TZ, Hx, Hy, Hz)
end

const default_rectilinear_dimension_attributes = Dict(
    "xC"          => Dict("long_name" => "Locations of the cell centers in the x-direction.", "units" => "m"),
    "xF"          => Dict("long_name" => "Locations of the cell faces in the x-direction.",   "units" => "m"),
    "yC"          => Dict("long_name" => "Locations of the cell centers in the y-direction.", "units" => "m"),
    "yF"          => Dict("long_name" => "Locations of the cell faces in the y-direction.",   "units" => "m"),
    "zC"          => Dict("long_name" => "Locations of the cell centers in the z-direction.", "units" => "m"),
    "zF"          => Dict("long_name" => "Locations of the cell faces in the z-direction.",   "units" => "m"),
    "time"        => Dict("long_name" => "Time", "units" => "s"),
    "particle_id" => Dict("long_name" => "Particle ID")
)

const default_curvilinear_dimension_attributes = Dict(
    "xC"          => Dict("long_name" => "Locations of the cell centers in the λ-direction.", "units" => "degrees"),
    "xF"          => Dict("long_name" => "Locations of the cell faces in the λ-direction.",   "units" => "degrees"),
    "yC"          => Dict("long_name" => "Locations of the cell centers in the φ-direction.", "units" => "degrees"),
    "yF"          => Dict("long_name" => "Locations of the cell faces in the φ-direction.",   "units" => "degrees"),
    "zC"          => Dict("long_name" => "Locations of the cell centers in the z-direction.", "units" => "m"),
    "zF"          => Dict("long_name" => "Locations of the cell faces in the z-direction.",   "units" => "m"),
    "time"        => Dict("long_name" => "Time", "units" => "s"),
    "particle_id" => Dict("long_name" => "Particle ID")
)

const default_output_attributes = Dict(
    "u" => Dict("long_name" => "Velocity in the x-direction", "units" => "m/s"),
    "v" => Dict("long_name" => "Velocity in the y-direction", "units" => "m/s"),
    "w" => Dict("long_name" => "Velocity in the z-direction", "units" => "m/s"),
    "b" => Dict("long_name" => "Buoyancy",                    "units" => "m/s²"),
    "T" => Dict("long_name" => "Conservative temperature",    "units" => "°C"),
    "S" => Dict("long_name" => "Absolute salinity",           "units" => "g/kg")
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
    NetCDFOutputWriter(model, outputs; filename, schedule
                                          dir = ".",
                                   array_type = Array{Float64},
                                      indices = nothing,
                                   with_halos = false,
                            global_attributes = Dict(),
                            output_attributes = Dict(),
                                   dimensions = Dict(),
                           overwrite_existing = false,
                                 deflatelevel = 0,
                                 max_filesize = Inf,
                                      verbose = false)

Construct a `NetCDFOutputWriter` that writes `(label, output)` pairs in `outputs` (which should
be a `Dict`) to a NetCDF file, where `label` is a string that labels the output and `output` is
either a `Field` (e.g. `model.velocities.u`) or a function `f(model)` that
returns something to be written to disk. Custom output requires the spatial `dimensions` (a
`Dict`) to be manually specified (see examples).

Keyword arguments
=================

## Filenaming

- `filename` (required): Descriptive filename. `".nc"` is appended to `filename` if `filename` does
                         not end in `".nc"`.

- `dir`: Directory to save output to.

## Output frequency and time-averaging

- `schedule` (required): `AbstractSchedule` that determines when output is saved.

## Slicing and type conversion prior to output

- `indices`: Tuple of indices of the output variables to include. Default is `(:, :, :)`, which
             includes the full fields.

- `with_halos`: Boolean defining whether or not to include halos in the outputs. Default: `false`.
                Note, that to postprocess saved output (e.g., compute derivatives, etc)
                information about the boundary conditions is often crucial. In that case
                you might need to set `with_halos = true`.

- `array_type`: The array type to which output arrays are converted to prior to saving.
                Default: `Array{Float64}`.

- `dimensions`: A `Dict` of dimension tuples to apply to outputs (required for function outputs).

## File management

- `overwrite_existing`: If `false`, `NetCDFOutputWriter` will be set to append to `filepath`. If `true`,
                        `NetCDFOutputWriter` will overwrite `filepath` if it exists or create it if not.
                        Default: `false`. See [NCDatasets.jl documentation](https://alexander-barth.github.io/NCDatasets.jl/stable/)
                        for more information about its `mode` option.

- `deflatelevel`: Determines the NetCDF compression level of data (integer 0-9; 0 (default) means no compression
                  and 9 means maximum compression). See [NCDatasets.jl documentation](https://alexander-barth.github.io/NCDatasets.jl/stable/variables/#Creating-a-variable)
                  for more information.

- `max_filesize`: The writer will stop writing to the output file once the file size exceeds `max_filesize`,
                  and write to a new one with a consistent naming scheme ending in `part1`, `part2`, etc.
                  Defaults to `Inf`.

## Miscellaneous keywords

- `global_attributes`: Dict of model properties to save with every file. Default: `Dict()`.

- `output_attributes`: Dict of attributes to be saved with each field variable (reasonable
                       defaults are provided for velocities, buoyancy, temperature, and salinity;
                       otherwise `output_attributes` *must* be user-provided).

Examples
========

Saving the ``u`` velocity field and temperature fields, the full 3D fields and surface 2D slices
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
└── array type: Array{Float64}
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
└── array type: Array{Float64}
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
└── array type: Array{Float64}
```

`NetCDFOutputWriter` also accepts output functions that write scalars and arrays to disk,
provided that their `dimensions` are provided:

```jldoctest
using Oceananigans

Nx, Ny, Nz = 16, 16, 16

grid = RectilinearGrid(size=(Nx, Ny, Nz), extent=(1, 2, 3))

model = NonhydrostaticModel(grid=grid)

simulation = Simulation(model, Δt=1.25, stop_iteration=3)

f(model) = model.clock.time^2; # scalar output

g(model) = model.clock.time .* exp.(znodes(Center, grid)) # vector/profile output

xC, yF = xnodes(grid, Center()), ynodes(grid, Face())

XC = [xC[i] for i in 1:Nx, j in 1:Ny]
YF = [yF[j] for i in 1:Nx, j in 1:Ny]

h(model) = @. model.clock.time * sin(XC) * cos(YF) # xy slice output

outputs = Dict("scalar" => f, "profile" => g, "slice" => h)

dims = Dict("scalar" => (), "profile" => ("zC",), "slice" => ("xC", "yC"))

output_attributes = Dict(
    "scalar"  => Dict("long_name" => "Some scalar", "units" => "bananas"),
    "profile" => Dict("long_name" => "Some vertical profile", "units" => "watermelons"),
    "slice"   => Dict("long_name" => "Some slice", "units" => "mushrooms")
)

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
└── array type: Array{Float64}
```
"""
function NetCDFOutputWriter(model, outputs; filename, schedule,
                                          dir = ".",
                                   array_type = Array{Float64},
                                      indices = (:, :, :),
                                   with_halos = false,
                            global_attributes = Dict(),
                            output_attributes = Dict(),
                                   dimensions = Dict(),
                           overwrite_existing = nothing,
                                 deflatelevel = 0,
                                         part = 1,
                                 max_filesize = Inf,
                                      verbose = false)
    mkpath(dir)
    filename = auto_extension(filename, ".nc")
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

    dataset, outputs = initialize_nc_file!(filepath, outputs, schedule, array_type, indices, with_halos, global_attributes, output_attributes, dimensions, overwrite_existing, deflatelevel, model)

    return NetCDFOutputWriter(filepath,
                              dataset,
                              outputs,
                              schedule,
                              array_type,
                              indices,
                              with_halos,
                              global_attributes,
                              output_attributes,
                              dimensions,
                              overwrite_existing,
                              deflatelevel,
                              part,
                              max_filesize,
                              verbose)
end

get_default_dimension_attributes(::RectilinearGrid) =
    default_rectilinear_dimension_attributes

get_default_dimension_attributes(::AbstractCurvilinearGrid) =
    default_curvilinear_dimension_attributes

get_default_dimension_attributes(grid::ImmersedBoundaryGrid) =
    get_default_dimension_attributes(grid.underlying_grid)

#####
##### Variable definition
#####

""" Defines empty variables for 'custom' user-supplied `output`. """
function define_output_variable!(dataset, output, name, array_type, deflatelevel, output_attributes, dimensions)
    name ∉ keys(dimensions) && error("Custom output $name needs dimensions!")

    defVar(dataset, name, eltype(array_type), (dimensions[name]..., "time"),
           deflatelevel=deflatelevel, attrib=output_attributes)

    return nothing
end


""" Defines empty field variable. """
define_output_variable!(dataset, output::AbstractField, name, array_type, deflatelevel, output_attributes, dimensions) =
    defVar(dataset, name, eltype(array_type),
           (netcdf_spatial_dimensions(output)..., "time"),
           deflatelevel=deflatelevel, attrib=output_attributes)

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
    ds[name][colons..., time_index:time_index] = data
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
    # TODO allow user to split by number of snapshots, rathern than filesize.
    # Start a new file if the filesize exceeds max_filesize
    filesize(ow.filepath) ≥ ow.max_filesize && start_next_file(model, ow)

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

    print(io, "NetCDFOutputWriter scheduled on $(summary(ow.schedule)):", "\n",
              "├── filepath: ", ow.filepath, "\n",
              "├── dimensions: $dims", "\n",
              "├── $Noutputs outputs: ", prettykeys(ow.outputs), show_averaging_schedule(averaging_schedule), "\n",
              "└── array type: ", show_array_type(ow.array_type))
end

#####
##### Support / hacks for Lagrangian particles output
#####

""" Defines empty variable for particle trackting. """
function define_output_variable!(dataset, output::LagrangianParticles, name, array_type, deflatelevel, output_attributes, dimensions)
    particle_fields = eltype(output.properties) |> fieldnames .|> string
    for particle_field in particle_fields
        defVar(dataset, particle_field, eltype(array_type),
               ("particle_id", "time"), deflatelevel=deflatelevel)
    end
end

dictify(outputs::LagrangianParticles) = Dict("particles" => outputs)

default_dimensions(outputs::Dict{String,<:LagrangianParticles}, grid, indices, with_halos) =
    Dict("particle_id" => collect(1:length(outputs["particles"])))

function start_next_file(model, ow::NetCDFOutputWriter)
    verbose = ow.verbose
    sz = filesize(ow.filepath)
    verbose && @info begin
        "Filesize $(pretty_filesize(sz)) has exceeded maximum file size $(pretty_filesize(ow.max_filesize))."
    end

    if ow.part == 1
        part1_path = replace(ow.filepath, r".nc$" => "_part1.nc")
        verbose && @info "Renaming first part: $(ow.filepath) -> $part1_path"
        mv(ow.filepath, part1_path, force=ow.overwrite_existing)
        ow.filepath = part1_path
    end

    ow.part += 1
    ow.filepath = replace(ow.filepath, r"part\d+.nc$" => "part" * string(ow.part) * ".nc")
    ow.overwrite_existing && isfile(ow.filepath) && rm(ow.filepath, force=true)
    verbose && @info "Now writing to: $(ow.filepath)"

    initialize_nc_file!(ow, model)
    
    return nothing
end

function initialize_nc_file!(filepath,
                             outputs,
                             schedule,
                             array_type,
                             indices,
                             with_halos,
                             global_attributes,
                             output_attributes,
                             dimensions,
                             overwrite_existing,
                             deflatelevel,
                             model)

    mode = overwrite_existing ? "c" : "a"

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

    default_dimension_attributes = get_default_dimension_attributes(model.grid)

    # Define variables for each dimension and attributes if this is a new file.
    if mode == "c"
        for (dim_name, dim_array) in dims
            defVar(dataset, dim_name, array_type(dim_array), (dim_name,),
                   deflatelevel=deflatelevel, attrib=default_dimension_attributes[dim_name])
        end

        # DateTime and TimeDate are both <: AbstractTime
        time_attrib = model.clock.time isa AbstractTime ?
            Dict("long_name" => "Time", "units" => "seconds since 2000-01-01 00:00:00") :
            Dict("long_name" => "Time", "units" => "seconds")

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
            define_output_variable!(dataset, output, name, array_type, deflatelevel, attributes, dimensions)
        end

        sync(dataset)
    end

    close(dataset)
    return dataset, outputs
end

initialize_nc_file!(ow::NetCDFOutputWriter, model) =
    initialize_nc_file!(ow.filepath,
                        ow.outputs,
                        ow.schedule,
                        ow.array_type,
                        ow.indices,
                        ow.with_halos,
                        ow.global_attributes,
                        ow.output_attributes,
                        ow.dimensions,
                        ow.overwrite_existing,
                        ow.deflatelevel,
                        model)
