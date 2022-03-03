using Printf
using JLD2
using Oceananigans.Utils
using Oceananigans.Models
using Oceananigans.Utils: TimeInterval, pretty_filesize, prettykeys
using Oceananigans.Fields: boundary_conditions, indices

default_included_properties(::NonhydrostaticModel) = [:grid, :coriolis, :buoyancy, :closure]
default_included_properties(::ShallowWaterModel) = [:grid, :coriolis, :closure]
default_included_properties(::HydrostaticFreeSurfaceModel) = [:grid, :coriolis, :buoyancy, :closure]

"""
    JLD2OutputWriter{I, T, O, IF, IN, KW} <: AbstractOutputWriter

An output writer for writing to JLD2 files.
"""
mutable struct JLD2OutputWriter{O, T, D, IF, IN, KW} <: AbstractOutputWriter
    filepath :: String
    outputs :: O
    schedule :: T
    array_type :: D
    init :: IF
    including :: IN
    part :: Int
    max_filesize :: Float64
    force :: Bool
    verbose :: Bool
    jld2_kw :: KW
end

noinit(args...) = nothing

"""
    JLD2OutputWriter(model, outputs; prefix, schedule,
                              dir = ".",
                          indices = (:, :, :),
                       with_halos = false,
                       array_type = Array{Float32},
                     max_filesize = Inf,
                            force = false,
                             init = noinit,
                        including = [:grid, :coriolis, :buoyancy, :closure],
                          verbose = false,
                             part = 1,
                          jld2_kw = Dict{Symbol, Any}())

Construct a `JLD2OutputWriter` for an Oceananigans `model` that writes `label, output` pairs
in `outputs` to a JLD2 file.

The argument `outputs` may be a `Dict` or `NamedTuple`. The keys of `outputs` are symbols or
strings that "name" output data. The values of `outputs` are either `AbstractField`s, objects that
are called with the signature `output(model)`, or `WindowedTimeAverage`s of `AbstractFields`s,
functions, or callable objects.

Keyword arguments
=================

  ## Filenaming

  - `prefix` (required): Descriptive filename prefixed to all output files.

  - `dir`: Directory to save output to.
           Default: "." (current working directory).

  ## Output frequency and time-averaging

  - `schedule` (required): `AbstractSchedule` that determines when output is saved.

  ## Slicing and type conversion prior to output

  - `indices`: Specifies the indices to write to disk with a `Tuple` of `Colon`, `UnitRange`,
               or `Int` elements. Defaults `(:, :, :)` or "all indices". If `!with_halos`,
               halo regions are removed from `indices`. For example, `indices = (:, :, 1)`
               will save xy-slices of the bottom-most index.

  - `with_halos` (Bool): Whether or not to slice halo regions from fields before writing output.

  - `array_type`: The array type to which output arrays are converted to prior to saving.
                  Default: `Array{Float32}`.

  ## File management

  - `max_filesize`: The writer will stop writing to the output file once the file size exceeds `max_filesize`,
                    and write to a new one with a consistent naming scheme ending in `part1`, `part2`, etc.
                    Defaults to `Inf`.

  - `force`: Remove existing files if their filenames conflict.
             Default: `false`.

  ## Output file metadata management

  - `init`: A function of the form `init(file, model)` that runs when a JLD2 output file is initialized.
            Default: `noinit(args...) = nothing`.

  - `including`: List of model properties to save with every file.
                 Default: `[:grid, :coriolis, :buoyancy, :closure]`

  ## Miscellaneous keywords

  - `verbose`: Log what the output writer is doing with statistics on compute/write times and file sizes.
               Default: `false`.

  - `part`: The starting part number used if `max_filesize` is finite.
            Default: 1.

  - `jld2_kw`: Dict of kwargs to be passed to `jldopen` when data is written.

Example
=======

Write out 3D fields for u, v, w, and a tracer c, along with a horizontal average:

```jldoctest jld2_output_writer
using Oceananigans
using Oceananigans.Utils: hour, minute

model = NonhydrostaticModel(grid=RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1)), tracers=(:c,))
simulation = Simulation(model, Δt=12, stop_time=1hour)

function init_save_some_metadata!(file, model)
    file["author"] = "Chim Riggles"
    file["parameters/coriolis_parameter"] = 1e-4
    file["parameters/density"] = 1027
    return nothing
end

c_avg =  Field(Average(model.tracers.c, dims=(1, 2)))

# Note that model.velocities is NamedTuple
simulation.output_writers[:velocities] = JLD2OutputWriter(model, model.velocities,
                                                          prefix = "some_data",
                                                          schedule = TimeInterval(20minute),
                                                          init = init_save_some_metadata!)

# output
JLD2OutputWriter scheduled on TimeInterval(20 minutes):
├── filepath: ./some_data.jld2
├── 3 outputs: (u, v, w)
├── array type: Array{Float32}
├── including: [:grid, :coriolis, :buoyancy, :closure]
└── max filesize: Inf YiB
```

and a time- and horizontal-average of tracer `c` every 20 minutes of simulation time
to a file called `some_averaged_data.jld2`

```jldoctest jld2_output_writer
simulation.output_writers[:avg_c] = JLD2OutputWriter(model, (; c=c_avg),
                                                     prefix = "some_averaged_data",
                                                     schedule = AveragedTimeInterval(20minute, window=5minute))

# output
JLD2OutputWriter scheduled on TimeInterval(20 minutes):
├── filepath: ./some_averaged_data.jld2
├── 1 outputs: c averaged on AveragedTimeInterval(window=5 minutes, stride=1, interval=20 minutes)
├── array type: Array{Float32}
├── including: [:grid, :coriolis, :buoyancy, :closure]
└── max filesize: Inf YiB
```
"""
function JLD2OutputWriter(model, outputs; prefix, schedule,
                                   dir = ".",
                               indices = (:, :, :),
                            with_halos = false,
                            array_type = Array{Float32},
                          max_filesize = Inf,
                                 force = false,
                                  init = noinit,
                             including = default_included_properties(model),
                               verbose = false,
                                  part = 1,
                               jld2_kw = Dict{Symbol, Any}())

    outputs = NamedTuple(Symbol(name) => construct_output(outputs[name], model.grid, indices, with_halos) for name in keys(outputs))

    # Convert each output to WindowedTimeAverage if schedule::AveragedTimeWindow is specified
    schedule, outputs = time_average_outputs(schedule, outputs, model)

    mkpath(dir)
    filepath = joinpath(dir, prefix * ".jld2")
    force && isfile(filepath) && rm(filepath, force=true)

    initialize_jld2_file!(filepath, init, jld2_kw, including, outputs, model)
    
    return JLD2OutputWriter(filepath, outputs, schedule, array_type, init,
                            including, part, max_filesize, force, verbose, jld2_kw)
end

function initialize_jld2_file!(filepath, init, jld2_kw, including, outputs, model)
    try
        jldopen(filepath, "a+"; jld2_kw...) do file
            init(file, model)
            saveproperties!(file, model, including)

            # Serialize properties in `including`.
            for property in including
                serializeproperty!(file, "serialized/$property", getproperty(model, property))
            end

            # Serialize the location and boundary conditions of each output.
            for (i, (field_name, field)) in enumerate(pairs(outputs))
                file["timeseries/$field_name/serialized/location"] = location(field)
                file["timeseries/$field_name/serialized/indices"] = indices(field)
                serializeproperty!(file, "timeseries/$field_name/serialized/boundary_conditions", boundary_conditions(field))
            end
        end
    catch err
        @warn """Initialization of $filepath failed because $(typeof(err)): $(sprint(showerror, err))"""
    end

    return nothing
end

initialize_jld2_file!(writer::JLD2OutputWriter, model) =
    initialize_jld2_file!(writer.filepath, writer.init, writer.jld2_kw, writer.including, writer.outputs, model)

function iteration_zero_exists(filepath)
    file = jldopen(filepath, "r")

    zero_exists = try
        t₀ = file["timeseries/t/0"]
        true
    catch
        false
    finally
        close(file)
    end

    return zero_exists
end

function write_output!(writer::JLD2OutputWriter, model)

    # Catch an error that occurs when a simulation is initialized but not time-stepped:
    if model.clock.iteration == 0 && iteration_zero_exists(writer.filepath)
        if writer.force
            # Re-initialize file:
            rm(writer.filepath, force=true)
            initialize_jld2_file!(writer, model)
        else
            error("Attempting to overwrite data at iteration 0, possibly because a simulation is being
                  re-initialized. Use `force=true` when constructing JLD2OutputWriter to
                  replace any existing files and avoid this error.")
        end
    end

    verbose = writer.verbose

    # Fetch JLD2 output and store in dictionary `data`
    verbose && @info @sprintf("Fetching JLD2 output %s...", keys(writer.outputs))

    tc = Base.@elapsed data = Dict((name, fetch_and_convert_output(output, model, writer)) for (name, output)
                                   in zip(keys(writer.outputs), values(writer.outputs)))

    verbose && @info "Fetching time: $(prettytime(tc))"

    # Start a new file if the filesize exceeds max_filesize
    filesize(writer.filepath) >= writer.max_filesize && start_next_file(model, writer)

    # Write output from `data`
    path = writer.filepath
    verbose && @info "Writing JLD2 output $(keys(writer.outputs)) to $path..."

    start_time, old_filesize = time_ns(), filesize(path)
    jld2output!(path, model.clock.iteration, model.clock.time, data, writer.jld2_kw)
    end_time, new_filesize = time_ns(), filesize(path)

    verbose && @info @sprintf("Writing done: time=%s, size=%s, Δsize=%s",
                              prettytime((end_time - start_time) / 1e9),
                              pretty_filesize(new_filesize),
                              pretty_filesize(new_filesize - old_filesize))

    return nothing
end

"""
    jld2output!(path, iter, time, data, kwargs)

Write the (name, value) pairs in `data`, including the simulation
`time`, to the JLD2 file at `path` in the `timeseries` group,
stamping them with `iter` and using `kwargs` when opening
the JLD2 file.
"""
function jld2output!(path, iter, time, data, kwargs)
    jldopen(path, "r+"; kwargs...) do file
        file["timeseries/t/$iter"] = time
        for (name, datum) in data
            file["timeseries/$name/$iter"] = datum
        end
    end
    return nothing
end

function start_next_file(model, writer::JLD2OutputWriter)
    verbose = writer.verbose
    sz = filesize(writer.filepath)
    verbose && @info begin
        "Filesize $(pretty_filesize(sz)) has exceeded maximum file size $(pretty_filesize(writer.max_filesize))."
    end

    if writer.part == 1
        part1_path = replace(writer.filepath, r".jld2$" => "_part1.jld2")
        verbose && @info "Renaming first part: $(writer.filepath) -> $part1_path"
        mv(writer.filepath, part1_path, force=writer.force)
        writer.filepath = part1_path
    end

    writer.part += 1
    writer.filepath = replace(writer.filepath, r"part\d+.jld2$" => "part" * string(writer.part) * ".jld2")
    writer.force && isfile(writer.filepath) && rm(writer.filepath, force=true)
    verbose && @info "Now writing to: $(writer.filepath)"

    initialize_jld2_file!(writer, model)
    
    return nothing
end

Base.summary(ow::JLD2OutputWriter) =
    string("JLD2OutputWriter writing ", prettykeys(ow.outputs), " to ", ow.filepath, " on ", summary(ow.schedule))

function Base.show(io::IO, ow::JLD2OutputWriter)

    averaging_schedule = output_averaging_schedule(ow)
    Noutputs = length(ow.outputs)

    print(io, "JLD2OutputWriter scheduled on $(summary(ow.schedule)):", '\n',
              "├── filepath: $(ow.filepath)", '\n',
              "├── $Noutputs outputs: ", prettykeys(ow.outputs), show_averaging_schedule(averaging_schedule), '\n',
              "├── array type: ", show_array_type(ow.array_type), '\n',
              "├── including: ", ow.including, '\n',
              "└── max filesize: ", pretty_filesize(ow.max_filesize))
end
