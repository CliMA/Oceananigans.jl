using Printf
using JLD2
using Oceananigans.Utils
using Oceananigans.Models
using Oceananigans.Utils: TimeInterval, prettykeys
using Oceananigans.Fields: boundary_conditions, indices

default_included_properties(::NonhydrostaticModel) = [:grid, :coriolis, :buoyancy, :closure]
default_included_properties(::ShallowWaterModel) = [:grid, :coriolis, :closure]
default_included_properties(::HydrostaticFreeSurfaceModel) = [:grid, :coriolis, :buoyancy, :closure]

mutable struct JLD2OutputWriter{O, T, D, IF, IN, FS, KW} <: AbstractOutputWriter
    filepath :: String
    outputs :: O
    schedule :: T
    array_type :: D
    init :: IF
    including :: IN
    part :: Int
    file_splitting :: FS
    overwrite_existing :: Bool
    verbose :: Bool
    jld2_kw :: KW
end

noinit(args...) = nothing
ext(::Type{JLD2OutputWriter}) = ".jld2"

"""
    JLD2OutputWriter(model, outputs; filename, schedule,
                     dir = ".",
                     indices = (:, :, :),
                     with_halos = true,
                     array_type = Array{Float64},
                     file_splitting = NoFileSplitting(),
                     overwrite_existing = false,
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

- `filename` (required): Descriptive filename. `".jld2"` is appended to `filename` in the file path
                         if `filename` does not end in `".jld2"`.

- `dir`: Directory to save output to. Default: `"."` (current working directory).

## Output frequency and time-averaging

- `schedule` (required): `AbstractSchedule` that determines when output is saved.

## Slicing and type conversion prior to output

- `indices`: Specifies the indices to write to disk with a `Tuple` of `Colon`, `UnitRange`,
             or `Int` elements. Indices must be `Colon`, `Int`, or contiguous `UnitRange`.
             Defaults to `(:, :, :)` or "all indices". If `!with_halos`,
             halo regions are removed from `indices`. For example, `indices = (:, :, 1)`
             will save xy-slices of the bottom-most index.

- `with_halos` (Bool): Whether or not to slice halo regions from fields before writing output.
                       Note, that to postprocess saved output (e.g., compute derivatives, etc)
                       information about the boundary conditions is often crucial. In that case
                       you might need to set `with_halos = true`.

- `array_type`: The array type to which output arrays are converted to prior to saving.
                Default: `Array{Float64}`.

## File management

- `file_splitting`: Schedule for splitting the output file. The new files will be suffixed with
                    `_part1`, `_part2`, etc. For example `file_splitting = FileSizeLimit(sz)` will
                    split the output file when its size exceeds `sz`. Another example is 
                    `file_splitting = TimeInterval(30days)`, which will split files every 30 days of
                    simulation time. The default incurs no splitting (`NoFileSplitting()`).
                    
- `overwrite_existing`: Remove existing files if their filenames conflict.
                        Default: `false`.

## Output file metadata management

- `init`: A function of the form `init(file, model)` that runs when a JLD2 output file is initialized.
          Default: `noinit(args...) = nothing`.

- `including`: List of model properties to save with every file.
               Default: `[:grid, :coriolis, :buoyancy, :closure]`

## Miscellaneous keywords

- `verbose`: Log what the output writer is doing with statistics on compute/write times and file sizes.
             Default: `false`.

- `part`: The starting part number used when file splitting.
          Default: 1.

- `jld2_kw`: Dict of kwargs to be passed to `jldopen` when data is written.

Example
=======

Write out 3D fields for ``u``, ``v``, ``w``, and a tracer ``c``, along with a horizontal average:

```
using Oceananigans
using Oceananigans.Utils: hour, minute

model = NonhydrostaticModel(grid=RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1)), tracers=:c)
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
                                                          filename = "some_data.jld2",
                                                          schedule = TimeInterval(20minute),
                                                          init = init_save_some_metadata!)

# output
JLD2OutputWriter scheduled on TimeInterval(20 minutes):
├── filepath: ./some_data.jld2
├── 3 outputs: (u, v, w)
├── array type: Array{Float64}
├── including: [:grid, :coriolis, :buoyancy, :closure]
├── file_splitting: NoFileSplitting
└── file size: 28.0 KiB
```

and a time- and horizontal-average of tracer ``c`` every 20 minutes of simulation time
to a file called `some_averaged_data.jld2`

```
simulation.output_writers[:avg_c] = JLD2OutputWriter(model, (; c=c_avg),
                                                     filename = "some_averaged_data.jld2",
                                                     schedule = AveragedTimeInterval(20minute, window=5minute))

# output
JLD2OutputWriter scheduled on TimeInterval(20 minutes):
├── filepath: ./some_averaged_data.jld2
├── 1 outputs: c averaged on AveragedTimeInterval(window=5 minutes, stride=1, interval=20 minutes)
├── array type: Array{Float64}
├── including: [:grid, :coriolis, :buoyancy, :closure]
├── file_splitting: NoFileSplitting
└── file size: 17.8 KiB
```
"""
function JLD2OutputWriter(model, outputs; filename, schedule,
                          dir = ".",
                          indices = (:, :, :),
                          with_halos = true,
                          array_type = Array{Float64},
                          file_splitting = NoFileSplitting(),
                          overwrite_existing = false,
                          init = noinit,
                          including = default_included_properties(model),
                          verbose = false,
                          part = 1,
                          jld2_kw = Dict{Symbol, Any}())

    mkpath(dir)
    filename = auto_extension(filename, ".jld2")
    filepath = joinpath(dir, filename)
    initialize!(file_splitting, model)
    update_file_splitting_schedule!(file_splitting, filepath)
    overwrite_existing && isfile(filepath) && rm(filepath, force=true)

    outputs = NamedTuple(Symbol(name) => construct_output(outputs[name], model.grid, indices, with_halos)
                         for name in keys(outputs))

    # Convert each output to WindowedTimeAverage if schedule::AveragedTimeWindow is specified
    schedule, outputs = time_average_outputs(schedule, outputs, model)

    initialize_jld2_file!(filepath, init, jld2_kw, including, outputs, model)

    return JLD2OutputWriter(filepath, outputs, schedule, array_type, init,
                            including, part, file_splitting, overwrite_existing, verbose, jld2_kw)
end

function initialize_jld2_file!(filepath, init, jld2_kw, including, outputs, model)
    try
        jldopen(filepath, "a+"; jld2_kw...) do file
            init(file, model)
        end
    catch err
        @warn """Failed to execute user `init` for $filepath because $(typeof(err)): $(sprint(showerror, err))"""
    end

    try 
        jldopen(filepath, "a+"; jld2_kw...) do file
            saveproperties!(file, model, including)

            # Serialize properties in `including`.
            for property in including
                serializeproperty!(file, "serialized/$property", getproperty(model, property))
            end
        end
    catch err
        @warn """Failed to save and serialize $including in $filepath because $(typeof(err)): $(sprint(showerror, err))"""
    end

    # Serialize the location and boundary conditions of each output.
    for (name, field) in pairs(outputs)
        try
            jldopen(filepath, "a+"; jld2_kw...) do file
                file["timeseries/$name/serialized/location"] = location(field)
                file["timeseries/$name/serialized/indices"] = indices(field)
                serializeproperty!(file, "timeseries/$name/serialized/boundary_conditions", boundary_conditions(field))
            end
        catch
        end
    end

    return nothing
end

initialize_jld2_file!(writer::JLD2OutputWriter, model) =
    initialize_jld2_file!(writer.filepath, writer.init, writer.jld2_kw, writer.including, writer.outputs, model)

function iteration_exists(filepath, iter=0)
    file = jldopen(filepath, "r")

    zero_exists = try
        t₀ = file["timeseries/t/$iter"]
        true
    catch # This can fail for various reasons:
          #     the path does not exist, "t" does not exist...
        false
    finally
        close(file)
    end

    return zero_exists
end

function write_output!(writer::JLD2OutputWriter, model)

    verbose = writer.verbose
    current_iteration = model.clock.iteration

    # Some logic to handle writing to existing files
    if iteration_exists(writer.filepath, current_iteration)

        if writer.overwrite_existing
            # Something went wrong, so we remove the file and re-initialize it.
            rm(writer.filepath, force=true)
            initialize_jld2_file!(writer, model)
        else # nothing we can do since we were asked not to overwrite_existing, so we skip output writing
            @warn "Iteration $current_iteration was found in $(writer.filepath). Skipping output writing (for now...)"
        end

    else # ok let's do this

        # Fetch JLD2 output and store in `data`
        verbose && @info @sprintf("Fetching JLD2 output %s...", keys(writer.outputs))

        tc = Base.@elapsed data = NamedTuple(name => fetch_and_convert_output(output, model, writer) for (name, output)
                                             in zip(keys(writer.outputs), values(writer.outputs)))

        verbose && @info "Fetching time: $(prettytime(tc))"

        # Start a new file if the file_splitting(model) is true
        writer.file_splitting(model) && start_next_file(model, writer)
        update_file_splitting_schedule!(writer.file_splitting, writer.filepath)
        # Write output from `data`
        verbose && @info "Writing JLD2 output $(keys(writer.outputs)) to $path..."

        start_time, old_filesize = time_ns(), filesize(writer.filepath)
        jld2output!(writer.filepath, model.clock.iteration, model.clock.time, data, writer.jld2_kw)
        end_time, new_filesize = time_ns(), filesize(writer.filepath)

        verbose && @info @sprintf("Writing done: time=%s, size=%s, Δsize=%s",
                                  prettytime((end_time - start_time) / 1e9),
                                  pretty_filesize(new_filesize),
                                  pretty_filesize(new_filesize - old_filesize))
    end

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
        for name in keys(data)
            file["timeseries/$name/$iter"] = data[name]
        end
    end
    return nothing
end

function start_next_file(model, writer::JLD2OutputWriter)
    verbose = writer.verbose

    verbose && @info begin
        schedule_type = summary(writer.file_splitting)
        "Splitting output because $(schedule_type) is activated."
    end

    if writer.part == 1
        part1_path = replace(writer.filepath, r".jld2$" => "_part1.jld2")
        verbose && @info "Renaming first part: $(writer.filepath) -> $part1_path"
        mv(writer.filepath, part1_path, force=writer.overwrite_existing)
        writer.filepath = part1_path
    end

    writer.part += 1
    writer.filepath = replace(writer.filepath, r"part\d+.jld2$" => "part" * string(writer.part) * ".jld2")
    writer.overwrite_existing && isfile(writer.filepath) && rm(writer.filepath, force=true)
    verbose && @info "Now writing to: $(writer.filepath)"

    initialize_jld2_file!(writer, model)

    return nothing
end

Base.summary(ow::JLD2OutputWriter) =
    string("JLD2OutputWriter writing ", prettykeys(ow.outputs), " to ", ow.filepath, " on ", summary(ow.schedule))

function Base.show(io::IO, ow::JLD2OutputWriter)

    averaging_schedule = output_averaging_schedule(ow)
    Noutputs = length(ow.outputs)

    print(io, "JLD2OutputWriter scheduled on $(summary(ow.schedule)):", "\n",
              "├── filepath: $(ow.filepath)", "\n",
              "├── $Noutputs outputs: ", prettykeys(ow.outputs), show_averaging_schedule(averaging_schedule), "\n",
              "├── array type: ", show_array_type(ow.array_type), "\n",
              "├── including: ", ow.including, "\n",
              "├── file_splitting: ", summary(ow.file_splitting), "\n",
              "└── file size: ", pretty_filesize(filesize(ow.filepath)))
end
