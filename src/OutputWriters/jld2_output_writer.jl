using Printf
using JLD2
using Oceananigans.Utils

"""
    JLD2OutputWriter{I, T, O, IF, IN, KW} <: AbstractOutputWriter

An output writer for writing to JLD2 files.
"""
mutable struct JLD2OutputWriter{I, T, O, FS, D, IF, IN, KW} <: AbstractOutputWriter
              filepath :: String
               outputs :: O
    iteration_interval :: I
         time_interval :: T
          field_slicer :: FS
            array_type :: D
                  init :: IF
             including :: IN
              previous :: Float64
                  part :: Int
          max_filesize :: Float64
                 force :: Bool
               verbose :: Bool
               jld2_kw :: KW
end

noinit(args...) = nothing

"""
    JLD2OutputWriter(model, outputs; prefix,
                                                       dir = ".",
                                        iteration_interval = nothing,
                                             time_interval = nothing,
                                     time_averaging_window = nothing,
                                     time_averaging_stride = 1,
                                              max_filesize = Inf,
                                                     force = false,
                                                      init = noinit,
                                                 including = [:grid, :coriolis, :buoyancy, :closure],
                                                   verbose = false,
                                                      part = 1,
                                                array_type = Array{Float32},
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

    - `prefix`: Descriptive filename prefixed to all output files.
    
    - `dir`: Directory to save output to.
             Default: "." (current working directory).

    ## Output frequency and time-averaging
    
    - `iteration_interval`: Save output every `iteration_interval` model iterations.
    
    - `time_interval`: Save output every `time_interval` units of `model.clock.time`.
    
    - `time_averaging_window`: Specifies a time window over which each member of `output` is averaged before    
                               being saved. For this each member of output is converted to 
                               `Oceananigans.Diagnostics.WindowedTimeAverage`.
                               Default `nothing` indicates no averaging.
    
    - `time_averaging_stride`: Specifies a iteration 'stride' between the calculation of each `output` during
                               time-averaging. Longer strides means that output is calculated less frequently,
                               and that the resulting time-average is faster to compute, but less accurate.
                               Default: 1.

    ## Slicing and type conversion prior to output

    - `field_slicer`: An object for slicing field output in ``(x, y, z)``, including omitting halos.
                      Has no effect on output that is not a field. `field_slicer = nothing` means
                      no slicing occurs, so that all field data, including halo regions, is saved.
                      Default: FieldSlicer(), which slices halo regions.

    - `array_type`: The array type to which output arrays are converted to prior to saving.
                    Default: Array{Float32}.

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
"""
function JLD2OutputWriter(model, outputs; prefix,
                                                            dir = ".",
                                             iteration_interval = nothing,
                                                  time_interval = nothing,
                                          time_averaging_window = nothing,
                                          time_averaging_stride = 1,
                                                   field_slicer = FieldSlicer(),
                                                     array_type = Array{Float32},
                                                   max_filesize = Inf,
                                                          force = false,
                                                           init = noinit,
                                                      including = [:grid, :coriolis, :buoyancy, :closure],
                                                        verbose = false,
                                                           part = 1,
                                                        jld2_kw = Dict{Symbol, Any}())

    validate_intervals(iteration_interval, time_interval)

    # Convert each output to WindowedTimeAverage if time_averaging_window is specified
    if !isnothing(time_averaging_window)

        !isnothing(iteration_interval) && error("Cannot specify iteration_interval with time_averaging_window.")

        output_names = Tuple(keys(outputs))

        averaged_output =
            Tuple(
                  WindowedTimeAverage(outputs[name], model; time_interval = time_interval,
                                                              time_window = time_averaging_window,
                                                                   stride = time_averaging_stride,
                                                             field_slicer = field_slicer)
                  for name in output_names
                 )

        outputs = NamedTuple{output_names}(averaged_output)
    end

    mkpath(dir)
    filepath = joinpath(dir, prefix * ".jld2")
    force && isfile(filepath) && rm(filepath, force=true)

    jldopen(filepath, "a+"; jld2_kw...) do file
        init(file, model)
        saveproperties!(file, model, including)
    end

    return JLD2OutputWriter(filepath,
                            outputs,
                            iteration_interval,
                            time_interval,
                            field_slicer,
                            array_type,
                            init,
                            including,
                            0.0,
                            part,
                            max_filesize,
                            force,
                            verbose, 
                            jld2_kw)
end

function write_output!(writer::JLD2OutputWriter, model)

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
                              prettytime((start_time - end_time) / 1e9),
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

    jldopen(writer.filepath, "a+"; writer.jld2_kw...) do file
        writer.init(file, model)
        saveproperties!(file, model, writer.including)
    end

    return nothing
end
