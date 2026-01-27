using Glob
using StructArrays: StructArray

using Oceananigans
using Oceananigans.TimeSteppers: QuasiAdamsBashforth2TimeStepper

import Oceananigans: prognostic_state, restore_prognostic_state!
import Oceananigans.Fields: set!

mutable struct Checkpointer{T} <: AbstractOutputWriter
    schedule :: T
    dir :: String
    prefix :: String
    overwrite_existing :: Bool
    verbose :: Bool
    cleanup :: Bool
end

"""
    Checkpointer(model;
                 schedule,
                 dir = ".",
                 prefix = "checkpoint",
                 overwrite_existing = false,
                 verbose = false,
                 cleanup = false)

Construct a `Checkpointer` that checkpoints the model to a JLD2 file on `schedule.`
The `model.clock.iteration` is included in the filename to distinguish between multiple checkpoint files.

To restart or "pickup" a model from a checkpoint, specify `pickup = true` when calling `run!`, ensuring
that the checkpoint file is in directory `dir`. See [`run!`](@ref) for more details.

The checkpointer attempts to serialize as much of the model to disk as possible,
but functions or objects containing functions cannot be serialized at this time.

Keyword arguments
=================

- `schedule` (required): Schedule that determines when to checkpoint.

- `dir`: Directory to save output to. Default: `"."` (current working directory).

- `prefix`: Descriptive filename prefixed to all output files. Default: `"checkpoint"`.

- `overwrite_existing`: Remove existing files if their filenames conflict. Default: `false`.

- `verbose`: Log what the output writer is doing with statistics on compute/write times
             and file sizes. Default: `false`.

- `cleanup`: Previous checkpoint files are deleted once a new checkpoint file is written.
             Default: `false`.
"""
function Checkpointer(model; schedule,
                      dir = ".",
                      prefix = "checkpoint",
                      overwrite_existing = false,
                      verbose = false,
                      cleanup = false)

    mkpath(dir)

    return Checkpointer(schedule, dir, prefix, overwrite_existing, verbose, cleanup)
end

#####
##### Checkpointer utils
#####

checkpointer_address(model) = ""

""" Return the full prefix (the `superprefix`) associated with `checkpointer`. """
checkpoint_superprefix(prefix) = prefix * "_iteration"

"""
    checkpoint_path(iteration::Int, checkpointer::Checkpointer)

Return the path to the `checkpointer` file associated with model `iteration`.
"""
checkpoint_path(iteration::Int, checkpointer::Checkpointer) =
    joinpath(checkpointer.dir, string(checkpoint_superprefix(checkpointer.prefix), iteration, ".jld2"))

""" Returns `filepath`. Shortcut for `run!(simulation, pickup=filepath)`. """
checkpoint_path(filepath::String, output_writers) = filepath

function checkpoint_path(pickup, output_writers)
    checkpointers = filter(writer -> writer isa Checkpointer, collect(values(output_writers)))
    length(checkpointers) == 0 && error("No checkpointers found: cannot pickup simulation!")
    length(checkpointers) > 1 && error("Multiple checkpointers found: not sure which one to pickup simulation from!")
    return checkpoint_path(pickup, first(checkpointers))
end

"""
    checkpoint_path(pickup::Bool, checkpointer::Checkpointer)

For `pickup=true`, parse the filenames in `checkpointer.dir` associated with
`checkpointer.prefix` and return the path to the file whose name contains
the largest iteration.
"""
function checkpoint_path(pickup::Bool, checkpointer::Checkpointer)
    filepaths = glob(checkpoint_superprefix(checkpointer.prefix) * "*.jld2", checkpointer.dir)

    if length(filepaths) == 0 # no checkpoint files found
        # https://github.com/CliMA/Oceananigans.jl/issues/1159
        @warn "pickup=true but no checkpoints were found. Simulation will run without picking up."
        return nothing
    else
        return latest_checkpoint(checkpointer, filepaths)
    end
end

function latest_checkpoint(checkpointer, filepaths)
    filenames = basename.(filepaths)
    leading = length(checkpoint_superprefix(checkpointer.prefix))
    trailing = length(".jld2") # 5
    iterations = map(name -> parse(Int, chop(name, head=leading, tail=trailing)), filenames)
    latest_iteration, idx = findmax(iterations)
    return filepaths[idx]
end

#####
##### Writing checkpoints
#####

prognostic_state(obj) = obj
prognostic_state(::NamedTuple{()}) = nothing

function prognostic_state(nt::NamedTuple)
    ks = keys(nt)
    vs = Tuple(prognostic_state(v) for v in values(nt))
    return NamedTuple{ks}(vs)
end

function prognostic_state(dict::AbstractDict)
    isempty(dict) && return nothing
    ks = tuple(keys(dict)...)
    vs = Tuple(prognostic_state(v) for v in values(dict))
    return NamedTuple{ks}(vs)
end

function cleanup_checkpoints(checkpointer)
    filepaths = glob(checkpoint_superprefix(checkpointer.prefix) * "*.jld2", checkpointer.dir)
    latest_checkpoint_filepath = latest_checkpoint(checkpointer, filepaths)
    [rm(filepath) for filepath in filepaths if filepath != latest_checkpoint_filepath]
    return nothing
end

function write_output!(c::Checkpointer, simulation)
    iter = iteration(simulation)
    filepath = checkpoint_path(iter, c)

    t1 = time_ns()

    state = prognostic_state(simulation)

    jldopen(filepath, "w") do file
        serializeproperty!(file, "simulation", state)
    end

    t2, sz = time_ns(), filesize(filepath)
    c.verbose && @info "Checkpointing done: time=$(prettytime((t2 - t1) * 1e-9)), size=$(pretty_filesize(sz))"

    c.cleanup && cleanup_checkpoints(c)

    return nothing
end

#####
##### Reading checkpoints and restoring from them
#####

"""
    load_nested_data(obj)

Recursively load data from a JLD2 group or dataset, reconstructing nested NamedTuples for
groups and returning raw data for leaf nodes.
"""
function load_nested_data(obj)
    if obj isa JLD2.Group
        group_keys = keys(obj)
        key_symbols = Symbol.(collect(group_keys))
        child_values = Tuple(load_nested_data(obj[key]) for key in group_keys)
        return NamedTuple{tuple(key_symbols...)}(child_values)
    else
        return obj
    end
end

"""
    load_checkpoint_state(filepath; base_path="simulation")

Load checkpoint data from a JLD2 file and return it as a nested NamedTuple.
"""
function load_checkpoint_state(filepath; base_path="simulation")
    jldopen(filepath, "r") do file
        return load_nested_data(file[base_path])
    end
end

# Handle case when no checkpoint file exists (filepath is nothing)
load_checkpoint_state(::Nothing; base_path="simulation") = nothing

restore_prognostic_state!(obj, ::Nothing) = nothing
restore_prognostic_state!(::NamedTuple{()}, state) = nothing
restore_prognostic_state!(::NamedTuple{()}, ::Nothing) = nothing
restore_prognostic_state!(::AbstractDict, ::Nothing) = nothing
restore_prognostic_state!(::Nothing, state) = nothing
restore_prognostic_state!(::Nothing, ::Nothing) = nothing

# To resolve dispatch ambiguities with `restore_prognostic_state!(obj, ::Nothing)`
restore_prognostic_state!(::AbstractArray, ::Nothing) = nothing
restore_prognostic_state!(::NamedTuple, ::Nothing) = nothing
restore_prognostic_state!(::StructArray, ::Nothing) = nothing
restore_prognostic_state!(::Ref, ::Nothing) = nothing
restore_prognostic_state!(::Checkpointer, ::Nothing) = nothing
restore_prognostic_state!(::Union{JLD2Writer, NetCDFWriter}, ::Nothing) = nothing

function restore_prognostic_state!(arr::AbstractArray, state)
    arch = architecture(arr)
    data = on_architecture(arch, state)
    copyto!(arr, data)
    return arr
end

function restore_prognostic_state!(dict::AbstractDict, state)
    for (name, value) in pairs(state)
        haskey(dict, name) && restore_prognostic_state!(dict[name], value)
    end
    return dict
end

function restore_prognostic_state!(nt::NamedTuple, state)
    for (name, value) in pairs(state)
        restore_prognostic_state!(nt[name], value)
    end
    return nt
end

function restore_prognostic_state!(sa::StructArray, state)
    # Get the architecture from one of the component arrays
    some_property = first(propertynames(sa))
    arch = architecture(getproperty(sa, some_property))

    # Copy each property
    for name in propertynames(sa)
        data = on_architecture(arch, getproperty(state, name))
        copyto!(getproperty(sa, name), data)
    end

    return sa
end

# Ref handling: dereference on save, set on restore
prognostic_state(r::Ref) = r[]
restore_prognostic_state!(r::Ref, value) = (r[] = value; r)

#####
##### Checkpointing the checkpointer
#####

function prognostic_state(checkpointer::Checkpointer)
    return (; schedule = prognostic_state(checkpointer.schedule))
end

function restore_prognostic_state!(checkpointer::Checkpointer, state)
    restore_prognostic_state!(checkpointer.schedule, state.schedule)
    return checkpointer
end

#####
##### Checkpointing file-based output writers (JLD2Writer, NetCDFWriter)
#####

output_key_to_symbol(name::Symbol) = name
output_key_to_symbol(name::AbstractString) = Symbol(name)

output_lookup_key(::JLD2Writer, name::Symbol) = name
output_lookup_key(::NetCDFWriter, name::Symbol) = string(name)

function prognostic_state(writer::Union{JLD2Writer, NetCDFWriter})
    wta_outputs = NamedTuple(output_key_to_symbol(name) => prognostic_state(output)
                             for (name, output) in pairs(writer.outputs)
                             if output isa WindowedTimeAverage)

    return (schedule = prognostic_state(writer.schedule),
            part = writer.part,
            windowed_time_averages = isempty(wta_outputs) ? nothing : wta_outputs)
end

function restore_prognostic_state!(writer::Union{JLD2Writer, NetCDFWriter}, state)
    restore_prognostic_state!(writer.schedule, state.schedule)
    writer.part = state.part

    if hasproperty(state, :windowed_time_averages) && !isnothing(state.windowed_time_averages)
        for (name, wta_state) in pairs(state.windowed_time_averages)
            key = output_lookup_key(writer, name)
            if haskey(writer.outputs, key) && writer.outputs[key] isa WindowedTimeAverage
                restore_prognostic_state!(writer.outputs[key], wta_state)
            end
        end
    end

    return writer
end

#####
##### Manual checkpointing
#####

"""
    checkpoint(simulation; filepath=nothing)

Manually checkpoint `simulation` state to a JLD2 file.

If `simulation.output_writers` contains a `Checkpointer`, it will be used
(respecting its `dir`, `prefix`, `cleanup`, and `verbose` settings).

Otherwise, the checkpoint is written to `filepath`, or to
`"checkpoint_iteration{N}.jld2"` in the current directory if `filepath` is not specified.
"""
function checkpoint(simulation; filepath=nothing)
    checkpointers = filter(w -> w isa Checkpointer, collect(values(simulation.output_writers)))

    if !isnothing(filepath)
        write_checkpoint_file(filepath, simulation)
    elseif length(checkpointers) == 1
        write_output!(first(checkpointers), simulation)
    else
        iter = iteration(simulation)
        default_filepath = "checkpoint_iteration$(iter).jld2"
        @warn "No checkpointer (or multiple checkpointers) found, using default filepath: $default_filepath"
        write_checkpoint_file(default_filepath, simulation)
    end

    return nothing
end

function write_checkpoint_file(filepath, simulation)
    state = prognostic_state(simulation)
    jldopen(filepath, "w") do file
        serializeproperty!(file, "simulation", state)
    end
    return filepath
end
