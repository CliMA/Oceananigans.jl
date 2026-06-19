using Glob
using StructArrays: StructArray

using Oceananigans
using Oceananigans.TimeSteppers: QuasiAdamsBashforth2TimeStepper
using Oceananigans.Grids: halo_size, with_halo
using Oceananigans.Architectures: on_architecture, architecture, CPU

import Oceananigans: prognostic_state, restore_prognostic_state!, fs_halo_size,
                     checkpoint_restore_mode, warn_if_cross_grid_pickup,
                     RestoreOnCurrentGrid, RestoreOnCompatibleGrid
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
            On distributed architectures, `"_rank{local_rank}"` is appended automatically.

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
    filename = with_architecture_suffix(architecture(model), string(prefix, ".jld2"), ".jld2")
    prefix = String(chop(filename, tail=length(".jld2")))

    return Checkpointer(schedule, dir, prefix, overwrite_existing, verbose, cleanup)
end

#####
##### Checkpointer utils
#####

checkpointer_address(model) = ""

""" Return the full prefix (the `superprefix`) associated with `checkpointer`. """
checkpoint_superprefix(prefix) = prefix * "_iteration"

"""
$(TYPEDSIGNATURES)

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
$(TYPEDSIGNATURES)

For `pickup=true`, parse the filenames in `checkpointer.dir` associated with
`checkpointer.prefix` and return the path to the most recently modified
checkpoint file.
"""
function checkpoint_path(pickup::Bool, checkpointer::Checkpointer)
    pickup || return nothing
    return checkpoint_path(:recent_time_stamp, checkpointer)
end

"""
$(TYPEDSIGNATURES)

For symbol-based pickup modes:

- `pickup=:recent_time_stamp` returns the most recently modified checkpoint file.
- `pickup=:highest_iteration` returns the checkpoint file with the largest iteration in its name.
- `pickup=:latest` is an alias for `:recent_time_stamp`.
"""
function checkpoint_path(pickup::Symbol, checkpointer::Checkpointer)
    mode = pickup === :latest ? :recent_time_stamp : pickup
    mode in (:recent_time_stamp, :highest_iteration) || throw(ArgumentError("Unsupported pickup mode $pickup. Supported modes are :recent_time_stamp and :highest_iteration."))

    filepaths = glob(checkpoint_superprefix(checkpointer.prefix) * "*.jld2", checkpointer.dir)

    if length(filepaths) == 0 # no checkpoint files found
        # https://github.com/CliMA/Oceananigans.jl/issues/1159
        @warn "pickup=$pickup but no checkpoints were found. Simulation will run without picking up."
        return nothing
    elseif mode === :highest_iteration
        return latest_checkpoint_by_iteration(checkpointer, filepaths)
    else
        return latest_checkpoint_by_time_stamp(checkpointer, filepaths)
    end
end

function latest_checkpoint_by_iteration(checkpointer, filepaths)
    filenames = basename.(filepaths)
    leading = length(checkpoint_superprefix(checkpointer.prefix))
    trailing = length(".jld2") # 5
    iterations = map(name -> parse(Int, chop(name, head=leading, tail=trailing)), filenames)
    _, idx = findmax(iterations)
    return filepaths[idx]
end

function latest_checkpoint_by_time_stamp(checkpointer, filepaths)
    modification_times = map(filepath -> stat(filepath).mtime, filepaths)
    latest_time = maximum(modification_times)
    indices_with_latest_time = findall(==(latest_time), modification_times)
    candidate_paths = filepaths[indices_with_latest_time]

    length(candidate_paths) == 1 && return first(candidate_paths)

    # Deterministic tie-breaker for coarse filesystem mtimes:
    # if multiple files have the same latest timestamp, pick the highest iteration.
    return latest_checkpoint_by_iteration(checkpointer, candidate_paths)
end

# Backward-compatibility shim: historical "latest_checkpoint" means "latest by iteration".
# Keep this behavior for existing internal/external callsites (for example cleanup logic),
# while the new :recent_time_stamp mode is selected explicitly via checkpoint_path(::Symbol, ...).
latest_checkpoint(checkpointer, filepaths) = latest_checkpoint_by_iteration(checkpointer, filepaths)

#####
##### Writing checkpoints
#####

checkpoint_restore_mode(::Nothing, grid) = RestoreOnCurrentGrid()

function checkpoint_restore_mode(checkpoint_grid, grid)
    same_interior = checkpoint_grid == with_halo(halo_size(checkpoint_grid), on_architecture(CPU(), grid))
    same_interior || throw(ArgumentError("Checkpoint pickup only supports the same interior grid with a different halo size. Restoring across different grids or resolutions is not supported by this path."))

    return halo_size(checkpoint_grid) == halo_size(grid) ?
           RestoreOnCurrentGrid() :
           RestoreOnCompatibleGrid(checkpoint_grid)
end

warn_if_cross_grid_pickup(checkpoint_grid, grid) = warn_if_cross_grid_pickup(checkpoint_restore_mode(checkpoint_grid, grid), grid)
warn_if_cross_grid_pickup(::RestoreOnCurrentGrid, grid) = nothing
function warn_if_cross_grid_pickup(mode::RestoreOnCompatibleGrid, grid)
    @warn "Picking up a checkpoint onto the same interior grid with a different halo size; interiors will be restored directly and halos will be rebuilt afterward." checkpoint_grid=mode.grid model_grid=grid
    return nothing
end

function checkpoint_restore_mode(restored, checkpoint_grid, checkpoint_free_surface_grid)
    same_interior = checkpoint_grid == with_halo(halo_size(checkpoint_grid), on_architecture(CPU(), restored.grid))
    same_interior || throw(ArgumentError("Checkpoint pickup only supports the same interior grid with a different halo size. Restoring across different grids or resolutions is not supported by this path."))

    checkpoint_free_surface_halo = halo_size(checkpoint_free_surface_grid)
    restored_free_surface_halo = fs_halo_size(restored)

    same_grid_halos = halo_size(checkpoint_grid) == halo_size(restored.grid)
    same_free_surface_halos = checkpoint_free_surface_halo == restored_free_surface_halo

    if same_grid_halos && same_free_surface_halos
        return RestoreOnCurrentGrid()
    end

    restore_halo = ntuple(d -> checkpoint_free_surface_halo[d] - halo_size(checkpoint_grid, d) + halo_size(restored.grid, d), 3)
    restore_grid = with_halo(restore_halo, on_architecture(CPU(), restored.grid))
    return RestoreOnCompatibleGrid(restore_grid)
end


prognostic_state(obj) = obj
prognostic_state(::NamedTuple{()}) = nothing
prognostic_state(::NoFileSplitting) = nothing
prognostic_state(::FileSizeLimit) = nothing
restore_prognostic_state!(::NoFileSplitting, from) = nothing
restore_prognostic_state!(::FileSizeLimit, from) = nothing

prognostic_state(tuple::Tuple) = Tuple(prognostic_state(t) for t in tuple)

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
$(TYPEDSIGNATURES)

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
restore_prognostic_state!(::NamedTuple{()}, from) = nothing
restore_prognostic_state!(::NamedTuple{()}, ::Nothing) = nothing
restore_prognostic_state!(::AbstractDict, ::Nothing) = nothing
restore_prognostic_state!(::Nothing, from) = nothing
restore_prognostic_state!(::Nothing, ::Nothing) = nothing

# To resolve dispatch ambiguities with `restore_prognostic_state!(obj, ::Nothing)`
restore_prognostic_state!(::AbstractArray, ::Nothing) = nothing
restore_prognostic_state!(::NamedTuple, ::Nothing) = nothing
restore_prognostic_state!(::StructArray, ::Nothing) = nothing
restore_prognostic_state!(::Ref, ::Nothing) = nothing
restore_prognostic_state!(::Checkpointer, ::Nothing) = nothing
restore_prognostic_state!(::Union{JLD2Writer, NetCDFWriter}, ::Nothing) = nothing

function restore_prognostic_state!(restored::AbstractArray, from)
    copyto!(restored, on_architecture(architecture(restored), from))
    return restored
end

function restore_prognostic_state!(restored::AbstractDict, from)
    for (name, value) in pairs(from)
        haskey(restored, name) && restore_prognostic_state!(restored[name], value)
    end
    return restored
end

function restore_prognostic_state!(restored::AbstractDict, from, mode)
    for (name, value) in pairs(from)
        haskey(restored, name) && restore_prognostic_state!(restored[name], value, mode)
    end
    return restored
end

function restore_prognostic_state!(restored::NamedTuple, from)
    for (name, value) in pairs(from)
        restore_prognostic_state!(restored[name], value)
    end
    return restored
end

function restore_prognostic_state!(restored::NamedTuple, from, mode)
    for (name, value) in pairs(from)
        restore_prognostic_state!(restored[name], value, mode)
    end
    return restored
end

function restore_prognostic_state!(t::Tuple, from::Tuple)
    new_t = tuple(restore_prognostic_state!(t[j], from[j]) for j in 1:length(t))
    return new_t
end

function restore_prognostic_state!(t::Tuple, from::Tuple, mode)
    new_t = tuple(restore_prognostic_state!(t[j], from[j], mode) for j in 1:length(t))
    return new_t
end

function restore_prognostic_state!(restored::StructArray, from)
    # Get the architecture from one of the component arrays
    some_property = first(propertynames(restored))
    arch = architecture(getproperty(restored, some_property))

    # Copy each property
    for name in propertynames(restored)
        data = on_architecture(arch, getproperty(from, name))
        copyto!(getproperty(restored, name), data)
    end

    return restored
end

restore_prognostic_state!(restored::StructArray, from, mode) = restore_prognostic_state!(restored, from)

# Ref handling: dereference on save, set on restore
prognostic_state(r::Ref) = r[]
restore_prognostic_state!(restored::Ref, from) = (restored[] = from; restored)
restore_prognostic_state!(restored::Ref, from, mode) = restore_prognostic_state!(restored, from)
restore_prognostic_state!(::Nothing, ::Nothing, mode) = nothing

#####
##### Checkpointing the checkpointer
#####

function prognostic_state(checkpointer::Checkpointer)
    return (; schedule = prognostic_state(checkpointer.schedule))
end

function restore_prognostic_state!(restored::Checkpointer, from)
    restore_prognostic_state!(restored.schedule, from.schedule)
    return restored
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
            file_splitting = prognostic_state(writer.file_splitting),
            windowed_time_averages = isempty(wta_outputs) ? nothing : wta_outputs)
end

function restore_prognostic_state!(restored::Union{JLD2Writer, NetCDFWriter}, from)
    restore_prognostic_state!(restored.schedule, from.schedule)
    restored.part = from.part

    # Update the filepath to match the restored part number so that
    # the writer appends to the correct part file after pickup.
    restored.filepath = filepath_for_part(restored.filepath, restored.part)

    # Restore file_splitting schedule state (e.g., TimeInterval actuations)
    # so splitting doesn't re-trigger immediately after pickup.
    # Backward compatible: old checkpoints may not have file_splitting.
    if hasproperty(from, :file_splitting) && !isnothing(from.file_splitting)
        restore_prognostic_state!(restored.file_splitting, from.file_splitting)
    end

    if hasproperty(from, :windowed_time_averages) && !isnothing(from.windowed_time_averages)
        for (name, wta_state) in pairs(from.windowed_time_averages)
            key = output_lookup_key(restored, name)
            if haskey(restored.outputs, key) && restored.outputs[key] isa WindowedTimeAverage
                restore_prognostic_state!(restored.outputs[key], wta_state)
            end
        end
    end

    return restored
end

function restore_prognostic_state!(restored::OffsetArray, from::AbstractArray)
    restored_parent = parent(restored)
    return restore_prognostic_state!(restored_parent, from)
end

function restore_prognostic_state!(restored::OffsetArray, from::OffsetArray)
    restored_parent = parent(restored)
    from_parent = parent(from)
    return restore_prognostic_state!(restored_parent, from_parent)
end

function restore_prognostic_state!(restored::Number, from::Number)
    restored = convert(typeof(restored), from)
    return restored
end

#####
##### Manual checkpointing
#####

"""
    checkpoint(simulation; filepath=nothing)

Manually checkpoint `simulation` state to a JLD2 file.

If `filepath` is provided, the checkpoint is written there. Otherwise, if
`simulation.output_writers` contains a single `Checkpointer`, it is used
(respecting its `dir`, `prefix`, `cleanup`, and `verbose` settings); if no
`filepath` is given and there is no single `Checkpointer`, the checkpoint is
written to `"checkpoint_iteration{N}.jld2"` in the current directory.
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
