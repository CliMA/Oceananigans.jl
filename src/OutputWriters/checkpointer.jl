using Glob

using Oceananigans
using Oceananigans: fields, prognostic_fields
using Oceananigans.Fields: offset_data
using Oceananigans.TimeSteppers: QuasiAdamsBashforth2TimeStepper

import Oceananigans.Fields: set!

mutable struct Checkpointer{T, P} <: AbstractOutputWriter
    schedule :: T
    dir :: String
    prefix :: String
    properties :: P
    overwrite_existing :: Bool
    verbose :: Bool
    cleanup :: Bool
end

function default_checkpointed_properties(model)
    properties = [:grid, :particles, :clock, :timestepper]
    #if has_ab2_timestepper(model)
    #    push!(properties, :timestepper)
    #end
    return properties
end

has_ab2_timestepper(model) = try
    model.timestepper isa QuasiAdamsBashforth2TimeStepper
catch
    false
end

"""
    Checkpointer(model;
                 schedule,
                 dir = ".",
                 prefix = "checkpoint",
                 overwrite_existing = false,
                 verbose = false,
                 cleanup = false,
                 properties = default_checkpointed_properties(model))

Construct a `Checkpointer` that checkpoints the model to a JLD2 file on `schedule.`
The `model.clock.iteration` is included in the filename to distinguish between multiple checkpoint files.

To restart or "pickup" a model from a checkpoint, specify `pickup = true` when calling `run!`, ensuring
that the checkpoint file is in directory `dir`. See [`run!`](@ref) for more details.

Note that extra model `properties` can be specified, but removing crucial properties
such as `:timestepper` will render restoring from the checkpoint impossible.

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

- `cleanup`: Previous checkpoint files will be deleted once a new checkpoint file is written.
             Default: `false`.

- `properties`: List of model properties to checkpoint. This list _must_ contain
                `:grid`, `:particles` and `:clock`, and if using AB2 timestepping then also
                `:timestepper`. Default: calls [`default_checkpointed_properties`](@ref) on
                `model` to get these properties.
"""
function Checkpointer(model; schedule,
                      dir = ".",
                      prefix = "checkpoint",
                      overwrite_existing = false,
                      verbose = false,
                      cleanup = false,
                      properties = default_checkpointed_properties(model))

    # Certain properties are required for `set!` to pickup from a checkpoint.
    required_properties = [:grid, :particles, :clock]

    if has_ab2_timestepper(model)
        push!(required_properties, :timestepper)
    end

    for rp in required_properties
        if rp ∉ properties
            @warn "$rp is required for checkpointing. It will be added to checkpointed properties"
            push!(properties, rp)
        end
    end

    for p in properties
        p isa Symbol || error("Property $p to be checkpointed must be a Symbol.")
        p ∉ propertynames(model) && error("Cannot checkpoint $p, it is not a model property!")

        if (p ∉ required_properties) && has_reference(Function, getproperty(model, p))
            @warn "model.$p contains a function somewhere in its hierarchy and will not be checkpointed."
            filter!(e -> e != p, properties)
        end
    end

    mkpath(dir)

    return Checkpointer(schedule, dir, prefix, properties, overwrite_existing, verbose, cleanup)
end

#####
##### Checkpointer utils
#####

checkpointer_address(::NonhydrostaticModel) = "NonhydrostaticModel"
checkpointer_address(::HydrostaticFreeSurfaceModel) = "HydrostaticFreeSurfaceModel"

""" Return the full prefix (the `superprefix`) associated with `checkpointer`. """
checkpoint_superprefix(prefix) = prefix * "_iteration"

# This is the default name used in the simulation.output_writers ordered dict.
defaultname(::Checkpointer, nelems) = :checkpointer

"""
    checkpoint_path(iteration::Int, c::Checkpointer)

Return the path to the `c`heckpointer file associated with model `iteration`.
"""
checkpoint_path(iteration::Int, c::Checkpointer) =
    joinpath(c.dir, string(checkpoint_superprefix(c.prefix), iteration, ".jld2"))

""" Returns `filepath`. Shortcut for `run!(simulation, pickup=filepath)`. """
checkpoint_path(filepath::String, output_writers) = filepath

function checkpoint_path(pickup, output_writers)
    checkpointers = filter(writer -> writer isa Checkpointer, collect(values(output_writers)))
    length(checkpointers) == 0 && error("No checkpointers found: cannot pickup simulation!")
    length(checkpointers) > 1 && error("Multiple checkpointers found: not sure which one to pickup simulation from!")
    return checkpoint_path(pickup, first(checkpointers))
end

"""
    checkpoint_path(pickup::Bool, checkpointer)

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

function write_output!(c::Checkpointer, model::AbstractModel)
    filepath = checkpoint_path(model.clock.iteration, c)
    c.verbose && @info "Checkpointing to file $filepath..."
    addr = checkpointer_address(model)

    t1 = time_ns()

    jldopen(filepath, "w") do file
        file["$addr/checkpointed_properties"] = c.properties
        serializeproperties!(file, model, c.properties, addr)
        model_fields = prognostic_fields(model)
        field_names = keys(model_fields)
        for name in field_names
            full_address = "$addr/$name"
            serializeproperty!(file, full_address, model_fields[name])
        end
    end

    t2, sz = time_ns(), filesize(filepath)
    c.verbose && @info "Checkpointing done: time=$(prettytime((t2 - t1) * 1e-9)), size=$(pretty_filesize(sz))"

    c.cleanup && cleanup_checkpoints(c)

    return nothing
end

function cleanup_checkpoints(checkpointer)
    filepaths = glob(checkpoint_superprefix(checkpointer.prefix) * "*.jld2", checkpointer.dir)
    latest_checkpoint_filepath = latest_checkpoint(checkpointer, filepaths)
    [rm(filepath) for filepath in filepaths if filepath != latest_checkpoint_filepath]
    return nothing
end

#####
##### set! for checkpointer filepaths
#####

"""
    set!(model, filepath::AbstractString)

Set data in `model.velocities`, `model.tracers`, `model.timestepper.Gⁿ`, and
`model.timestepper.G⁻` to checkpointed data stored at `filepath`.
"""
function set!(model::AbstractModel, filepath::AbstractString)

    addr = checkpointer_address(model)

    jldopen(filepath, "r") do file

        # Validate the grid
        checkpointed_grid = file["$addr/grid"]

        model.grid == checkpointed_grid ||
            @warn "The grid associated with $filepath and model.grid are not the same!"

        model_fields = prognostic_fields(model)

        for name in keys(model_fields)
            if string(name) ∈ keys(file[addr]) # Test if variable exists in checkpoint.
                model_field = model_fields[name]
                parent_data = file["$addr/$name/data"]
                copyto!(parent(model_field), parent_data)
            else
                @warn "Field $name does not exist in checkpoint and could not be restored."
            end
        end

        set_time_stepper!(model.timestepper, file, model_fields, addr)

        if !isnothing(model.particles)
            copyto!(model.particles.properties, file["$addr/particles"])
        end

        checkpointed_clock = file["$addr/clock"]

        # Update model clock
        model.clock.iteration = checkpointed_clock.iteration
        model.clock.time = checkpointed_clock.time
        model.clock.last_Δt = checkpointed_clock.last_Δt
    end

    return nothing
end

function set_time_stepper_tendencies!(timestepper, file, model_fields, addr)
    for name in propertynames(model_fields)
        if string(name) ∈ keys(file["$addr/timestepper/Gⁿ"]) # Test if variable tendencies exist in checkpoint
            # Tendency "n"
            parent_data = file["$addr/timestepper/Gⁿ/$name/data"]

            tendencyⁿ_field = timestepper.Gⁿ[name]
            copyto!(tendencyⁿ_field.data.parent, parent_data)

            # Tendency "n-1"
            parent_data = file["$addr/timestepper/G⁻/$name/data"]

            tendency⁻_field = timestepper.G⁻[name]
            copyto!(tendency⁻_field.data.parent, parent_data)
        else
            @warn "Tendencies for $name do not exist in checkpoint and could not be restored."
        end
    end

    return nothing
end

# For self-starting timesteppers like RK3 we do nothing
set_time_stepper!(timestepper, args...) = nothing

set_time_stepper!(timestepper::QuasiAdamsBashforth2TimeStepper, args...) =
    set_time_stepper_tendencies!(timestepper, args...)
