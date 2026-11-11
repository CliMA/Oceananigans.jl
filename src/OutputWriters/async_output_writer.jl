using Base.Threads: @spawn

#####
##### Async I/O for output writers
#####
##### `AbstractOutputWriter{A}` is parameterized by an I/O mode tag `A`,
##### either `Synchronous` (the default) or `Asynchronous`. Asynchronous
##### writers run their disk-write phase on a background task so GPU
##### computation can continue while the CPU writes output to disk. The
##### GPU→CPU data transfer always happens synchronously, since the data
##### must be captured before the GPU advances to the next time step.
#####
##### Writers opt into async support by specializing two functions:
#####
#####     prepare_async_write(writer, model)  -> snapshot
#####     commit_async_write!(writer, snapshot)
#####
##### Synchronous `write_output!` runs both inline; asynchronous
##### `write_output!` runs `prepare_async_write` inline and spawns
##### `commit_async_write!` on a background task.
#####

"""
    Synchronous

Singleton stored in an output writer's `asynchronous` property, indicating that its
disk-write phase runs on the main thread. The default mode for all output writers.
"""
struct Synchronous end

"""
    Asynchronous

Stored in an output writer's `asynchronous` property, indicating that its disk-write
phase runs on a background task. Holds the in-flight write `task`. The synchronous
GPU→CPU copy still happens on the main thread, so output content is identical to the
synchronous case. Only the slow disk write is deferred.

Use `wait_for_async_writes!` to flush pending writes — this is called
automatically at the end of `run!`.
"""
mutable struct Asynchronous
    task :: Union{Task, Nothing}
end

Asynchronous() = Asynchronous(nothing)

io_mode(asynchronous) = asynchronous ? Asynchronous() : Synchronous()

"""
    AsyncOutputWriter

Type alias for `AbstractOutputWriter{Asynchronous}`. Matches any output writer
configured to run its disk-write phase on a background task.
"""
const AsyncOutputWriter = AbstractOutputWriter{Asynchronous}

"""
    SyncOutputWriter

Type alias for `AbstractOutputWriter{Synchronous}`.
"""
const SyncOutputWriter = AbstractOutputWriter{Synchronous}

"""
    is_asynchronous(writer)

Return `true` if `writer` is configured for async I/O.
"""
is_asynchronous(::AbstractOutputWriter{Asynchronous}) = true
is_asynchronous(::AbstractOutputWriter) = false

#####
##### Async task lifecycle
#####

"""
    wait_for_async_writes!(writer::AsyncOutputWriter)

Block until any in-flight async write started by `writer` has completed. If the
background task threw an exception, it is re-thrown here. No-op for
synchronous writers.
"""
function wait_for_async_writes!(writer::AsyncOutputWriter)
    task = writer.asynchronous.task
    if task !== nothing
        try
            wait(task)
        finally
            writer.asynchronous.task = nothing
        end
    end
    return nothing
end

wait_for_async_writes!(::AbstractOutputWriter) = nothing

"""
    wait_for_async_writes!(simulation)

Block until all async output writers in `simulation.output_writers` have flushed
their pending writes. Called automatically at the end of `run!`.
"""
function wait_for_async_writes!(sim)
    for writer in values(sim.output_writers)
        wait_for_async_writes!(writer)
    end
    return nothing
end

#####
##### Async write_output!
#####
##### Synchronous writers' `write_output!` is defined per writer (e.g. for
##### JLD2Writer it runs `prepare_async_write` then `commit_async_write!`
##### inline). Asynchronous writers share this generic dispatch:
#####

"""
    write_output!(writer::AsyncOutputWriter, model::AbstractModel)

Wait for any prior async write to complete, then synchronously fetch output
data (GPU→CPU) via `prepare_async_write` and spawn a background task that
calls `commit_async_write!`. The `::AbstractModel` constraint disambiguates
from the `::Simulation` pass-through defined in `Simulations/simulation.jl`.
"""
Oceananigans.write_output!(writer::AsyncOutputWriter, model::AbstractModel) = spawn_async_write!(writer, model)

"""
    spawn_async_write!(writer, model_or_simulation)

Shared body of the asynchronous `write_output!`: wait for the prior write, run
`prepare_async_write` inline, and spawn `commit_async_write!` on a background task.
Writers whose `write_output!` takes something other than an `AbstractModel`
(the `Checkpointer`, which needs the `Simulation`) call this directly.
"""
function spawn_async_write!(writer, model)
    # Make sure the previous async write has finished. This both avoids
    # unbounded task accumulation and re-throws any exception from the prior
    # write at the next synchronization point on the main thread.
    wait_for_async_writes!(writer)

    snapshot = prepare_async_write(writer, model)

    # `nothing` indicates no write should occur (e.g. iteration already exists
    # and we don't overwrite).
    snapshot === nothing && return nothing

    writer.asynchronous.task = @spawn commit_in_background!(writer, snapshot)

    return nothing
end

function commit_in_background!(writer, snapshot)
    try
        commit_async_write!(writer, snapshot)
    catch err
        @error "Async output writer failed" exception=(err, catch_backtrace())
        rethrow()
    end
    return nothing
end

#####
##### CPU aliasing validation
#####
##### `fetch_and_convert_output` returns the field's underlying array without
##### copying when the field's eltype matches `eltype(array_type)` on CPU. In
##### async mode this is a data race: the worker thread reads the array while
##### the main thread mutates it during the next `time_step!`. We refuse to
##### construct such a writer.
#####

output_aliases_live_data(out, FT) = false
output_aliases_live_data(field::AbstractField, FT) = eltype(field) === FT
output_aliases_live_data(wta::WindowedTimeAverage{<:AbstractField}, FT) = eltype(wta.operand) === FT

"""
    validate_async_outputs(outputs, array_type, model)

Throw an `ArgumentError` if any output would alias live model state when written
asynchronously on a CPU model. Aliasing happens when the output is an
`AbstractField` (or `WindowedTimeAverage` of one) whose `eltype` matches
`eltype(array_type)` — `convert_output` then returns the input array unchanged.
On GPU the conversion always copies, so this check is a no-op.
"""
function validate_async_outputs(outputs, array_type, model)
    architecture(model) isa GPU && return nothing
    array_eltype = eltype(array_type)
    aliasing = String[]
    for (name, out) in pairs(outputs)
        output_aliases_live_data(out, array_eltype) && push!(aliasing, string(name))
    end
    isempty(aliasing) && return nothing
    throw(ArgumentError(
        "Asynchronous output on CPU would alias live model state for outputs " *
        "$aliasing: their eltype matches `array_type = $array_type`, so " *
        "`fetch_and_convert_output` returns the live array without copying and the " *
        "worker thread would race with the next `time_step!`. " *
        "Either pick an `array_type` whose eltype differs from the field eltype " *
        "(e.g., `Array{Float32}` for a Float64 model), or set `asynchronous=false`."))
end

"""
    OutputSnapshot

Everything needed to commit a write to disk after the GPU→CPU copy has completed.
Created by `prepare_async_write` on the main thread and consumed by `commit_async_write!`.
`destination` (a filepath or a store) is captured by value so that a later file split
does not redirect an in-flight write.
"""
struct OutputSnapshot{T, D, P}
    iteration :: Int
    time :: T
    data :: D
    destination :: P
end

"""
    prepare_async_write(writer, model)

Synchronously perform the GPU→CPU portion of `write_output!` and return an
`OutputSnapshot` that contains everything needed to commit the write to disk.
Return `nothing` to indicate no write should occur.

Each writer that supports async I/O must specialize this function.
"""
function prepare_async_write end

"""
    commit_async_write!(writer, snapshot)

Run the disk-write portion of `write_output!`. Called either inline (sync
mode) or from a background task (async mode), so it must not touch GPU state
or any data that the main thread may mutate concurrently with the write.

Each writer that supports async I/O must specialize this function.
"""
function commit_async_write! end
