using Printf: @sprintf
using Zarr
using Oceananigans.Utils
using Oceananigans.Utils: TimeInterval, prettykeys, materialize_schedule
using Oceananigans.Fields: indices

struct TimeSnapshots end

mutable struct ZarrWriter{O, T, D, IF, IN, CS} <: AbstractOutputWriter
    filepath :: String
    outputs :: O
    schedule :: T
    array_type :: D
    init :: IF
    including :: IN
    part :: Int
    overwrite_existing :: Bool
    chunk_size :: CS
    verbose :: Bool
    initialized :: Bool
end

ext(::Type{ZarrWriter}) = ".zarr"

"""
    ZarrWriter(model, outputs; filename, schedule,
               dir = ".",
               indices = (:, :, :),
               with_halos = true,
               array_type = Array{Float32},
               overwrite_existing = false,
               init = noinit,
               chunk_size = TimeSnapshots(),
               including = default_included_properties(model),
               verbose = false,
               part = 1)

Construct a `ZarrWriter` for an Oceananigans `model` that writes `label, output` pairs
in `outputs` to a Zarr store.

The argument `outputs` may be a `Dict` or `NamedTuple`. The keys of `outputs` are symbols or
strings that "name" output data. The values of `outputs` are either `AbstractField`s, objects that
are called with the signature `output(model)`, or `WindowedTimeAverage`s of `AbstractFields`s,
functions, or callable objects.

Each output is stored as a separate ZArray within the Zarr group, with time appended along
the last dimension. A `t` array stores the simulation time for each snapshot.

Keyword arguments
=================

## Filenaming

- `filename` (required): Descriptive filename. `".zarr"` is appended to `filename` in the file path
                         if `filename` does not end in `".zarr"`.

- `dir`: Directory to save output to. Default: `"."` (current working directory).

## Output frequency and time-averaging

- `schedule` (required): `AbstractSchedule` that determines when output is saved.

## Slicing and type conversion prior to output

- `indices`: Specifies the indices to write to disk with a `Tuple` of `Colon`, `UnitRange`,
             or `Int` elements. Indices must be `Colon`, `Int`, or contiguous `UnitRange`.
             Defaults to `(:, :, :)` or "all indices". If `!with_halos`,
             halo regions are removed from `indices`. For example, `indices = (:, :, 1)`
             will save xy-slices of the bottom-most index.

- `with_halos` (Bool): Whether or not to slice off or keep halo regions from fields before writing output.
                       Preserving halo region data can be useful for postprocessing. Default: true.

- `array_type`: The array type to which output arrays are converted to prior to saving.
                Default: `Array{Float32}`.

## Chunking

- `chunk_size`: Controls the chunk size for the Zarr arrays. Default: `TimeSnapshots()`,
                which sets chunk sizes to the full spatial extent with a single time snapshot
                per chunk (i.e. `(Nx, Ny, Nz, 1)` for 3D fields). Can also be a `Tuple`
                specifying chunk sizes directly.

## File management

- `overwrite_existing`: Remove existing Zarr store if its path conflicts.
                        Default: `false`.

## Output file metadata management

- `including`: List of model properties to save with every file.
               Default depends on the type of model: `default_included_properties(model)`

## Miscellaneous keywords

- `verbose`: Log what the output writer is doing with statistics on compute/write times and file sizes.
             Default: `false`.
"""
function ZarrWriter(model, outputs; filename, schedule,
                    dir = ".",
                    indices = (:, :, :),
                    with_halos = true,
                    array_type = Array{Float32},
                    overwrite_existing = false,
                    init = noinit,
                    chunk_size = TimeSnapshots(),
                    including = default_included_properties(model),
                    verbose = false)

    mkpath(dir)
    filename = auto_extension(filename, ".zarr")
    filepath = abspath(joinpath(dir, filename))

    nt_outputs = NamedTuple(Symbol(name) => construct_output(outputs[name], indices, with_halos) for name in keys(outputs))
    schedule = materialize_schedule(schedule)

    # Convert each output to WindowedTimeAverage if schedule::AveragedTimeWindow is specified
    schedule, d_outputs = time_average_outputs(schedule, nt_outputs, model)

    return ZarrWriter(filepath, d_outputs, schedule, array_type,
                      including, overwrite_existing, chunk_size, verbose, false)
end

#####
##### Zarr store initialization
#####

compute_chunk_size(chunk_size::TimeSnapshots, data_size) = (data_size..., 1)
compute_chunk_size(chunk_size::Tuple, data_size) = chunk_size

function initialize_zarr_store!(filepath, outputs, writer, chunk_size, model)
    store = zgroup(filepath)

    # Create a ZArray for time
    zcreate(Float64, store, "t", 0; chunks=(1,))

    # Create a ZArray for each output, sampling to determine sizes
    for (name, output) in pairs(outputs)
        sample = fetch_and_convert_output(output, model, writer)
        data_size = size(sample)
        T = eltype(sample)
        chunks = compute_chunk_size(chunk_size, data_size)
        zcreate(T, store, string(name), data_size..., 0; chunks)
    end

    return nothing
end

"""
    initialize!(writer::ZarrWriter, model)

Initialize a `ZarrWriter` by creating its Zarr store and setting up arrays for each output.

This function is called automatically when a `Simulation` containing the writer is initialized
(typically during the first call to `run!`). The initialization is skipped if the writer
has already been initialized, preventing stores from being overwritten when `run!` is called
multiple times.

When resuming a simulation (i.e., `model.clock.iteration > 0`, such as when picking up from
a checkpoint), existing stores are preserved regardless of the `overwrite_existing` setting.
"""
function initialize!(writer::ZarrWriter, model)
    # Skip if already initialized (e.g., when run! is called multiple times)
    writer.initialized && return nothing

    # Remove existing store if overwrite_existing is true,
    # but only if we're starting fresh (iteration == 0).
    # When resuming (iteration > 0), we preserve existing stores.
    starting_fresh = model.clock.iteration == 0
    if writer.overwrite_existing && starting_fresh
        isdir(writer.filepath) && rm(writer.filepath, recursive=true, force=true)
    end

    if !isdir(writer.filepath)
        initialize_zarr_store!(writer.filepath, writer.outputs, writer,
                               writer.chunk_size, model)
    end

    writer.initialized = true

    return nothing
end

#####
##### Writing output
#####

function write_output!(writer::ZarrWriter, model)
    # Ensure the writer is initialized before writing
    if !writer.initialized
        initialize!(writer, model)
    end

    verbose = writer.verbose

    verbose && @info @sprintf("Fetching Zarr output %s...", keys(writer.outputs))

    tc = Base.@elapsed data = NamedTuple(name => fetch_and_convert_output(output, model, writer)
                                         for (name, output) in zip(keys(writer.outputs),
                                                                   values(writer.outputs)))

    verbose && @info "Fetching time: $(prettytime(tc))"
    verbose && @info "Writing Zarr output $(keys(writer.outputs)) to $(writer.filepath)..."

    start_time = time_ns()
    store = zopen(writer.filepath, "w")

    # Append time
    append!(store["t"], [model.clock.time])

    # Append each output along the last (time) dimension
    for name in keys(data)
        arr = data[name]
        append!(store[string(name)], arr)
    end

    end_time = time_ns()

    verbose && @info @sprintf("Writing done: time=%s", prettytime((end_time - start_time) / 1e9))

    return nothing
end

#####
##### Show methods
#####

Base.summary(ow::ZarrWriter) =
    string("ZarrWriter writing ", prettykeys(ow.outputs), " to ", ow.filepath, " on ", summary(ow.schedule))

function Base.show(io::IO, ow::ZarrWriter)
    averaging_schedule = output_averaging_schedule(ow)
    Noutputs = length(ow.outputs)

    print(io, "ZarrWriter scheduled on $(summary(ow.schedule)):", "\n",
              "├── filepath: ", relpath(ow.filepath), "\n",
              "├── $Noutputs outputs: ", prettykeys(ow.outputs), show_averaging_schedule(averaging_schedule), "\n",
              "├── array_type: ", show_array_type(ow.array_type), "\n",
              "├── chunk_size: ", ow.chunk_size, "\n",
              "└── including: ", ow.including)
end
