#####
##### ZarrWriter struct definition
#####
##### ZarrWriter functionality is implemented in ext/OceananigansZarrExt
#####

mutable struct ZarrWriter{O, T, S, A, FS, C, CH, G} <: AbstractOutputWriter
    filepath :: String
    store :: S
    grids :: G                  # Tuple of unique grids across all outputs
    output_grid_map :: Dict     # name (String) -> grid index in `grids` (Int) or Nothing
    outputs :: O
    schedule :: T
    array_type :: A
    indices :: Tuple
    with_halos :: Bool
    overwrite_existing :: Bool
    verbose :: Bool
    part :: Int
    file_splitting :: FS
    compressor :: C
    chunks :: CH
    dimensions :: Dict
    initialized :: Bool
end

# method in OceananigansZarrExt
"""
    ZarrWriter(model, outputs; filename, schedule,
               dir = ".",
               indices = (:, :, :),
               with_halos = true,
               array_type = Array{Float32},
               file_splitting = NoFileSplitting(),
               overwrite_existing = false,
               verbose = false,
               part = 1,
               store = nothing,
               chunks = nothing,
               compressor = nothing)

Construct a `ZarrWriter` for an Oceananigans `model` that writes `label, output` pairs in
`outputs` to a Zarr store.

!!! note "Zarr required"
    `ZarrWriter` requires Zarr.jl to be loaded: `using Zarr`

The argument `outputs` may be a `Dict` or `NamedTuple`. The keys of `outputs` are symbols
or strings that name output data. The values of `outputs` are `AbstractField`s,
`AbstractOperation`s, `Reduction`s, `WindowedTimeAverage`s, or functions that take a
`model` and return data.

Each output is stored as a chunked Zarr array of shape `(field_dims..., time)`, growing
along the time axis. A top-level `time` array tracks simulation time at each step. Grid
reconstruction data is stored under a `grid/` subgroup; multi-grid writers use
`grid_1/`, `grid_2/`, ... .

Keyword arguments
=================

## Filenaming / storage

- `filename` (required unless `store` is provided): Name of the Zarr store. `".zarr"` is
                appended if not present. On distributed architectures the local rank is
                *not* appended — all ranks write into the same store.

- `dir`: Directory to save output to. Default: `"."` (current working directory).

- `store`: Pre-constructed Zarr store (e.g. `Zarr.DictStore()`, `Zarr.S3Store(...)`).
           If provided, `filename` and `dir` are ignored. `Zarr.ZipStore` is *not*
           accepted — it is read-only in Zarr.jl; write to a `DirectoryStore` and finalize
           with `Zarr.writezip(io, writer.store)` if you want a single-file artifact.

## Output frequency and time-averaging

- `schedule` (required): `AbstractSchedule` that determines when output is saved.

## Slicing and type conversion prior to output

- `indices`: Tuple of `Colon`, `UnitRange`, or `Int` specifying the slice of each field to
             write. Default: `(:, :, :)`.

- `with_halos`: Whether to include halo regions. Default: `true`.

- `array_type`: The array type. Only `eltype(array_type)` is used — it sets the Zarr
                array's on-disk `dtype`, which is fixed for the array's lifetime. Default:
                `Array{Float32}`.

## Chunking and compression

- `chunks`: Tuple specifying the chunk shape (excluding the time axis, which is always
            chunked at 1). Default `nothing` chooses a sensible per-axis default: the
            full local extent on each axis for serial runs; GCD of local extents across
            ranks for distributed runs. Each component must divide every rank's local
            extent along its axis.

- `compressor`: A Zarr compressor (e.g. `Zarr.BloscCompressor()`, `Zarr.ZlibCompressor()`)
                or `nothing` for no compression. Default: `nothing`.

## File management

- `file_splitting`: Schedule for splitting the output store into successive parts. Options
                    include `NoFileSplitting()` (default), `FileSizeLimit(sz)`,
                    `TimeInterval(Δt)`.

- `overwrite_existing`: Remove an existing store before writing. Default: `false`. When
                        `false` and the store already exists, the writer appends new
                        timesteps to the existing time axis.

## Miscellaneous

- `verbose`: Log compute/write times and sizes. Default: `false`.

- `part`: Starting part number for file splitting. Default: `1`.

Reading output back
===================

A serial reader works on any `ZarrWriter` output, including output produced by a
distributed-MPI run:

```julia
using Oceananigans, Zarr

fts = FieldTimeSeries("output.zarr", "u")
```

Convert a `DirectoryStore` to a single zip file at the end of a run:

```julia
open("output.zip", "w") do io
    Zarr.writezip(io, writer.store)
end
```
"""
function ZarrWriter(model, outputs; kw...)
    error("""
    ZarrWriter is provided via an extension and requires Zarr.

    Fix:
      julia> using Zarr

      julia> ZarrWriter(...)

    If Zarr isn't installed:
      julia> using Pkg; Pkg.add("Zarr")
    """)
end

# Hooks for the extension to fill in.
function initialize_zarr_store! end
function write_zarr_grid_reconstruction! end
function reconstruct_zarr_grid end
