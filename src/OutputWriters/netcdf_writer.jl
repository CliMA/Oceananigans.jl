#####
##### NetCDFWriter struct definition
#####
##### NetCDFWriter functionality is implemented in ext/OceananigansNCDatasetsExt
#####

using Oceananigans.Grids: topology, Flat, StaticVerticalDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

#####
##### Dimension name generators
#####

function suffixed_dim_name_generator(var_name, grid::AbstractGrid{FT, TX}, LX, LY, LZ, dim::Val{:x}; connector="_", location_letters) where {FT, TX}
    if TX == Flat || isnothing(LX)
        return ""
    else
        return "$(var_name)" * connector * location_letters
    end
end

function suffixed_dim_name_generator(var_name, grid::AbstractGrid{FT, TX, TY}, LX, LY, LZ, dim::Val{:y}; connector="_", location_letters) where {FT, TX, TY}
    if TY == Flat || isnothing(LY)
        return ""
    else
        return "$(var_name)" * connector * location_letters
    end
end

function suffixed_dim_name_generator(var_name, grid::AbstractGrid{FT, TX, TY, TZ}, LX, LY, LZ, dim::Val{:z}; connector="_", location_letters) where {FT, TX, TY, TZ}
    if TZ == Flat || isnothing(LZ)
        return ""
    else
        return "$(var_name)" * connector * location_letters
    end
end

suffixed_dim_name_generator(var_name, ::StaticVerticalDiscretization, LX, LY, LZ, dim::Val{:z}; connector="_", location_letters) = var_name * connector * location_letters

loc2letter(::Face, full=true) = "f"
loc2letter(::Center, full=true) = "c"
loc2letter(::Nothing, full=true) = full ? "a" : ""

minimal_location_string(::RectilinearGrid, LX, LY, LZ, ::Val{:x}) = loc2letter(LX, false)
minimal_location_string(::RectilinearGrid, LX, LY, LZ, ::Val{:y}) = loc2letter(LY, false)

minimal_dim_name(var_name, grid, LX, LY, LZ, dim) =
    suffixed_dim_name_generator(var_name, grid, LX, LY, LZ, dim; connector="_", location_letters=minimal_location_string(grid, LX, LY, LZ, dim))
minimal_dim_name(var_name, grid::ImmersedBoundaryGrid, args...) = minimal_dim_name(var_name, grid.underlying_grid, args...)

trilocation_location_string(::RectilinearGrid, LX, LY, LZ, ::Val{:x}) = loc2letter(LX) * "aa"
trilocation_location_string(::RectilinearGrid, LX, LY, LZ, ::Val{:y}) = "a" * loc2letter(LY) * "a"

trilocation_location_string(::LatitudeLongitudeGrid, LX, LY, LZ, ::Val{:x}) = loc2letter(LX) * loc2letter(LY) * "a"
trilocation_location_string(::LatitudeLongitudeGrid, LX, LY, LZ, ::Val{:y}) = loc2letter(LX) * loc2letter(LY) * "a"

trilocation_location_string(grid::AbstractGrid,             LX, LY, LZ, dim::Val{:z}) = trilocation_location_string(grid.z, LX, LY, LZ, dim)
trilocation_location_string(::StaticVerticalDiscretization, LX, LY, LZ, dim::Val{:z}) = "aa" * loc2letter(LZ)
trilocation_location_string(grid,                           LX, LY, LZ, dim)          = loc2letter(LX) * loc2letter(LY) * loc2letter(LZ)

trilocation_dim_name(var_name, grid, LX, LY, LZ, dim) =
    suffixed_dim_name_generator(var_name, grid, LX, LY, LZ, dim, connector="_", location_letters=trilocation_location_string(grid, LX, LY, LZ, dim))

trilocation_dim_name(var_name, grid::ImmersedBoundaryGrid, args...) = trilocation_dim_name(var_name, grid.underlying_grid, args...)

dimension_name_generator_free_surface(dimension_name_generator, var_name, grid, LX, LY, LZ, dim) = dimension_name_generator(var_name, grid, LX, LY, LZ, dim)
dimension_name_generator_free_surface(dimension_name_generator, var_name, grid, LX, LY, LZ, dim::Val{:z}) = dimension_name_generator(var_name, grid, LX, LY, LZ, dim) * "_displacement"

mutable struct NetCDFWriter{G, D, O, T, A, FS, DN, DT} <: AbstractOutputWriter
    grid :: G
    filepath :: String
    dataset :: D
    outputs :: O
    schedule :: T
    array_type :: A
    indices :: Tuple
    global_attributes :: Dict
    output_attributes :: Dict
    dimensions :: Dict
    with_halos :: Bool
    include_grid_metrics :: Bool
    overwrite_existing :: Bool
    verbose :: Bool
    deflatelevel :: Int
    part :: Int
    file_splitting :: FS
    dimension_name_generator :: DN
    dimension_type :: DT
end

# method in OceananigansNCDatasetsExt
"""
    NetCDFWriter(model::AbstractModel, outputs;
                 filename,
                 schedule,
                 grid = model.grid,
                 dir = ".",
                 array_type = Array{Float32},
                 indices = (:, :, :),
                 global_attributes = Dict(),
                 output_attributes = Dict(),
                 dimensions = Dict(),
                 with_halos = false,
                 include_grid_metrics = true,
                 overwrite_existing = nothing,
                 verbose = false,
                 deflatelevel = 0,
                 part = 1,
                 file_splitting = NoFileSplitting(),
                 dimension_name_generator = trilocation_dim_name,
                 dimension_type = Float64)

Construct a `NetCDFWriter` that writes `(label, output)` pairs in `outputs` to a NetCDF file.

!!! note "NCDatasets required"
    `NetCDFWriter` requires NCDatasets.jl to be loaded: `using NCDatasets`

The `outputs` can be a `Dict` or `NamedTuple` where each `label` is a symbol or string and each `output` is one of:

- An `AbstractField` (e.g., `model.velocities.u`, `model.tracers.T`)
- An `AbstractOperation` or `Reduction` (e.g., `Average(model.tracers.T, dims=(1, 2))`)
- `LagrangianParticles` for particle tracking data
- A function `f(model)` that returns data to be written to disk

If any `outputs` are not `AbstractField`, `AbstractOperation`, `Reduction`, or `LagrangianParticles`,
their spatial `dimensions` must be provided as a `Dict` or `NamedTuple` mapping output names to
dimension name tuples.

Required arguments
==================

- `model`: The Oceananigans model instance.

- `outputs`: A collection of outputs to write, specified as either:
  * A `Dict` with `Symbol` or `String` keys and field/operation/function values
  * A `NamedTuple` of fields, operations, or functions

Required keyword arguments
==========================

- `filename`: Descriptive filename. `".nc"` is appended automatically if not present.

- `schedule`: An `AbstractSchedule` that determines when output is saved. Options include:
  * `TimeInterval(dt)`: Save every `dt` seconds of simulation time
  * `IterationInterval(n)`: Save every `n` iterations
  * `AveragedTimeInterval(dt; window, stride)`: Time-average output over a window before saving
  * `WallTimeInterval(dt)`: Save every `dt` seconds of wall clock time

Optional keyword arguments
==========================

- `grid`: The grid associated with `outputs`. Default: `model.grid`.
          Use this to specify a different grid when outputs are interpolated or regridded.

- `dir`: Directory to save output to. Default: `"."`.

- `array_type`: Type to convert outputs to before saving. Default: `Array{Float32}`.

- `indices`: Tuple of indices of the output variables to include. Default is `(:, :, :)`, which
             includes the full fields. This allows saving specific slices of the domain.

- `global_attributes`: `Dict` or `NamedTuple` of global attributes or metadata to save with every file.
                       Default: `Dict()`. This is useful for saving information specific to the simulation.
                       Some useful global attributes are included by default but will be overwritten if
                       included in this `Dict`.

- `output_attributes`: `Dict` or `NamedTuple` of attributes to save with each output variable.
                       Default: `Dict()`.
                       Reasonable defaults (long_name, units) are provided for standard variables
                       (u, v, w, T, S, b) and can be overwritten here.

- `dimensions`: A `Dict` or `NamedTuple` of dimension tuples to apply to outputs (required for function
                outputs that return custom data).

- `with_halos`: Boolean defining whether to include halos in the outputs. Default: `false`.
                Note that to postprocess saved output (e.g., compute derivatives, etc.),
                information about the boundary conditions is often crucial. In those cases,
                you might need to set `with_halos = true`. Cannot be used with custom `indices`.

- `include_grid_metrics`: Include grid metrics such as grid spacings, areas, and volumes as
                          additional variables. Default: `true`. Note that even with
                          `include_grid_metrics = false`, core grid coordinates are still saved.

- `overwrite_existing`: If `false`, `NetCDFWriter` will append to existing files. If `true`,
                        it will overwrite existing files or create new ones. Default: `true` if the
                        file does not exist, `false` if it does.

- `verbose`: Log variable compute times, file write times, and file sizes. Default: `false`.

- `deflatelevel`: Determines the NetCDF compression level of data (integer 0-9; 0 (default) means no compression
                  and 9 means maximum compression). See [NCDatasets.jl documentation](https://juliageo.org/NCDatasets.jl/stable/variables/#Creating-a-variable)
                  for more information.

- `part`: Starting part number for file splitting. Default: `1`.

- `file_splitting`: Schedule for splitting the output file. The new files will be suffixed with
                    `_part1`, `_part2`, etc. Options include:
                    * `FileSizeLimit(sz)`: Split when file size exceeds `sz` (e.g., `200KiB`).
                    * `TimeInterval(interval)`: Split every `interval` of simulation time.
                    * `NoFileSplitting()` (default): Don't split files.

- `dimension_name_generator`: A function with signature `(var_name, grid, LX, LY, LZ, dim)` where `dim` is
                              either `Val(:x)`, `Val(:y)`, or `Val(:z)` that returns a string corresponding
                              to the name of the dimension `var_name` on `grid` with location `(LX, LY, LZ)`
                              along `dim`. This advanced option can be used to rename dimensions and variables
                              to satisfy certain naming conventions. Default: `trilocation_dim_name`.

- `dimension_type`: Floating point type for dimension coordinate arrays. Default: `Float64`.
                    Use `Float32` to reduce file size if needed.

Examples
========

Saving the ``u`` velocity field and temperature fields, the full 3D fields and surface 2D slices
to separate NetCDF files:

```jldoctest netcdf1
using Oceananigans, NCDatasets

grid = RectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1))

model = NonhydrostaticModel(grid, tracers=:c)

simulation = Simulation(model, Δt=12, stop_time=3600)

fields = Dict("u" => model.velocities.u, "c" => model.tracers.c)

simulation.output_writers[:field_writer] =
    NetCDFWriter(model, fields, filename="fields.nc", schedule=TimeInterval(60))

# output

NetCDFWriter scheduled on TimeInterval(1 minute):
├── filepath: fields.nc
├── dimensions: time(0), y_afa(16), x_faa(16), x_caa(16), y_aca(16), z_aaf(17), z_aac(16)
├── 2 outputs: (c, u)
├── array_type: Array{Float32}
├── file_splitting: NoFileSplitting
└── file size: 32.6 KiB
```

```jldoctest netcdf1
simulation.output_writers[:surface_slice_writer] =
    NetCDFWriter(model, fields, filename="surface_xy_slice.nc",
                 schedule=TimeInterval(60), indices=(:, :, grid.Nz))

# output

NetCDFWriter scheduled on TimeInterval(1 minute):
├── filepath: surface_xy_slice.nc
├── dimensions: time(0), y_afa(16), x_faa(16), x_caa(16), y_aca(16), z_aaf(1), z_aac(1)
├── 2 outputs: (c, u)
├── array_type: Array{Float32}
├── file_splitting: NoFileSplitting
└── file size: 32.6 KiB
```

```jldoctest netcdf1
simulation.output_writers[:averaged_profile_writer] =
    NetCDFWriter(model, fields,
                 filename = "averaged_z_profile.nc",
                 schedule = AveragedTimeInterval(60, window=20),
                 indices = (1, 1, :))
# output

NetCDFWriter scheduled on TimeInterval(1 minute):
├── filepath: averaged_z_profile.nc
├── dimensions: time(0), y_afa(1), x_faa(1), x_caa(1), y_aca(1), z_aaf(17), z_aac(16)
├── 2 outputs: (c, u) averaged on AveragedTimeInterval(window=20 seconds, stride=1, interval=1 minute)
├── array_type: Array{Float32}
├── file_splitting: NoFileSplitting
└── file size: 33.8 KiB
```

`NetCDFWriter` also accepts output functions that write scalars and arrays to disk,
provided that their `dimensions` are provided:

```jldoctest netcdf2
using Oceananigans, NCDatasets

Nx, Ny, Nz = 16, 16, 16

grid = RectilinearGrid(size=(Nx, Ny, Nz), extent=(1, 2, 3))

model = NonhydrostaticModel(grid)

simulation = Simulation(model, Δt=1.25, stop_iteration=3)

f(model) = model.clock.time^2 # scalar output

zC = znodes(grid, Center())
g(model) = model.clock.time .* exp.(zC) # vector/profile output

xC, yF = xnodes(grid, Center()), ynodes(grid, Face())
XC = [xC[i] for i in 1:Nx, j in 1:Ny]
YF = [yF[j] for i in 1:Nx, j in 1:Ny]
h(model) = @. model.clock.time * sin(XC) * cos(YF) # xy slice output

outputs = Dict("scalar" => f, "profile" => g, "slice" => h)

dims = Dict("scalar" => (), "profile" => ("zC",), "slice" => ("xC", "yC"))

output_attributes = Dict(
    "scalar"  => Dict("long_name" => "Some scalar", "units" => "bananas"),
    "profile" => Dict("long_name" => "Some vertical profile", "units" => "watermelons"),
    "slice"   => Dict("long_name" => "Some slice", "units" => "mushrooms"))

global_attributes = Dict("location" => "Bay of Fundy", "onions" => 7)

simulation.output_writers[:things] =
    NetCDFWriter(model, outputs,
                 schedule=IterationInterval(1), filename="things.nc", dimensions=dims, verbose=true,
                 global_attributes=global_attributes, output_attributes=output_attributes)

# output

NetCDFWriter scheduled on IterationInterval(1):
├── filepath: things.nc
├── dimensions: time(0), y_afa(16), x_faa(16), x_caa(16), y_aca(16), z_aaf(17), z_aac(16)
├── 3 outputs: (profile, slice, scalar)
├── array_type: Array{Float32}
├── file_splitting: NoFileSplitting
└── file size: 31.5 KiB
```

`NetCDFWriter` can also be configured for `outputs` that are interpolated or regridded
to a different grid than `model.grid`. To use this functionality, include the keyword argument
`grid = output_grid`.

```jldoctest netcdf3
using Oceananigans, NCDatasets
using Oceananigans.Fields: interpolate!

grid = RectilinearGrid(size=(1, 1, 8), extent=(1, 1, 1));
model = NonhydrostaticModel(grid)

coarse_grid = RectilinearGrid(size=(grid.Nx, grid.Ny, grid.Nz÷2), extent=(grid.Lx, grid.Ly, grid.Lz))
coarse_u = Field{Face, Center, Center}(coarse_grid)

interpolate_u(model) = interpolate!(coarse_u, model.velocities.u)
outputs = (; u = interpolate_u)

output_writer = NetCDFWriter(model, outputs;
                             grid = coarse_grid,
                             filename = "coarse_u.nc",
                             schedule = IterationInterval(1))

# output

NetCDFWriter scheduled on IterationInterval(1):
├── filepath: coarse_u.nc
├── dimensions: time(0), y_afa(1), x_faa(1), x_caa(1), y_aca(1), z_aaf(5), z_aac(4)
├── 1 outputs: u
├── array_type: Array{Float32}
├── file_splitting: NoFileSplitting
└── file size: 31.4 KiB
```
"""
function NetCDFWriter(model, outputs; kw...)
    error("""
    NetCDFWriter is provided via an extension and requires NCDatasets.

    Fix:
      julia> using NCDatasets

      julia> NetCDFWriter(...)

    If NCDatasets isn't installed:
      julia> using Pkg; Pkg.add("NCDatasets")
    """)
end

function write_grid_reconstruction_data! end
function convert_for_netcdf end
function materialize_from_netcdf end
function reconstruct_grid end
