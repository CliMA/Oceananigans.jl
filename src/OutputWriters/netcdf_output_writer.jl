using NCDatasets

using Dates: AbstractTime, UTC, now

using Oceananigans.Fields

using Oceananigans.Grids: AbstractCurvilinearGrid, RectilinearGrid, StaticVerticalCoordinate
using Oceananigans.Grids: topology, halo_size, parent_index_range, ξnodes, ηnodes, rnodes, validate_index, peripheral_node
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GFBIBG
using Oceananigans.Utils: versioninfo_with_gpu, oceananigans_versioninfo, prettykeys
using Oceananigans.TimeSteppers: float_or_date_time
using Oceananigans.Fields: reduced_dimensions, reduced_location, location

#####
##### Utils
#####

dictify(outputs) = outputs
dictify(outputs::NamedTuple) = Dict(string(k) => dictify(v) for (k, v) in zip(keys(outputs), values(outputs)))

function collect_dim(ξ, ℓ, T, N, H, inds, with_halos)
    if with_halos
        # collect to ensure we return an array instead of a range or offset array
        return collect(ξ)
    else
        inds = validate_index(inds, ℓ, T, N, H)
        inds = restrict_to_interior(inds, ℓ, T, N)
        return collect(view(ξ, inds))
    end
end

#####
##### Dimension name generator (the default one)
#####

loc2letter(::Face) = "f"
loc2letter(::Center) = "c"
loc2letter(::Nothing) = ""

function default_dim_name(var_name, ::RectilinearGrid{FT, TX}, LX, LY, LZ, ::Val{:x}) where {FT, TX}
    if TX == Flat || isnothing(LX)
        return ""
    else
        return "$(var_name)_" * loc2letter(LX)
    end
end

function default_dim_name(var_name, ::RectilinearGrid{FT, TX, TY}, LX, LY, LZ, ::Val{:y}) where {FT, TX, TY}
    if TY == Flat || isnothing(LY)
        return ""
    else
        return "$(var_name)_" * loc2letter(LY)
    end
end

function default_dim_name(var_name, ::RectilinearGrid{FT, TX, TY, TZ}, LX, LY, LZ, ::Val{:z}) where {FT, TX, TY, TZ}
    if TZ == Flat || isnothing(LZ)
        return ""
    else
        return "$(var_name)_" * loc2letter(LZ)
    end
end

default_dim_name(var_name, ::StaticVerticalCoordinate, LX, LY, LZ, ::Val{:z}) = "$(var_name)_" * loc2letter(LZ)

default_dim_name(var_name, grid, LX, LY, LZ, dim) = "$(var_name)_" * loc2letter(LX) * loc2letter(LY) * loc2letter(LZ)

default_dim_name(var_name, grid::ImmersedBoundaryGrid, args...) =
    default_dim_name(var_name, grid.underlying_grid, args...)

#####
##### Gathering of grid dimensions
#####

function maybe_add_particle_dims!(dims, outputs)
    if "particles" in keys(outputs)  # TODO: Change this to look for ::LagrangianParticles in outputs?
        dims["particle_id"] = collect(1:length(outputs["particles"]))
    end
    return dims
end

function gather_vertical_dimensions(coordinate::StaticVerticalCoordinate, TZ, Nz, Hz, z_indices, with_halos, dim_name_generator)
    zᵃᵃᶠ_name = dim_name_generator("z", coordinate, nothing, nothing, f, Val(:z))
    zᵃᵃᶜ_name = dim_name_generator("z", coordinate, nothing, nothing, c, Val(:z))

    zᵃᵃᶠ_data = collect_dim(coordinate.cᵃᵃᶠ, f, TZ(), Nz, Hz, z_indices, with_halos)
    zᵃᵃᶜ_data = collect_dim(coordinate.cᵃᵃᶜ, c, TZ(), Nz, Hz, z_indices, with_halos)

    return Dict(
        zᵃᵃᶠ_name => zᵃᵃᶠ_data,
        zᵃᵃᶜ_name => zᵃᵃᶜ_data
    )
end

function gather_dimensions(outputs, grid::RectilinearGrid, indices, with_halos, dim_name_generator)
    TX, TY, TZ = topology(grid)
    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    dims = Dict()

    if TX != Flat
        xᶠᵃᵃ_name = dim_name_generator("x", grid, f, nothing, nothing, Val(:x))
        xᶜᵃᵃ_name = dim_name_generator("x", grid, c, nothing, nothing, Val(:x))

        xᶠᵃᵃ_data = collect_dim(grid.xᶠᵃᵃ, f, TX(), Nx, Hx, indices[1], with_halos)
        xᶜᵃᵃ_data = collect_dim(grid.xᶜᵃᵃ, c, TX(), Nx, Hx, indices[1], with_halos)

        dims[xᶠᵃᵃ_name] = xᶠᵃᵃ_data
        dims[xᶜᵃᵃ_name] = xᶜᵃᵃ_data
    end

    if TY != Flat
        yᵃᶠᵃ_name = dim_name_generator("y", grid, nothing, f, nothing, Val(:y))
        yᵃᶜᵃ_name = dim_name_generator("y", grid, nothing, c, nothing, Val(:y))

        yᵃᶠᵃ_data = collect_dim(grid.yᵃᶠᵃ, f, TY(), Ny, Hy, indices[2], with_halos)
        yᵃᶜᵃ_data = collect_dim(grid.yᵃᶜᵃ, c, TY(), Ny, Hy, indices[2], with_halos)

        dims[yᵃᶠᵃ_name] = yᵃᶠᵃ_data
        dims[yᵃᶜᵃ_name] = yᵃᶜᵃ_data
    end

    if TZ != Flat
        vertical_dims = gather_vertical_dimensions(grid.z, TZ, Nz, Hz, indices[3], with_halos, dim_name_generator)
        dims = merge(dims, vertical_dims)
    end

    maybe_add_particle_dims!(dims, outputs)

    return dims
end

gather_dimensions(outputs, grid::ImmersedBoundaryGrid, args...) =
    gather_dimensions(outputs, grid.underlying_grid, args...)

#####
##### Gathering of grid metrics
#####

function gather_grid_metrics(grid::RectilinearGrid, indices, dim_name_generator)
    TX, TY, TZ = topology(grid)

    metrics = Dict()

    if TX != Flat
        Δxᶠᵃᵃ_name = dim_name_generator("dx", grid, f, nothing, nothing, Val(:x))
        Δxᶜᵃᵃ_name = dim_name_generator("dx", grid, c, nothing, nothing, Val(:x))

        Δxᶠᵃᵃ_field = Field(xspacings(grid, f); indices)
        Δxᶜᵃᵃ_field = Field(xspacings(grid, c); indices)

        metrics[Δxᶠᵃᵃ_name] = Δxᶠᵃᵃ_field
        metrics[Δxᶜᵃᵃ_name] = Δxᶜᵃᵃ_field
    end

    if TY != Flat
        Δyᵃᶠᵃ_name = dim_name_generator("dy", grid, nothing, f, nothing, Val(:y))
        Δyᵃᶜᵃ_name = dim_name_generator("dy", grid, nothing, c, nothing, Val(:y))

        Δyᵃᶠᵃ_field = Field(yspacings(grid, f); indices)
        Δyᵃᶜᵃ_field = Field(yspacings(grid, c); indices)

        metrics[Δyᵃᶠᵃ_name] = Δyᵃᶠᵃ_field
        metrics[Δyᵃᶜᵃ_name] = Δyᵃᶜᵃ_field
    end

    if TZ != Flat
        Δzᵃᵃᶠ_name = dim_name_generator("dz", grid, nothing, nothing, f, Val(:z))
        Δzᵃᵃᶜ_name = dim_name_generator("dz", grid, nothing, nothing, c, Val(:z))

        Δzᵃᵃᶠ_field = Field(zspacings(grid, f); indices)
        Δzᵃᵃᶜ_field = Field(zspacings(grid, c); indices)

        metrics[Δzᵃᵃᶠ_name] = Δzᵃᵃᶠ_field
        metrics[Δzᵃᵃᶜ_name] = Δzᵃᵃᶜ_field
    end

    return metrics
end

gather_grid_metrics(grid::ImmersedBoundaryGrid, args...) =
    gather_grid_metrics(grid.underlying_grid, args...)

#####
##### Gathering of immersed boundary fields
#####

# TODO: Proper masks for 2D models?
flat_loc(T, L) = T == Flat ? nothing : L

# For Immersed Boundary Grids (IBG) with a Grid Fitted Bottom (GFB)
function gather_immersed_boundary(grid::GFBIBG, indices, dim_name_generator)
    op_mask_ccc = KernelFunctionOperation{Center, Center, Center}(peripheral_node, grid, Center(), Center(), Center())
    op_mask_fcc = KernelFunctionOperation{Face, Center, Center}(peripheral_node, grid, Face(), Center(), Center())
    op_mask_cfc = KernelFunctionOperation{Center, Face, Center}(peripheral_node, grid, Center(), Face(), Center())
    op_mask_ccf = KernelFunctionOperation{Center, Center, Face}(peripheral_node, grid, Center(), Center(), Face())

    return Dict(
        "bottom_height" => Field(grid.immersed_boundary.bottom_height; indices),
        "immersed_boundary_mask_ccc" => Field(op_mask_ccc; indices),
        "immersed_boundary_mask_fcc" => Field(op_mask_fcc; indices),
        "immersed_boundary_mask_cfc" => Field(op_mask_cfc; indices),
        "immersed_boundary_mask_ccf" => Field(op_mask_ccf; indices)
    )
end

#####
##### Mapping outputs/fields to dimensions
#####

function field_dimensions(field::AbstractField{LX, LY, LZ}, dim_name_generator) where {LX, LY, LZ}
    x_dim_name = dim_name_generator("x", field.grid, LX(), LY(), LZ(), Val(:x))
    y_dim_name = dim_name_generator("y", field.grid, LX(), LY(), LZ(), Val(:y))
    z_dim_name = dim_name_generator("z", field.grid, LX(), LY(), LZ(), Val(:z))

    x_dim_name = isempty(x_dim_name) ? tuple() : tuple(x_dim_name)
    y_dim_name = isempty(y_dim_name) ? tuple() : tuple(y_dim_name)
    z_dim_name = isempty(z_dim_name) ? tuple() : tuple(z_dim_name)

    return tuple(x_dim_name..., y_dim_name..., z_dim_name...)
end

#####
##### Dimension attributes
#####

const base_dimension_attributes = Dict(
    "time"        => Dict("long_name" => "Time", "units" => "s"),
    "particle_id" => Dict("long_name" => "Particle ID")
)

function default_vertical_dimension_attributes(coordinate::StaticVerticalCoordinate, dim_name_generator)
    zᵃᵃᶠ_name = dim_name_generator("z", coordinate, nothing, nothing, f, Val(:z))
    zᵃᵃᶜ_name = dim_name_generator("z", coordinate, nothing, nothing, c, Val(:z))

    Δzᵃᵃᶠ_name = dim_name_generator("dz", coordinate, nothing, nothing, f, Val(:z))
    Δzᵃᵃᶜ_name = dim_name_generator("dz", coordinate, nothing, nothing, c, Val(:z))

    zᵃᵃᶠ_attrs = Dict("long_name" => "Locations of the cell faces in the z-direction.",   "units" => "m")
    zᵃᵃᶜ_attrs = Dict("long_name" => "Locations of the cell centers in the z-direction.", "units" => "m")

    Δzᵃᵃᶠ_attrs = Dict("long_name" => "Spacings between the cell centers (located at the cell faces) in the z-direction.", "units" => "m")
    Δzᵃᵃᶜ_attrs = Dict("long_name" => "Spacings between the cell faces (located at the cell centers) in the z-direction.", "units" => "m")

    return Dict(
        zᵃᵃᶠ_name => zᵃᵃᶠ_attrs,
        zᵃᵃᶜ_name => zᵃᵃᶜ_attrs,
        Δzᵃᵃᶠ_name => Δzᵃᵃᶠ_attrs,
        Δzᵃᵃᶜ_name => Δzᵃᵃᶜ_attrs,
    )
end

function default_dimension_attributes(grid::RectilinearGrid, dim_name_generator)
    vertical_dimension_attributes = default_vertical_dimension_attributes(grid.z, dim_name_generator)

    xᶠᵃᵃ_name = dim_name_generator("x", grid, f, nothing, nothing, Val(:x))
    xᶜᵃᵃ_name = dim_name_generator("x", grid, c, nothing, nothing, Val(:x))
    yᵃᶠᵃ_name = dim_name_generator("y", grid, nothing, f, nothing, Val(:y))
    yᵃᶜᵃ_name = dim_name_generator("y", grid, nothing, c, nothing, Val(:y))

    Δxᶠᵃᵃ_name = dim_name_generator("dx", grid, f, nothing, nothing, Val(:x))
    Δxᶜᵃᵃ_name = dim_name_generator("dx", grid, c, nothing, nothing, Val(:x))
    Δyᵃᶠᵃ_name = dim_name_generator("dy", grid, nothing, f, nothing, Val(:y))
    Δyᵃᶜᵃ_name = dim_name_generator("dy", grid, nothing, c, nothing, Val(:y))

    xᶠᵃᵃ_attrs = Dict("long_name" => "Locations of the cell faces in the x-direction.",   "units" => "m")
    xᶜᵃᵃ_attrs = Dict("long_name" => "Locations of the cell centers in the x-direction.", "units" => "m")
    yᵃᶠᵃ_attrs = Dict("long_name" => "Locations of the cell faces in the y-direction.",   "units" => "m")
    yᵃᶜᵃ_attrs = Dict("long_name" => "Locations of the cell centers in the y-direction.", "units" => "m")

    Δxᶠᵃᵃ_attrs = Dict("long_name" => "Spacings between the cell centers (located at the cell faces) in the x-direction.", "units" => "m")
    Δxᶜᵃᵃ_attrs = Dict("long_name" => "Spacings between the cell faces (located at the cell centers) in the x-direction.", "units" => "m")
    Δyᵃᶠᵃ_attrs = Dict("long_name" => "Spacings between the cell centers (located at the cell faces) in the y-direction.", "units" => "m")
    Δyᵃᶜᵃ_attrs = Dict("long_name" => "Spacings between the cell faces (located at the cell centers) in the y-direction.", "units" => "m")

    horizontal_dimension_attributes = Dict(
        xᶠᵃᵃ_name => xᶠᵃᵃ_attrs,
        xᶜᵃᵃ_name => xᶜᵃᵃ_attrs,
        yᵃᶠᵃ_name => yᵃᶠᵃ_attrs,
        yᵃᶜᵃ_name => yᵃᶜᵃ_attrs,
        Δxᶠᵃᵃ_name => Δxᶠᵃᵃ_attrs,
        Δxᶜᵃᵃ_name => Δxᶜᵃᵃ_attrs,
        Δyᵃᶠᵃ_name => Δyᵃᶠᵃ_attrs,
        Δyᵃᶜᵃ_name => Δyᵃᶜᵃ_attrs
    )

    return merge(
        base_dimension_attributes,
        vertical_dimension_attributes,
        horizontal_dimension_attributes
    )
end

default_dimension_attributes(grid::ImmersedBoundaryGrid, dim_name_generator) =
    default_dimension_attributes(grid.underlying_grid, dim_name_generator)

#####
##### Variable attributes
#####

const default_output_attributes = Dict(
    "u" => Dict("long_name" => "Velocity in the x-direction", "units" => "m/s"),
    "v" => Dict("long_name" => "Velocity in the y-direction", "units" => "m/s"),
    "w" => Dict("long_name" => "Velocity in the z-direction", "units" => "m/s"),
    "b" => Dict("long_name" => "Buoyancy",                    "units" => "m/s²"),
    "T" => Dict("long_name" => "Conservative temperature",    "units" => "°C"),
    "S" => Dict("long_name" => "Absolute salinity",           "units" => "g/kg")
)

#####
##### Saving schedule metadata as global attributes
#####

add_schedule_metadata!(attributes, schedule) = nothing

function add_schedule_metadata!(global_attributes, schedule::IterationInterval)
    global_attributes["schedule"] = "IterationInterval"
    global_attributes["interval"] = schedule.interval
    global_attributes["output iteration interval"] =
        "Output was saved every $(schedule.interval) iteration(s)."

    return nothing
end

function add_schedule_metadata!(global_attributes, schedule::TimeInterval)
    global_attributes["schedule"] = "TimeInterval"
    global_attributes["interval"] = schedule.interval
    global_attributes["output time interval"] =
        "Output was saved every $(prettytime(schedule.interval))."

    return nothing
end

function add_schedule_metadata!(global_attributes, schedule::WallTimeInterval)
    global_attributes["schedule"] = "WallTimeInterval"
    global_attributes["interval"] = schedule.interval
    global_attributes["output time interval"] =
        "Output was saved every $(prettytime(schedule.interval))."

    return nothing
end

function add_schedule_metadata!(global_attributes, schedule::AveragedTimeInterval)
    global_attributes["schedule"] = "AveragedTimeInterval"
    global_attributes["interval"] = schedule.interval
    global_attributes["output time interval"] =
        "Output was time-averaged and saved every $(prettytime(schedule.interval))."

    global_attributes["time_averaging_window"] = schedule.window
    global_attributes["time averaging window"] =
        "Output was time averaged with a window size of $(prettytime(schedule.window))"

    global_attributes["time_averaging_stride"] = schedule.stride
    global_attributes["time averaging stride"] =
        "Output was time averaged with a stride of $(schedule.stride) iteration(s) within the time averaging window."

    return nothing
end

#####
##### NetCDFOutputWriter definition and constructor
#####

mutable struct NetCDFOutputWriter{G, D, O, T, A, FS, DN} <: AbstractOutputWriter
    grid :: G
    filepath :: String
    dataset :: D
    outputs :: O
    schedule :: T
    array_type :: A
    indices :: Tuple
    with_halos :: Bool
    global_attributes :: Dict
    output_attributes :: Dict
    dimensions :: Dict
    overwrite_existing :: Bool
    deflatelevel :: Int
    part :: Int
    file_splitting :: FS
    verbose :: Bool
    dimension_name_generator :: DN
    include_grid_metrics :: Bool
end

"""
    NetCDFOutputWriter(model, outputs; filename, schedule,
                       grid = model.grid,
                       dir = ".",
                       array_type = Array{Float64},
                       indices = nothing,
                       with_halos = false,
                       global_attributes = Dict(),
                       output_attributes = Dict(),
                       dimensions = Dict(),
                       overwrite_existing = false,
                       deflatelevel = 0,
                       part = 1,
                       file_splitting = NoFileSplitting(),
                       verbose = false)

Construct a `NetCDFOutputWriter` that writes `(label, output)` pairs in `outputs` (which should
be a `Dict`) to a NetCDF file, where `label` is a string that labels the output and `output` is
either a `Field` (e.g. `model.velocities.u`) or a function `f(model)` that
returns something to be written to disk.

If any of `outputs` are not `AbstractField`, their spatial `dimensions` must be provided.

To use `outputs` on a `grid` not equal to `model.grid`, provide the keyword argument `grid.`

Keyword arguments
=================

- `grid`: The grid associated with `outputs`. Defaults to `model.grid`.

## Filenaming

- `filename` (required): Descriptive filename. `".nc"` is appended to `filename` if `filename` does
                         not end in `".nc"`.

- `dir`: Directory to save output to.

## Output frequency and time-averaging

- `schedule` (required): `AbstractSchedule` that determines when output is saved.

## Slicing and type conversion prior to output

- `indices`: Tuple of indices of the output variables to include. Default is `(:, :, :)`, which
             includes the full fields.

- `with_halos`: Boolean defining whether or not to include halos in the outputs. Default: `false`.
                Note, that to postprocess saved output (e.g., compute derivatives, etc)
                information about the boundary conditions is often crucial. In that case
                you might need to set `with_halos = true`.

- `array_type`: The array type to which output arrays are converted to prior to saving.
                Default: `Array{Float64}`.

- `dimensions`: A `Dict` of dimension tuples to apply to outputs (required for function outputs).

## File management

- `overwrite_existing`: If `false`, `NetCDFOutputWriter` will be set to append to `filepath`. If `true`,
                        `NetCDFOutputWriter` will overwrite `filepath` if it exists or create it if not.
                        Default: `false`. See [NCDatasets.jl documentation](https://alexander-barth.github.io/NCDatasets.jl/stable/)
                        for more information about its `mode` option.

- `deflatelevel`: Determines the NetCDF compression level of data (integer 0-9; 0 (default) means no compression
                  and 9 means maximum compression). See [NCDatasets.jl documentation](https://alexander-barth.github.io/NCDatasets.jl/stable/variables/#Creating-a-variable)
                  for more information.

- `file_splitting`: Schedule for splitting the output file. The new files will be suffixed with
          `_part1`, `_part2`, etc. For example `file_splitting = FileSizeLimit(sz)` will
          split the output file when its size exceeds `sz`. Another example is
          `file_splitting = TimeInterval(30days)`, which will split files every 30 days of
          simulation time. The default incurs no splitting (`NoFileSplitting()`).

## Miscellaneous keywords

- `verbose`: Log what the output writer is doing with statistics on compute/write times and file sizes.
             Default: `false`.

- `part`: The starting part number used when file splitting.

- `global_attributes`: Dict of model properties to save with every file. Default: `Dict()`.

- `output_attributes`: Dict of attributes to be saved with each field variable (reasonable
                       defaults are provided for velocities, buoyancy, temperature, and salinity;
                       otherwise `output_attributes` *must* be user-provided).

Examples
========

Saving the ``u`` velocity field and temperature fields, the full 3D fields and surface 2D slices
to separate NetCDF files:

```@example netcdf1
using Oceananigans

grid = RectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1))

model = NonhydrostaticModel(grid=grid, tracers=:c)

simulation = Simulation(model, Δt=12, stop_time=3600)

fields = Dict("u" => model.velocities.u, "c" => model.tracers.c)

simulation.output_writers[:field_writer] =
    NetCDFOutputWriter(model, fields, filename="fields.nc", schedule=TimeInterval(60))
```

```@example netcdf1
simulation.output_writers[:surface_slice_writer] =
    NetCDFOutputWriter(model, fields, filename="surface_xy_slice.nc",
                       schedule=TimeInterval(60), indices=(:, :, grid.Nz))
```

```@example netcdf1
simulation.output_writers[:averaged_profile_writer] =
    NetCDFOutputWriter(model, fields,
                       filename = "averaged_z_profile.nc",
                       schedule = AveragedTimeInterval(60, window=20),
                       indices = (1, 1, :))
```

`NetCDFOutputWriter` also accepts output functions that write scalars and arrays to disk,
provided that their `dimensions` are provided:

```@example
using Oceananigans

Nx, Ny, Nz = 16, 16, 16

grid = RectilinearGrid(size=(Nx, Ny, Nz), extent=(1, 2, 3))

model = NonhydrostaticModel(; grid)

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
    "slice"   => Dict("long_name" => "Some slice", "units" => "mushrooms")
)

global_attributes = Dict("location" => "Bay of Fundy", "onions" => 7)

simulation.output_writers[:things] =
    NetCDFOutputWriter(model, outputs,
                       schedule=IterationInterval(1), filename="things.nc", dimensions=dims, verbose=true,
                       global_attributes=global_attributes, output_attributes=output_attributes)
```

`NetCDFOutputWriter` can also be configured for `outputs` that are interpolated or regridded
to a different grid than `model.grid`. To use this functionality, include the keyword argument
`grid = output_grid`.

```@example
using Oceananigans
using Oceananigans.Fields: interpolate!

grid = RectilinearGrid(size=(1, 1, 8), extent=(1, 1, 1));
model = NonhydrostaticModel(; grid)

coarse_grid = RectilinearGrid(size=(grid.Nx, grid.Ny, grid.Nz÷2), extent=(grid.Lx, grid.Ly, grid.Lz))
coarse_u = Field{Face, Center, Center}(coarse_grid)

interpolate_u(model) = interpolate!(coarse_u, model.velocities.u)
outputs = (; u = interpolate_u)

output_writer = NetCDFOutputWriter(model, outputs;
                                   grid = coarse_grid,
                                   filename = "coarse_u.nc",
                                   schedule = IterationInterval(1))
```
"""
function NetCDFOutputWriter(model, outputs;
                            filename,
                            schedule,
                            grid = model.grid,
                            dir = ".",
                            array_type = Array{Float64},
                            indices = (:, :, :),
                            with_halos = false,
                            include_grid_metrics = true,
                            overwrite_existing = nothing,
                            verbose = false,
                            global_attributes = Dict(),
                            output_attributes = Dict(),
                            dimensions = Dict(),
                            deflatelevel = 0,
                            part = 1,
                            file_splitting = NoFileSplitting(),
                            dimension_name_generator = default_dim_name)

    if with_halos && indices != (:, :, :)
        throw(ArgumentError("If with_halos=true then you cannot pass indices: $indices"))
    end

    mkpath(dir)
    filename = auto_extension(filename, ".nc")
    filepath = abspath(joinpath(dir, filename))

    initialize!(file_splitting, model)
    update_file_splitting_schedule!(file_splitting, filepath)

    if isnothing(overwrite_existing)
        if isfile(filepath)
            overwrite_existing = false
        else
            overwrite_existing = true
        end
    else
        if isfile(filepath) && !overwrite_existing
            @warn "$filepath already exists and `overwrite_existing = false`. Mode will be set to append to existing file. " *
                  "You might experience errors when writing output if the existing file belonged to a different simulation!"

        elseif isfile(filepath) && overwrite_existing
            @warn "Overwriting existing $filepath."
        end
    end

    outputs = Dict(string(name) => construct_output(outputs[name], grid, indices, with_halos) for name in keys(outputs))

    output_attributes = dictify(output_attributes)
    global_attributes = dictify(global_attributes)
    dimensions = dictify(dimensions)

    # Ensure we can add any kind of metadata to the global attributes later by converting to Dict{Any, Any}.
    global_attributes = Dict{Any, Any}(global_attributes)

    dataset, outputs, schedule = initialize_nc_file!(filepath,
                                                     outputs,
                                                     schedule,
                                                     array_type,
                                                     indices,
                                                     with_halos,
                                                     global_attributes,
                                                     output_attributes,
                                                     dimensions,
                                                     overwrite_existing,
                                                     deflatelevel,
                                                     grid,
                                                     model,
                                                     dimension_name_generator,
                                                     include_grid_metrics)

    return NetCDFOutputWriter(grid,
                              filepath,
                              dataset,
                              outputs,
                              schedule,
                              array_type,
                              indices,
                              with_halos,
                              global_attributes,
                              output_attributes,
                              dimensions,
                              overwrite_existing,
                              deflatelevel,
                              part,
                              file_splitting,
                              verbose,
                              dimension_name_generator,
                              include_grid_metrics)
end

#####
##### NetCDF file initialization
#####

function initialize_nc_file!(filepath,
                             outputs,
                             schedule,
                             array_type,
                             indices,
                             with_halos,
                             global_attributes,
                             output_attributes,
                             dimensions,
                             overwrite_existing,
                             deflatelevel,
                             grid,
                             model,
                             dimension_name_generator,
                             include_grid_metrics)

    mode = overwrite_existing ? "c" : "a"

    # Add useful metadata
    global_attributes["date"] = "This file was generated on $(now()) local time ($(now(UTC)) UTC)."
    global_attributes["Julia"] = "This file was generated using " * versioninfo_with_gpu()
    global_attributes["Oceananigans"] = "This file was generated using " * oceananigans_versioninfo()

    add_schedule_metadata!(global_attributes, schedule)

    # Convert schedule to TimeInterval and each output to WindowedTimeAverage if
    # schedule::AveragedTimeInterval
    schedule, outputs = time_average_outputs(schedule, outputs, model)

    dims = gather_dimensions(outputs, grid, indices, with_halos, dimension_name_generator)

    # Open the NetCDF dataset file
    dataset = NCDataset(filepath, mode, attrib=global_attributes)

    default_dim_attrs = default_dimension_attributes(grid, dimension_name_generator)

    # Define variables for each dimension and attributes if this is a new file.
    if mode == "c"
        # DateTime and TimeDate are both <: AbstractTime
        time_attrib = model.clock.time isa AbstractTime ?
            Dict("long_name" => "Time", "units" => "seconds since 2000-01-01 00:00:00") :
            Dict("long_name" => "Time", "units" => "seconds")

        # Create an unlimited dimension "time"
        # Time should always be Float64 to be extra safe from rounding errors.
        # See: https://github.com/CliMA/Oceananigans.jl/issues/3056
        defDim(dataset, "time", Inf)
        defVar(dataset, "time", Float64, ("time",), attrib=time_attrib)

        # Create spatial dimensions
        for (dim_name, dim_array) in dims
            defVar(dataset, dim_name, array_type(dim_array), (dim_name,),
                   deflatelevel=deflatelevel, attrib=default_dim_attrs[dim_name])
        end

        # Use default output attributes for known outputs if the user has not specified any.
        # Unknown outputs get an empty tuple (no output attributes).
        for c in keys(outputs)
            if !haskey(output_attributes, c)
                output_attributes[c] = c in keys(default_output_attributes) ? default_output_attributes[c] : ()
            end
        end

        time_independent_vars = Dict()

        if include_grid_metrics
            grid_metrics = gather_grid_metrics(grid, indices, dimension_name_generator)
            merge!(time_independent_vars, grid_metrics)
        end

        if grid isa ImmersedBoundaryGrid
            immersed_boundary_vars = gather_immersed_boundary(grid, indices, dimension_name_generator)
            merge!(time_independent_vars, immersed_boundary_vars)
        end

        if !isempty(time_independent_vars)
            for (name, output) in sort(collect(pairs(time_independent_vars)), by=first)
                output = construct_output(output, grid, indices, with_halos)
                attributes = try default_dim_attrs[name]; catch; Dict(); end
                materialized = materialize_output(output, model)
                time_dependent = false

                define_output_variable!(
                    dataset,
                    materialized,
                    name,
                    array_type,
                    deflatelevel,
                    attributes,
                    dimensions,
                    filepath, # for better error messages
                    dimension_name_generator,
                    false # time_dependent = false
                )

                save_output!(dataset, output, model, name, array_type)
            end
        end

        for (name, output) in sort(collect(pairs(outputs)), by=first)
            attributes = try output_attributes[name]; catch; Dict(); end
            materialized = materialize_output(output, model)

            define_output_variable!(
                dataset,
                materialized,
                name,
                array_type,
                deflatelevel,
                attributes,
                dimensions,
                filepath, # for better error messages
                dimension_name_generator,
                true # time_dependent = true
            )
        end

        sync(dataset)
    end

    close(dataset)

    return dataset, outputs, schedule
end

initialize_nc_file!(ow::NetCDFOutputWriter, model) =
    initialize_nc_file!(ow.filepath,
                        ow.outputs,
                        ow.schedule,
                        ow.array_type,
                        ow.indices,
                        ow.with_halos,
                        ow.global_attributes,
                        ow.output_attributes,
                        ow.dimensions,
                        ow.overwrite_existing,
                        ow.deflatelevel,
                        ow.grid,
                        model,
                        ow.dimension_name_generator,
                        ow.include_grid_metrics)

#####
##### Variable definition
#####

materialize_output(func, model) = func(model)
materialize_output(field::AbstractField, model) = field
materialize_output(particles::LagrangianParticles, model) = particles
materialize_output(output::WindowedTimeAverage{<:AbstractField}, model) = output

""" Defines empty variables for 'custom' user-supplied `output`. """
function define_output_variable!(dataset, output, name, array_type,
                                 deflatelevel, attrib, dimensions, filepath,
                                 dimension_name_generator, time_dependent)

    if name ∉ keys(dimensions)
        msg = string("dimensions[$name] for output $name=$(typeof(output)) into $filepath" *
                     " must be provided when constructing NetCDFOutputWriter")
        throw(ArgumentError(msg))
    end

    dims = dimensions[name]
    FT = eltype(array_type)
    defVar(dataset, name, FT, (dims..., "time"); deflatelevel, attrib)

    return nothing
end

""" Defines empty field variable. """
function define_output_variable!(dataset, output::AbstractField, name, array_type,
                                 deflatelevel, attrib, dimensions, filepath,
                                 dimension_name_generator, time_dependent)

    dims = field_dimensions(output, dimension_name_generator)
    FT = eltype(array_type)

    all_dims = time_dependent ? (dims..., "time") : dims

    defVar(dataset, name, FT, all_dims; deflatelevel, attrib)

    return nothing
end

""" Defines empty field variable for `WindowedTimeAverage`s over fields. """
define_output_variable!(dataset, output::WindowedTimeAverage{<:AbstractField}, args...) =
    define_output_variable!(dataset, output.operand, args...)


""" Defines empty variable for particle trackting. """
function define_output_variable!(dataset, output::LagrangianParticles, name, array_type,
                                 deflatelevel, args...)

    particle_fields = eltype(output.properties) |> fieldnames .|> string
    T = eltype(array_type)

    for particle_field in particle_fields
        defVar(dataset, particle_field, T, ("particle_id", "time"); deflatelevel)
    end

    return nothing
end

#####
##### Write output
#####

Base.open(nc::NetCDFOutputWriter) = NCDataset(nc.filepath, "a")
Base.close(nc::NetCDFOutputWriter) = close(nc.dataset)

# Saving outputs with no time dependence (e.g. grid metrics)
function save_output!(ds, output, model, name, array_type)
    fetched = fetch_output(output, model)
    data = convert_output(fetched, array_type)
    data = drop_output_dims(output, data)
    colons = Tuple(Colon() for _ in 1:ndims(data))
    ds[name][colons...] = data
    return nothing
end

# Saving time-dependent outputs
function save_output!(ds, output, model, ow, time_index, name)
    data = fetch_and_convert_output(output, model, ow)
    data = drop_output_dims(output, data)
    colons = Tuple(Colon() for _ in 1:ndims(data))
    ds[name][colons..., time_index:time_index] = data
    return nothing
end

function save_output!(ds, output::LagrangianParticles, model, ow, time_index, name)
    data = fetch_and_convert_output(output, model, ow)
    for (particle_field, vals) in pairs(data)
        ds[string(particle_field)][:, time_index] = vals
    end

    return nothing
end

"""
    write_output!(ow::NetCDFOutputWriter, model)

Write output to netcdf file `output_writer.filepath` at specified intervals. Increments the `time` dimension
every time an output is written to the file.
"""
function write_output!(ow::NetCDFOutputWriter, model)
    # Start a new file if the file_splitting(model) is true
    ow.file_splitting(model) && start_next_file(model, ow)
    update_file_splitting_schedule!(ow.file_splitting, ow.filepath)

    ow.dataset = open(ow)

    ds, verbose, filepath = ow.dataset, ow.verbose, ow.filepath

    time_index = length(ds["time"]) + 1
    ds["time"][time_index] = float_or_date_time(model.clock.time)

    if verbose
        @info "Writing to NetCDF: $filepath..."
        @info "Computing NetCDF outputs for time index $(time_index): $(keys(ow.outputs))..."

        # Time and file size before computing any outputs.
        t0, sz0 = time_ns(), filesize(filepath)
    end

    for (name, output) in ow.outputs
        # Time before computing this output.
        verbose && (t0′ = time_ns())

        save_output!(ds, output, model, ow, time_index, name)

        if verbose
            # Time after computing this output.
            t1′ = time_ns()
            @info "Computing $name done: time=$(prettytime((t1′-t0′) / 1e9))"
        end
    end

    sync(ds)
    close(ow)

    if verbose
        # Time and file size after computing and writing all outputs to disk.
        t1, sz1 = time_ns(), filesize(filepath)
        verbose && @info begin
            @sprintf("Writing done: time=%s, size=%s, Δsize=%s",
                    prettytime((t1-t0)/1e9), pretty_filesize(sz1), pretty_filesize(sz1-sz0))
        end
    end

    return nothing
end

drop_output_dims(output, data) = data # fallback
drop_output_dims(output::WindowedTimeAverage{<:Field}, data) = drop_output_dims(output.operand, data)

function drop_output_dims(field::Field, data)
    reduced_dims = reduced_dimensions(field)
    flat_dims = Tuple(i for (i, T) in enumerate(topology(field.grid)) if T == Flat)
    dims = (reduced_dims..., flat_dims...)
    dims = Tuple(Set(dims)) # ensure dims are unique
    return dropdims(data; dims)
end

#####
##### Show
#####

Base.summary(ow::NetCDFOutputWriter) =
    string("NetCDFOutputWriter writing ", prettykeys(ow.outputs), " to ", ow.filepath, " on ", summary(ow.schedule))

function Base.show(io::IO, ow::NetCDFOutputWriter)
    dims = NCDataset(ow.filepath, "r") do ds
        join([dim * "(" * string(length(ds[dim])) * "), "
              for dim in keys(ds.dim)])[1:end-2]
    end

    averaging_schedule = output_averaging_schedule(ow)
    num_outputs = length(ow.outputs)

    print(io, "NetCDFOutputWriter scheduled on $(summary(ow.schedule)):", "\n",
              "├── filepath: ", relpath(ow.filepath), "\n",
              "├── dimensions: $dims", "\n",
              "├── $num_outputs outputs: ", prettykeys(ow.outputs), show_averaging_schedule(averaging_schedule), "\n",
              "└── array type: ", show_array_type(ow.array_type), "\n",
              "├── file_splitting: ", summary(ow.file_splitting), "\n",
              "└── file size: ", pretty_filesize(filesize(ow.filepath)))
end

#####
##### File splitting
#####

function start_next_file(model, ow::NetCDFOutputWriter)
    verbose = ow.verbose

    verbose && @info begin
        schedule_type = summary(ow.file_splitting)
        "Splitting output because $(schedule_type) is activated."
    end

    if ow.part == 1
        part1_path = replace(ow.filepath, r".nc$" => "_part1.nc")
        verbose && @info "Renaming first part: $(ow.filepath) -> $part1_path"
        mv(ow.filepath, part1_path, force=ow.overwrite_existing)
        ow.filepath = part1_path
    end

    ow.part += 1
    ow.filepath = replace(ow.filepath, r"part\d+.nc$" => "part" * string(ow.part) * ".nc")
    ow.overwrite_existing && isfile(ow.filepath) && rm(ow.filepath, force=true)
    verbose && @info "Now writing to: $(ow.filepath)"

    initialize_nc_file!(ow, model)

    return nothing
end

#####
##### More utils
#####

ext(::Type{NetCDFOutputWriter}) = ".nc"
