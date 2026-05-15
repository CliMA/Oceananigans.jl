#####
##### Grid Reconstruction for NetCDF Files
#####
#
# This file handles serialization (saving) and deserialization (loading) of Oceananigans
# grids to/from NetCDF files.
#
# Why is Grid Reconstruction Necessary?
#
# NetCDF files are just arrays and metadata - they don't inherently store Julia object
# structure. To properly reconstruct an Oceananigans simulation from a NetCDF file, we
# need to rebuild the exact grid that was used, including:
#
# 1. GRID CONSTRUCTION PARAMETERS
#    - Size (Nx, Ny, Nz)
#    - Extent or coordinate arrays
#    - Topology (Periodic, Bounded, Flat)
#    - Halo size
#    - Architecture (CPU/GPU)
#
#    Why? These parameters determine how Oceananigans computes derivatives, interpolation,
#    and boundary conditions. Getting them wrong would give incorrect physics.
#
# 2. IMMERSED BOUNDARIES CONSTRUCTION PARAMETERS
#    For grids with immersed boundaries (e.g., GridFittedBottom for bathymetry):
#    - Bottom height field
#    - Immersed boundary type
#    - Immersed condition parameters
#
#    Why? Immersed boundaries modify which cells are active/inactive. This affects
#    every computation in the domain. Without this information, we can't properly
#    reconstruct flow near boundaries.
#
#
# Reconstruction Process:
#
# WRITING (write_grid_reconstruction_data!):
# 1. Extract grid constructor arguments
# 2. Serialize to NetCDF-compatible format (strings, numbers, arrays) and save them as in NetCDF groups dedicated to that
# 3. Do the same for the immersed boundary construction parameters.
# 4. If immersed boundary, save boundary field (mask or bottom height) as variable in the same NetCDF group as above
#
# READING (reconstruct_grid):
# 1. Read "underlying_grid_type" attribute to determine grid type
# 2. Read "underlying_grid_reconstruction_args" and "underlying_grid_reconstruction_kwargs" and deserialize
# 3. Call appropriate grid constructor with saved arguments
# 4. For immersed boundaries, reconstruct boundary object
# 5. Return the grid
#####

#####
##### Gathering of grid metrics
#####

import Oceananigans.Architectures

"""
    gather_grid_metrics(grid, indices, dim_name_generator)

Gather the grid metrics for the grid. Not strictly necessary for grid reconstruction, but it is
implemented and used as a quality of life improvement since it gives users easy access to relevant grid
metrics when opening the NetCDF file.
"""
function gather_grid_metrics(grid::RectilinearGrid, indices, dim_name_generator; grid_index=nothing)
    TX, TY, TZ = topology(grid)

    metrics = Dict()

    if TX != Flat
        Δxᶠᵃᵃ_name = dim_name_generator("Δx", grid, f, nothing, nothing, Val(:x))
        Δxᶜᵃᵃ_name = dim_name_generator("Δx", grid, c, nothing, nothing, Val(:x))

        Δxᶠᵃᵃ_field = Field(xspacings(grid, f); indices)
        Δxᶜᵃᵃ_field = Field(xspacings(grid, c); indices)

        metrics[Δxᶠᵃᵃ_name] = Δxᶠᵃᵃ_field
        metrics[Δxᶜᵃᵃ_name] = Δxᶜᵃᵃ_field
    end

    if TY != Flat
        Δyᵃᶠᵃ_name = dim_name_generator("Δy", grid, nothing, f, nothing, Val(:y))
        Δyᵃᶜᵃ_name = dim_name_generator("Δy", grid, nothing, c, nothing, Val(:y))

        Δyᵃᶠᵃ_field = Field(yspacings(grid, f); indices)
        Δyᵃᶜᵃ_field = Field(yspacings(grid, c); indices)

        metrics[Δyᵃᶠᵃ_name] = Δyᵃᶠᵃ_field
        metrics[Δyᵃᶜᵃ_name] = Δyᵃᶜᵃ_field
    end

    if TZ != Flat
        Δz = "Δ" * vertical_coordinate_name(grid)
        Δzᵃᵃᶠ_name = dim_name_generator(Δz, grid, nothing, nothing, f, Val(:z))
        Δzᵃᵃᶜ_name = dim_name_generator(Δz, grid, nothing, nothing, c, Val(:z))

        Δzᵃᵃᶠ_field = Field(zspacings(grid, f); indices)
        Δzᵃᵃᶜ_field = Field(zspacings(grid, c); indices)

        metrics[Δzᵃᵃᶠ_name] = Δzᵃᵃᶠ_field
        metrics[Δzᵃᵃᶜ_name] = Δzᵃᵃᶜ_field
    end

    return suffix_grid_keys(metrics, grid_index)
end

function gather_grid_metrics(grid::LatitudeLongitudeGrid, indices, dim_name_generator; grid_index=nothing)
    TΛ, TΦ, TZ = topology(grid)

    metrics = Dict()

    if TΛ != Flat
        Δλᶠᵃᵃ_name = dim_name_generator("Δλ", grid, f, nothing, nothing, Val(:x))
        Δλᶜᵃᵃ_name = dim_name_generator("Δλ", grid, c, nothing, nothing, Val(:x))

        Δλᶠᵃᵃ_field = Field(λspacings(grid, f); indices)
        Δλᶜᵃᵃ_field = Field(λspacings(grid, c); indices)

        metrics[Δλᶠᵃᵃ_name] = Δλᶠᵃᵃ_field
        metrics[Δλᶜᵃᵃ_name] = Δλᶜᵃᵃ_field

        Δxᶠᶠᵃ_name = dim_name_generator("Δx", grid, f, f, nothing, Val(:x))
        Δxᶠᶜᵃ_name = dim_name_generator("Δx", grid, f, c, nothing, Val(:x))
        Δxᶜᶠᵃ_name = dim_name_generator("Δx", grid, c, f, nothing, Val(:x))
        Δxᶜᶜᵃ_name = dim_name_generator("Δx", grid, c, c, nothing, Val(:x))

        Δxᶠᶠᵃ_field = Field(xspacings(grid, f, f); indices)
        Δxᶠᶜᵃ_field = Field(xspacings(grid, f, c); indices)
        Δxᶜᶠᵃ_field = Field(xspacings(grid, c, f); indices)
        Δxᶜᶜᵃ_field = Field(xspacings(grid, c, c); indices)

        metrics[Δxᶠᶠᵃ_name] = Δxᶠᶠᵃ_field
        metrics[Δxᶠᶜᵃ_name] = Δxᶠᶜᵃ_field
        metrics[Δxᶜᶠᵃ_name] = Δxᶜᶠᵃ_field
        metrics[Δxᶜᶜᵃ_name] = Δxᶜᶜᵃ_field
    end

    if TΦ != Flat
        Δφᵃᶠᵃ_name = dim_name_generator("Δλ", grid, nothing, f, nothing, Val(:y))
        Δφᵃᶜᵃ_name = dim_name_generator("Δλ", grid, nothing, c, nothing, Val(:y))

        Δφᵃᶠᵃ_field = Field(φspacings(grid, f); indices)
        Δφᵃᶜᵃ_field = Field(φspacings(grid, c); indices)

        metrics[Δφᵃᶠᵃ_name] = Δφᵃᶠᵃ_field
        metrics[Δφᵃᶜᵃ_name] = Δφᵃᶜᵃ_field

        Δyᶠᶠᵃ_name = dim_name_generator("Δy", grid, f, f, nothing, Val(:y))
        Δyᶠᶜᵃ_name = dim_name_generator("Δy", grid, f, c, nothing, Val(:y))
        Δyᶜᶠᵃ_name = dim_name_generator("Δy", grid, c, f, nothing, Val(:y))
        Δyᶜᶜᵃ_name = dim_name_generator("Δy", grid, c, c, nothing, Val(:y))

        Δyᶠᶠᵃ_field = Field(yspacings(grid, f, f); indices)
        Δyᶠᶜᵃ_field = Field(yspacings(grid, f, c); indices)
        Δyᶜᶠᵃ_field = Field(yspacings(grid, c, f); indices)
        Δyᶜᶜᵃ_field = Field(yspacings(grid, c, c); indices)

        metrics[Δyᶠᶠᵃ_name] = Δyᶠᶠᵃ_field
        metrics[Δyᶠᶜᵃ_name] = Δyᶠᶜᵃ_field
        metrics[Δyᶜᶠᵃ_name] = Δyᶜᶠᵃ_field
        metrics[Δyᶜᶜᵃ_name] = Δyᶜᶜᵃ_field
    end

    if TZ != Flat
        Δz = "Δ" * vertical_coordinate_name(grid)
        Δzᵃᵃᶠ_name = dim_name_generator(Δz, grid, nothing, nothing, f, Val(:z))
        Δzᵃᵃᶜ_name = dim_name_generator(Δz, grid, nothing, nothing, c, Val(:z))

        Δzᵃᵃᶠ_field = Field(zspacings(grid, f); indices)
        Δzᵃᵃᶜ_field = Field(zspacings(grid, c); indices)

        metrics[Δzᵃᵃᶠ_name] = Δzᵃᵃᶠ_field
        metrics[Δzᵃᵃᶜ_name] = Δzᵃᵃᶜ_field
    end

    return suffix_grid_keys(metrics, grid_index)
end

# OSSG metrics: 8 × Δx, 8 × Δy, 4 × Az at the four Arakawa-C stagger locations,
# plus vertical Δz/Δr. The 2D horizontal metrics are wrapped in Fields so they
# go through the standard output path and pick up the (i_*, j_*) bare dim names
# from `field_dimensions(::AbstractField, ::OSSG, …)` and the `coordinates`
# attribute from `add_aux_coordinates_attribute!`.
function gather_grid_metrics(grid::OrthogonalSphericalShellGrid, indices, dim_name_generator; grid_index=nothing)
    metrics = Dict()

    for (lx, ly) in ((c, c), (f, c), (c, f), (f, f))
        Δx_name = dim_name_generator("Δx", grid, lx, ly, nothing, Val(:x))
        Δy_name = dim_name_generator("Δy", grid, lx, ly, nothing, Val(:y))
        Az_name = dim_name_generator("Az", grid, lx, ly, nothing, Val(:x))

        metrics[Δx_name] = Field(xspacings(grid, lx, ly); indices)
        metrics[Δy_name] = Field(yspacings(grid, lx, ly); indices)
        # Az is on the same horizontal stagger as Δx/Δy at (lx, ly). The `Az_at_node`
        # accessor returns `grid.Azᶜᶜᵃ[i, j]` (etc.) at the requested location; wrapping it
        # in a `KernelFunctionOperation` gives a 2D `Field` we can write through the normal
        # output path.
        Az_op = KernelFunctionOperation{typeof(lx), typeof(ly), Nothing}(Az_at_node, grid, lx, ly)
        metrics[Az_name] = Field(Az_op; indices)
    end

    TZ = topology(grid, 3)
    # Skip vertical Δz/Δr metrics on `ConformalCubedSpherePanelGrid`: the resulting
    # `Field(zspacings(grid, c))` triggers a Julia 1.12 GC segfault during writer
    # init for grids whose `CubedSphereConformalMapping{Rotation, Fξ, Fη, Cξ, Cη}`
    # carries nested `StepRangeLen{Float64, TwicePrecision{Float64}, …}` type
    # parameters. Other OSSG variants (TripolarGrid, RotatedLatitudeLongitudeGrid)
    # have shallower conformal-mapping types and emit Δz metrics normally; the 1D
    # reference `z_*`/`r_*` coordinate variables are always written.
    skip_vertical_metrics = grid isa ConformalCubedSpherePanelGrid
    if TZ != Flat && !skip_vertical_metrics
        Δz = "Δ" * vertical_coordinate_name(grid)
        Δzᵃᵃᶠ_name = dim_name_generator(Δz, grid, nothing, nothing, f, Val(:z))
        Δzᵃᵃᶜ_name = dim_name_generator(Δz, grid, nothing, nothing, c, Val(:z))

        metrics[Δzᵃᵃᶠ_name] = Field(zspacings(grid, f); indices)
        metrics[Δzᵃᵃᶜ_name] = Field(zspacings(grid, c); indices)
    end

    return suffix_grid_keys(metrics, grid_index)
end

# Az is unstaggered in z; access the appropriate 2D area array at the given horizontal stagger.
@inline Az_at_node(i, j, k, grid, ::Center, ::Center) = @inbounds grid.Azᶜᶜᵃ[i, j]
@inline Az_at_node(i, j, k, grid, ::Face,   ::Center) = @inbounds grid.Azᶠᶜᵃ[i, j]
@inline Az_at_node(i, j, k, grid, ::Center, ::Face)   = @inbounds grid.Azᶜᶠᵃ[i, j]
@inline Az_at_node(i, j, k, grid, ::Face,   ::Face)   = @inbounds grid.Azᶠᶠᵃ[i, j]

#####
##### Gathering of immersed boundary fields
#####

gather_grid_metrics(grid::ImmersedBoundaryGrid, args...; kw...) = gather_grid_metrics(grid.underlying_grid, args...; kw...)

# TODO: Proper masks for 2D models?
flat_loc(T, L) = T == Flat ? nothing : L

const PCBorGFBIBG = Union{GFBIBG, PCBIBG}

"""
    gather_immersed_boundary(grid, indices, dim_name_generator)

Gather the construction parameters for the immersed boundary of the grid. This isn't
strictly necessary for grid reconstruction, but it is implemented and used a quality of life improvement
since it gives users easy access to relevant immersed boundary parameters when opening the NetCDF file.
"""
function gather_immersed_boundary(grid::PCBorGFBIBG, indices, dim_name_generator; grid_index=nothing)
    op_peripheral_nodes_ccc = KernelFunctionOperation{Center, Center, Center}(peripheral_node, grid, Center(), Center(), Center())
    op_peripheral_nodes_fcc = KernelFunctionOperation{Face, Center, Center}(peripheral_node, grid, Face(), Center(), Center())
    op_peripheral_nodes_cfc = KernelFunctionOperation{Center, Face, Center}(peripheral_node, grid, Center(), Face(), Center())
    op_peripheral_nodes_ccf = KernelFunctionOperation{Center, Center, Face}(peripheral_node, grid, Center(), Center(), Face())

    ib_vars = Dict("bottom_height" => Field(grid.immersed_boundary.bottom_height; indices),
                   "peripheral_nodes_ccc" => Field(op_peripheral_nodes_ccc; indices),
                   "peripheral_nodes_fcc" => Field(op_peripheral_nodes_fcc; indices),
                   "peripheral_nodes_cfc" => Field(op_peripheral_nodes_cfc; indices),
                   "peripheral_nodes_ccf" => Field(op_peripheral_nodes_ccf; indices))

    return suffix_grid_keys(ib_vars, grid_index)
end

const GFBoundaryIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:GridFittedBoundary}

function gather_immersed_boundary(grid::GFBoundaryIBG, indices, dim_name_generator; grid_index=nothing)
    op_peripheral_nodes_ccc = KernelFunctionOperation{Center, Center, Center}(peripheral_node, grid, Center(), Center(), Center())
    op_peripheral_nodes_fcc = KernelFunctionOperation{Face, Center, Center}(peripheral_node, grid, Face(), Center(), Center())
    op_peripheral_nodes_cfc = KernelFunctionOperation{Center, Face, Center}(peripheral_node, grid, Center(), Face(), Center())
    op_peripheral_nodes_ccf = KernelFunctionOperation{Center, Center, Face}(peripheral_node, grid, Center(), Center(), Face())

    ib_vars = Dict("mask" => Field(grid.immersed_boundary.mask; indices),
                   "peripheral_nodes_ccc" => Field(op_peripheral_nodes_ccc; indices),
                   "peripheral_nodes_fcc" => Field(op_peripheral_nodes_fcc; indices),
                   "peripheral_nodes_cfc" => Field(op_peripheral_nodes_cfc; indices),
                   "peripheral_nodes_ccf" => Field(op_peripheral_nodes_ccf; indices))

    return suffix_grid_keys(ib_vars, grid_index)
end


#####
##### Grid reconstruction
#####

netcdf_string(obj) = typeof(obj).name.wrapper |> string
# OSSG variants are type aliases of OrthogonalSphericalShellGrid; record the alias name
# so reconstruction dispatches on the right constructor (which accepts a different
# kwarg set than the base OSSG constructor).
netcdf_string(::TripolarGrid) = "TripolarGrid"
netcdf_string(::RotatedLatitudeLongitudeGrid) = "RotatedLatitudeLongitudeGrid"

function netcdf_grid_constructor_info(grid)
    underlying_grid_args, underlying_grid_kwargs = constructor_arguments(grid)

    immersed_grid_args = Dict()

    underlying_grid_type = netcdf_string(grid) # Save type of grid for reconstruction
    grid_metadata = Dict(:immersed_boundary_type => nothing,
                         :underlying_grid_type => underlying_grid_type)
    return underlying_grid_args, underlying_grid_kwargs, immersed_grid_args, grid_metadata
end

function netcdf_grid_constructor_info(grid::ImmersedBoundaryGrid)
    underlying_grid_args, underlying_grid_kwargs, immersed_grid_args = constructor_arguments(grid)

    immersed_boundary_type = netcdf_string(grid.immersed_boundary) # Save type of immersed boundary for reconstruction
    underlying_grid_type   = netcdf_string(grid.underlying_grid) # Save type of underlying grid for reconstruction

    grid_metadata = Dict(:immersed_boundary_type => immersed_boundary_type,
                         :underlying_grid_type => underlying_grid_type)
    return underlying_grid_args, underlying_grid_kwargs, immersed_grid_args, grid_metadata
end

function write_immersed_boundary_data!(ds, grid::ImmersedBoundaryGrid, immersed_grid_args, prefix)
    group_name = "$(prefix)immersed_grid_reconstruction_args"
    if (grid.immersed_boundary isa GridFittedBottom) || (grid.immersed_boundary isa PartialCellBottom)
        bottom_height = pop!(immersed_grid_args, :bottom_height)
        ibg_group = defGroup(ds, group_name; attrib=convert_for_netcdf(immersed_grid_args))
        defVar(ibg_group, "bottom_height", bottom_height)

    elseif grid.immersed_boundary isa GridFittedBoundary
        mask = pop!(immersed_grid_args, :mask)
        ibg_group = defGroup(ds, group_name; attrib=convert_for_netcdf(immersed_grid_args))
        defVar(ibg_group, "mask", mask)
    end

    return ds
end

write_immersed_boundary_data!(ds, grid, immersed_grid_args, prefix) = nothing

# When grid_index is nothing (single grid), use unprefixed group names
# for backward compatibility with the legacy format.
# When grid_index is an integer (multi-grid), prefix groups with "grid_N_".
function write_grid_reconstruction_data!(ds, grid, grid_index; array_type=Array{eltype(grid)}, deflatelevel=0)
    underlying_grid_args, underlying_grid_kwargs, immersed_grid_args, grid_metadata = netcdf_grid_constructor_info(grid)
    underlying_grid_args, underlying_grid_kwargs, grid_metadata = map(convert_for_netcdf, (underlying_grid_args, underlying_grid_kwargs, grid_metadata))

    prefix = isnothing(grid_index) ? "" : "grid_$(grid_index)_"
    defGroup(ds, "$(prefix)underlying_grid_reconstruction_args"; attrib = underlying_grid_args)
    defGroup(ds, "$(prefix)underlying_grid_reconstruction_kwargs"; attrib = underlying_grid_kwargs)
    defGroup(ds, "$(prefix)grid_reconstruction_metadata"; attrib = grid_metadata)

    # For OSSG variants, also record the `conformal_mapping` so the type alias
    # (TripolarGrid / RotatedLatitudeLongitudeGrid / …) can be reconstructed.
    write_ossg_conformal_mapping!(ds, grid, prefix)

    write_immersed_boundary_data!(ds, grid, immersed_grid_args, prefix)

    return ds
end

# Default: no conformal_mapping to record (Rectilinear, LLG, …).
write_ossg_conformal_mapping!(ds, grid, prefix) = nothing

function write_ossg_conformal_mapping!(ds, grid::OrthogonalSphericalShellGrid, prefix)
    cm_attrs = conformal_mapping_info(grid.conformal_mapping)
    defGroup(ds, "$(prefix)conformal_mapping"; attrib = convert_for_netcdf(cm_attrs))
    return nothing
end

# Defer through ImmersedBoundaryGrid.
write_ossg_conformal_mapping!(ds, grid::ImmersedBoundaryGrid, prefix) =
    write_ossg_conformal_mapping!(ds, grid.underlying_grid, prefix)

function reconstruct_grid(filename::String; grid_index=1, architecture=nothing)
    ds = NCDataset(filename, "r")
    grid = reconstruct_grid(ds; grid_index, architecture)
    close(ds)
    return grid
end

function reconstruct_immersed_boundary(ds, ::Val{:GridFittedBoundary}, prefix)
    ibg_group = ds.group["$(prefix)immersed_grid_reconstruction_args"]
    mask = Array(ibg_group["mask"])
    return GridFittedBoundary(mask)
end

function reconstruct_immersed_boundary(ds, ::Val{:GridFittedBottom}, prefix)
    ibg_group = ds.group["$(prefix)immersed_grid_reconstruction_args"]
    bottom_height = Array(ibg_group["bottom_height"])
    immersed_condition = ibg_group.attrib["immersed_condition"] |> materialize_from_netcdf
    return GridFittedBottom(bottom_height, immersed_condition)
end

function reconstruct_immersed_boundary(ds, ::Val{:PartialCellBottom}, prefix)
    ibg_group = ds.group["$(prefix)immersed_grid_reconstruction_args"]
    bottom_height = Array(ibg_group["bottom_height"])
    minimum_fractional_cell_height = ibg_group.attrib["minimum_fractional_cell_height"] |> materialize_from_netcdf
    return PartialCellBottom(bottom_height, minimum_fractional_cell_height)
end

reconstruct_immersed_boundary(ds, immersed_boundary_type, prefix) = error("Unsupported immersed boundary type: $immersed_boundary_type")

function reconstruct_immersed_boundary(ds, prefix)
    grid_reconstruction_metadata = ds.group["$(prefix)grid_reconstruction_metadata"].attrib
    immersed_boundary_type = grid_reconstruction_metadata[:immersed_boundary_type]
    immersed_boundary = reconstruct_immersed_boundary(ds, Val(Symbol(immersed_boundary_type)), prefix)
    return immersed_boundary
end

function reconstruct_grid(ds; grid_index=1, architecture=nothing)
    # Try prefixed format (multi-grid) first, fall back to unprefixed format (single-grid / legacy)
    prefixed_key = "grid_$(grid_index)_underlying_grid_reconstruction_args"
    prefix = haskey(ds.group, prefixed_key) ? "grid_$(grid_index)_" : ""

    # Read back the grid reconstruction metadata
    underlying_grid_reconstruction_args   = ds.group["$(prefix)underlying_grid_reconstruction_args"].attrib |> Dict
    if !isnothing(architecture) # If architecture is specified, force it into the underlying grid reconstruction arguments before materializing
        underlying_grid_reconstruction_args["architecture"] = architecture
    end
    underlying_grid_reconstruction_args   = underlying_grid_reconstruction_args |> materialize_from_netcdf
    underlying_grid_reconstruction_kwargs = ds.group["$(prefix)underlying_grid_reconstruction_kwargs"].attrib |> materialize_from_netcdf
    grid_reconstruction_metadata          = ds.group["$(prefix)grid_reconstruction_metadata"].attrib |> materialize_from_netcdf

    # Pop out information about the underlying grid
    underlying_grid_type = grid_reconstruction_metadata[:underlying_grid_type]

    # OSSG (TripolarGrid, RotatedLatitudeLongitudeGrid, ConformalCubedSpherePanelGrid, …)
    # is rebuilt directly from the saved λ/φ/Δx/Δy/Az/z arrays — bypassing the user-facing
    # constructor of the original alias. The reconstructed grid is a generic
    # `OrthogonalSphericalShellGrid` (with `conformal_mapping = nothing`), which is the
    # most we can faithfully recover from on-disk state alone.
    if underlying_grid_type <: OrthogonalSphericalShellGrid
        underlying_grid = reconstruct_ossg_grid(ds, prefix,
                                                underlying_grid_reconstruction_args,
                                                underlying_grid_reconstruction_kwargs)
    else
        underlying_grid = underlying_grid_type(values(underlying_grid_reconstruction_args)...; underlying_grid_reconstruction_kwargs...)
    end

    # If this is an ImmersedBoundaryGrid, reconstruct the immersed boundary, otherwise underlying grid is the final grid
    if isnothing(grid_reconstruction_metadata[:immersed_boundary_type])
        grid = underlying_grid
    else
        immersed_boundary = reconstruct_immersed_boundary(ds, prefix)
        immersed_boundary = on_architecture(Architectures.architecture(underlying_grid), immersed_boundary)
        grid = ImmersedBoundaryGrid(underlying_grid, immersed_boundary)
    end

    return grid
end

#####
##### Metrics-based OrthogonalSphericalShellGrid reconstruction
#####
#
# OSSG variants (TripolarGrid, RotatedLatitudeLongitudeGrid, ConformalCubedSpherePanelGrid)
# are rebuilt directly from the eight λ/φ aux-coord arrays + twelve Δx/Δy/Az metric
# arrays + the z vertical-coordinate scaffold — without going through the original
# constructor. This means:
#   - Reconstruction is uniform across OSSG aliases (one code path).
#   - The reconstructed grid is a generic `OrthogonalSphericalShellGrid` with
#     `conformal_mapping = nothing`. The type-alias identity (e.g. `TripolarGrid`)
#     is *not* preserved — a faithful copy of the grid arrays is the contract.
#   - Requires `include_grid_metrics = true` on the writer (the default).
#
# Halo regions: if the writer was run with `with_halos = true`, the saved arrays
# already include halos and are copied in directly. Otherwise the file holds interior
# values only, and the halo cells of the reconstructed arrays are left as zeros
# (NaN-filling would be a kinder choice if floating-point hazards matter to consumers).

"""
    reconstruct_ossg_grid(ds, prefix, args, kwargs)

Rebuild an `OrthogonalSphericalShellGrid` from the metric arrays stored in `ds`. Used
internally by `reconstruct_grid` for any underlying grid type that is a subtype of
`OrthogonalSphericalShellGrid` (TripolarGrid, RotatedLatitudeLongitudeGrid, etc.).
"""
function reconstruct_ossg_grid(ds, prefix, args, kwargs)
    arch = args[:architecture]
    FT   = args[:number_type]

    # Size/halo come back from the file as `Int32`; normalize to `Int`.
    Nx, Ny, Nz = map(Int, kwargs[:size])
    Hx, Hy, Hz = map(Int, kwargs[:halo])
    topo       = kwargs[:topology]
    radius     = FT(kwargs[:radius])
    TX, TY, TZ = topo
    topo_instances = (TX(), TY(), TZ())

    file_has_halos = haskey(ds.attrib, "output_includes_halos")

    # Vertical: detect whether the file used "z" (Static) or "r" (Mutable) for the
    # reference 1D coordinate. Read the Face nodes and let `generate_coordinate`
    # rebuild the full halo-padded `StaticVerticalDiscretization`.
    z_face_var = "$(prefix)z_aaf"
    r_face_var = "$(prefix)r_aaf"
    if r_face_var ∈ keys(ds)
        face_var = r_face_var
    elseif z_face_var ∈ keys(ds)
        face_var = z_face_var
    else
        throw(ArgumentError("No vertical coordinate variable (z_aaf or r_aaf) found in dataset for OSSG reconstruction."))
    end
    z_face_data = collect(ds[face_var])
    interior_z_faces = file_has_halos ? z_face_data[Hz+1:Hz+Nz+1] : z_face_data
    Lz, z_disc = generate_coordinate(FT, TZ(), Nz, Hz, collect(interior_z_faces), :z, arch)

    # Read 2D aux coords + metrics and pad with halos as needed.
    read_2d(name, lx, ly) = read_ossg_halo_padded_array(ds, "$(prefix)$(name)",
                                                       FT, arch, lx, ly,
                                                       topo_instances, (Nx, Ny, Nz), (Hx, Hy, Hz),
                                                       file_has_halos)

    λcc = read_2d("λ_cca", Center(), Center())
    λfc = read_2d("λ_fca", Face(),   Center())
    λcf = read_2d("λ_cfa", Center(), Face())
    λff = read_2d("λ_ffa", Face(),   Face())

    φcc = read_2d("φ_cca", Center(), Center())
    φfc = read_2d("φ_fca", Face(),   Center())
    φcf = read_2d("φ_cfa", Center(), Face())
    φff = read_2d("φ_ffa", Face(),   Face())

    # Metrics may not be present if the writer ran with `include_grid_metrics=false`.
    have_metrics = "$(prefix)Δx_cca" ∈ keys(ds)
    if !have_metrics
        throw(ArgumentError("OrthogonalSphericalShellGrid reconstruction requires grid metrics " *
                            "(Δx_**, Δy_**, Az_**). Re-run the writer with `include_grid_metrics=true`."))
    end

    Δxcc = read_2d("Δx_cca", Center(), Center())
    Δxfc = read_2d("Δx_fca", Face(),   Center())
    Δxcf = read_2d("Δx_cfa", Center(), Face())
    Δxff = read_2d("Δx_ffa", Face(),   Face())

    Δycc = read_2d("Δy_cca", Center(), Center())
    Δyfc = read_2d("Δy_fca", Face(),   Center())
    Δycf = read_2d("Δy_cfa", Center(), Face())
    Δyff = read_2d("Δy_ffa", Face(),   Face())

    Azcc = read_2d("Az_cca", Center(), Center())
    Azfc = read_2d("Az_fca", Face(),   Center())
    Azcf = read_2d("Az_cfa", Center(), Face())
    Azff = read_2d("Az_ffa", Face(),   Face())

    # Reconstruct the conformal_mapping (if saved) so the resulting grid keeps its
    # type-alias identity (TripolarGrid / RotatedLatitudeLongitudeGrid). This is what
    # downstream code (boundary-condition defaults, kernel dispatch) keys on.
    cm_group_key = "$(prefix)conformal_mapping"
    conformal_mapping = haskey(ds.group, cm_group_key) ?
        reconstruct_conformal_mapping(ds.group[cm_group_key].attrib, TY) : nothing

    return OrthogonalSphericalShellGrid{FT, TX, TY, TZ}(arch,
                                                         Nx, Ny, Nz, Hx, Hy, Hz,
                                                         FT(Lz),
                                                         λcc, λfc, λcf, λff,
                                                         φcc, φfc, φcf, φff,
                                                         z_disc,
                                                         Δxcc, Δxfc, Δxcf, Δxff,
                                                         Δycc, Δyfc, Δycf, Δyff,
                                                         Azcc, Azfc, Azcf, Azff,
                                                         radius,
                                                         conformal_mapping)
end

# Rebuild the `conformal_mapping` from its serialized attributes (see
# `conformal_mapping_info` in `src/OrthogonalSphericalShellGrids/`). The `TY`
# argument is the y-topology type — for `Tripolar`, that's the fold flavor
# (`RightCenterFolded`/`RightFaceFolded`) which lives as a type-parameter on
# the struct rather than a runtime field.
reconstruct_conformal_mapping(attrib, TY) = reconstruct_conformal_mapping(attrib, Val(Symbol(attrib["type"])), TY)

reconstruct_conformal_mapping(attrib, ::Val{:Nothing}, TY) = nothing

function reconstruct_conformal_mapping(attrib, ::Val{:Tripolar}, TY)
    return Tripolar(
        attrib["north_poles_latitude"],
        attrib["first_pole_longitude"],
        attrib["southernmost_latitude"],
        TY,
    )
end

function reconstruct_conformal_mapping(attrib, ::Val{:LatitudeLongitudeRotation}, TY)
    return LatitudeLongitudeRotation((attrib["north_pole_λ"], attrib["north_pole_φ"]))
end

# Unknown conformal-mapping types (e.g., `CubedSphereConformalMapping`) — leave as
# `nothing`; the grid will still be usable as a generic OSSG.
reconstruct_conformal_mapping(attrib, ::Val, TY) = nothing

# Read a 2D OSSG metric/coord variable from the file and pad it out to the halo-included
# shape that the OSSG constructor expects. Returns an OffsetMatrix indexed `[1-Hx:Nx+Hx, …]`.
function read_ossg_halo_padded_array(ds, name, FT, arch, lx, ly, topo_instances, sz, halo_sz, file_has_halos)
    Nx, Ny, Nz = sz
    TX, TY, _  = topo_instances

    # Allocate a halo-padded 2D array via `Oceananigans.Grids.new_data` (imported as
    # `allocate_grid_data` to avoid colliding with `OutputReaders`' separate `new_data`),
    # then drop the singleton vertical dim so we get an `OffsetMatrix` the OSSG
    # constructor accepts.
    full3d = allocate_grid_data(FT, arch, (lx, ly, nothing), topo_instances, sz, halo_sz)
    full2d = OffsetArray(dropdims(parent(full3d), dims=3), full3d.offsets[1:2]...)

    raw = collect(ds[name])

    if file_has_halos
        # The saved array already includes halos. Sanity-check the size and assign through.
        expected_full = size(parent(full2d))
        size(raw) == expected_full || throw(ArgumentError(
            "Saved array '$name' has size $(size(raw)) but expected halo-included size $expected_full."))
        parent(full2d) .= FT.(raw)
    else
        # The saved array is the interior only. Copy it into the interior of the halo array.
        i_range = interior_indices(lx, TX, Nx)
        j_range = interior_indices(ly, TY, Ny)
        expected_interior = (length(i_range), length(j_range))
        size(raw) == expected_interior || throw(ArgumentError(
            "Saved array '$name' has size $(size(raw)) but expected interior size $expected_interior."))
        full2d[i_range, j_range] .= FT.(raw)
    end

    return full2d
end
