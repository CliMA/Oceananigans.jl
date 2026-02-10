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

"""
    gather_grid_metrics(grid, indices, dim_name_generator)

Gather the grid metrics for the grid. Not strictly necessary for grid reconstruction, but it is
implemented and used as a quality of life improvement since it gives users easy access to relevant grid
metrics when opening the NetCDF file.
"""
function gather_grid_metrics(grid::RectilinearGrid, indices, dim_name_generator)
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
        Δzᵃᵃᶠ_name = dim_name_generator("Δz", grid, nothing, nothing, f, Val(:z))
        Δzᵃᵃᶜ_name = dim_name_generator("Δz", grid, nothing, nothing, c, Val(:z))

        Δzᵃᵃᶠ_field = Field(zspacings(grid, f); indices)
        Δzᵃᵃᶜ_field = Field(zspacings(grid, c); indices)

        metrics[Δzᵃᵃᶠ_name] = Δzᵃᵃᶠ_field
        metrics[Δzᵃᵃᶜ_name] = Δzᵃᵃᶜ_field
    end

    return metrics
end

function gather_grid_metrics(grid::LatitudeLongitudeGrid, indices, dim_name_generator)
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
        Δzᵃᵃᶠ_name = dim_name_generator("Δz", grid, nothing, nothing, f, Val(:z))
        Δzᵃᵃᶜ_name = dim_name_generator("Δz", grid, nothing, nothing, c, Val(:z))

        Δzᵃᵃᶠ_field = Field(zspacings(grid, f); indices)
        Δzᵃᵃᶜ_field = Field(zspacings(grid, c); indices)

        metrics[Δzᵃᵃᶠ_name] = Δzᵃᵃᶠ_field
        metrics[Δzᵃᵃᶜ_name] = Δzᵃᵃᶜ_field
    end

    return metrics
end

#####
##### Gathering of immersed boundary fields
#####

gather_grid_metrics(grid::ImmersedBoundaryGrid, args...) = gather_grid_metrics(grid.underlying_grid, args...)

# TODO: Proper masks for 2D models?
flat_loc(T, L) = T == Flat ? nothing : L

const PCBorGFBIBG = Union{GFBIBG, PCBIBG}

"""
    gather_immersed_boundary(grid, indices, dim_name_generator)

Gather the construction parameters for the immersed boundary of the grid. This isn't
strictly necessary for grid reconstruction, but it is implemented and used a quality of life improvement
since it gives users easy access to relevant immersed boundary parameters when opening the NetCDF file.
"""
function gather_immersed_boundary(grid::PCBorGFBIBG, indices, dim_name_generator)
    op_peripheral_nodes_ccc = KernelFunctionOperation{Center, Center, Center}(peripheral_node, grid, Center(), Center(), Center())
    op_peripheral_nodes_fcc = KernelFunctionOperation{Face, Center, Center}(peripheral_node, grid, Face(), Center(), Center())
    op_peripheral_nodes_cfc = KernelFunctionOperation{Center, Face, Center}(peripheral_node, grid, Center(), Face(), Center())
    op_peripheral_nodes_ccf = KernelFunctionOperation{Center, Center, Face}(peripheral_node, grid, Center(), Center(), Face())

    return Dict("bottom_height" => Field(grid.immersed_boundary.bottom_height; indices),
                "peripheral_nodes_ccc" => Field(op_peripheral_nodes_ccc; indices),
                "peripheral_nodes_fcc" => Field(op_peripheral_nodes_fcc; indices),
                "peripheral_nodes_cfc" => Field(op_peripheral_nodes_cfc; indices),
                "peripheral_nodes_ccf" => Field(op_peripheral_nodes_ccf; indices))
end

const GFBoundaryIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:GridFittedBoundary}

function gather_immersed_boundary(grid::GFBoundaryIBG, indices, dim_name_generator)
    op_peripheral_nodes_ccc = KernelFunctionOperation{Center, Center, Center}(peripheral_node, grid, Center(), Center(), Center())
    op_peripheral_nodes_fcc = KernelFunctionOperation{Face, Center, Center}(peripheral_node, grid, Face(), Center(), Center())
    op_peripheral_nodes_cfc = KernelFunctionOperation{Center, Face, Center}(peripheral_node, grid, Center(), Face(), Center())
    op_peripheral_nodes_ccf = KernelFunctionOperation{Center, Center, Face}(peripheral_node, grid, Center(), Center(), Face())

    return Dict("mask" => Field(grid.immersed_boundary.mask; indices),
                "peripheral_nodes_ccc" => Field(op_peripheral_nodes_ccc; indices),
                "peripheral_nodes_fcc" => Field(op_peripheral_nodes_fcc; indices),
                "peripheral_nodes_cfc" => Field(op_peripheral_nodes_cfc; indices),
                "peripheral_nodes_ccf" => Field(op_peripheral_nodes_ccf; indices))
end


#####
##### Grid reconstruction
#####

netcdf_string(obj) = typeof(obj).name.wrapper |> string

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

function write_immersed_boundary_data!(ds, grid::ImmersedBoundaryGrid, immersed_grid_args)
    group_name = "immersed_grid_reconstruction_args"
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

write_immersed_boundary_data!(ds, grid, immersed_grid_args) = nothing

function write_grid_reconstruction_data!(ds, grid; array_type=Array{eltype(grid)}, deflatelevel=0)
    underlying_grid_args, underlying_grid_kwargs, immersed_grid_args, grid_metadata = netcdf_grid_constructor_info(grid)
    underlying_grid_args, underlying_grid_kwargs, grid_metadata = map(convert_for_netcdf, (underlying_grid_args, underlying_grid_kwargs, grid_metadata))

    defGroup(ds, "underlying_grid_reconstruction_args"; attrib = underlying_grid_args)
    defGroup(ds, "underlying_grid_reconstruction_kwargs"; attrib = underlying_grid_kwargs)
    defGroup(ds, "grid_reconstruction_metadata"; attrib = grid_metadata)

    write_immersed_boundary_data!(ds, grid, immersed_grid_args)

    return ds
end

function reconstruct_grid(filename::String)
    ds = NCDataset(filename, "r")
    grid = reconstruct_grid(ds)
    close(ds)
    return grid
end

function reconstruct_immersed_boundary(ds, ::Val{:GridFittedBoundary})
    ibg_group = ds.group["immersed_grid_reconstruction_args"]
    mask = Array(ibg_group["mask"])
    return GridFittedBoundary(mask)
end

function reconstruct_immersed_boundary(ds, ::Val{:GridFittedBottom})
    ibg_group = ds.group["immersed_grid_reconstruction_args"]
    bottom_height = Array(ibg_group["bottom_height"])
    immersed_condition = ibg_group.attrib["immersed_condition"] |> materialize_from_netcdf
    return GridFittedBottom(bottom_height, immersed_condition)
end

function reconstruct_immersed_boundary(ds, ::Val{:PartialCellBottom})
    ibg_group = ds.group["immersed_grid_reconstruction_args"]
    bottom_height = Array(ibg_group["bottom_height"])
    minimum_fractional_cell_height = ibg_group.attrib["minimum_fractional_cell_height"] |> materialize_from_netcdf
    return PartialCellBottom(bottom_height, minimum_fractional_cell_height)
end

reconstruct_immersed_boundary(ds, immersed_boundary_type) = error("Unsupported immersed boundary type: $immersed_boundary_type")

function reconstruct_immersed_boundary(ds)
    grid_reconstruction_metadata = ds.group["grid_reconstruction_metadata"].attrib
    immersed_boundary_type = grid_reconstruction_metadata[:immersed_boundary_type]
    immersed_boundary = reconstruct_immersed_boundary(ds, Val(Symbol(immersed_boundary_type)))
    return immersed_boundary
end

function reconstruct_grid(ds; arch=nothing)
    # Read back the grid reconstruction metadata
    underlying_grid_reconstruction_args   = ds.group["underlying_grid_reconstruction_args"].attrib |> Dict
    if !isnothing(arch) # If architecture is specified, force it into the underlying grid reconstruction arguments before materializing
        underlying_grid_reconstruction_args["architecture"] = arch
    end
    underlying_grid_reconstruction_args   = underlying_grid_reconstruction_args |> materialize_from_netcdf
    underlying_grid_reconstruction_kwargs = ds.group["underlying_grid_reconstruction_kwargs"].attrib |> materialize_from_netcdf
    grid_reconstruction_metadata          = ds.group["grid_reconstruction_metadata"].attrib |> materialize_from_netcdf

    # Pop out infomration about the underlying grid
    underlying_grid_type = grid_reconstruction_metadata[:underlying_grid_type]
    underlying_grid = underlying_grid_type(values(underlying_grid_reconstruction_args)...; underlying_grid_reconstruction_kwargs...)

    # If this is an ImmersedBoundaryGrid, reconstruct the immersed boundary, otherwise underlying grid is the final grid
    if isnothing(grid_reconstruction_metadata[:immersed_boundary_type])
        grid = underlying_grid
    else
        immersed_boundary = reconstruct_immersed_boundary(ds)
        immersed_boundary = on_architecture(architecture(underlying_grid), immersed_boundary)
        grid = ImmersedBoundaryGrid(underlying_grid, immersed_boundary)
    end

    return grid
end
