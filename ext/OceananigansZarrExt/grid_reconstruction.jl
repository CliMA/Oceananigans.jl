#####
##### Grid Reconstruction for Zarr Files
#####
#
# This file handles serialization (saving) and deserialization (loading) of Oceananigans
# grids to/from Zarr files.
#
# Why is Grid Reconstruction Necessary?
#
# Zarr files are just arrays and metadata - they don't inherently store Julia object
# structure. To properly reconstruct an Oceananigans simulation from a Zarr file, we
# need to rebuild the exact grid that was used, including:
#
# 1. GRID CONSTRUCTION PARAMETERS
#    - Size (Nx, Ny, Nz)
#    - Extent or coordinate arrays
#    - Topology (Periodic, Bounded, Flat)
#    - Halo size
#    - Architecture (CPU/GPU)
#
# 2. IMMERSED BOUNDARIES CONSTRUCTION PARAMETERS
#    For grids with immersed boundaries (e.g., GridFittedBottom for bathymetry):
#    - Bottom height field
#    - Immersed boundary type
#    - Immersed condition parameters
#
# Reconstruction Process:
#
# WRITING (write_grid_reconstruction_data!):
# 1. Extract grid constructor arguments
# 2. Serialize to Zarr-compatible format (strings, numbers, arrays) and save them as in Zarr groups dedicated to that
# 3. Do the same for the immersed boundary construction parameters.
# 4. If immersed boundary, save boundary field (mask or bottom height) as variable in the same Zarr group as above
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
metrics when opening the Zarr file.
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

    add_vertical_metrics!(metrics, grid, indices, dim_name_generator)

    return suffix_grid_keys(metrics, grid_index)
end

function add_vertical_metrics!(metrics, grid, indices, dim_name_generator)
    TZ = topology(grid, 3)
    TZ == Flat && return metrics

    Δprefix = "Δ" * vertical_coordinate_name(grid)
    Δᵃᵃᶠ_name = dim_name_generator(Δprefix, grid, nothing, nothing, f, Val(:z))
    Δᵃᵃᶜ_name = dim_name_generator(Δprefix, grid, nothing, nothing, c, Val(:z))

    metrics[Δᵃᵃᶠ_name] = vertical_spacing_field(grid, f, indices)
    metrics[Δᵃᵃᶜ_name] = vertical_spacing_field(grid, c, indices)

    return metrics
end

function vertical_spacing_field(grid, lz, indices)
    field = Field{Nothing, Nothing, typeof(lz)}(grid; indices)
    Δ_raw = lz isa Center ? grid.z.Δᵃᵃᶜ : grid.z.Δᵃᵃᶠ
    Δ = Δ_raw isa Number ? Δ_raw : on_architecture(CPU(), Δ_raw)
    Nz_int = length(interior_indices(lz, topology(grid, 3)(), size(grid, 3)))
    full = Δ isa Number ? fill(eltype(grid)(Δ), Nz_int) :
                          eltype(grid).(collect(Δ[1:Nz_int]))
    z_slice = indices[3] isa Colon ? (1:Nz_int) : indices[3]
    # Must be a plain 3D `Array` so `set!` hits `set_to_array!` (which handles arch
    # transfer); a `ReshapedArray{Vector}` falls through to broadcast and breaks on GPU.
    interior_arr = collect(reshape(view(full, z_slice), (1, 1, length(z_slice))))
    set!(field, interior_arr)
    return field
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

    add_vertical_metrics!(metrics, grid, indices, dim_name_generator)

    return suffix_grid_keys(metrics, grid_index)
end

# OSSG horizontal metrics: 4 × Δx, 4 × Δy, 4 × Az at the four Arakawa-C stagger
# locations. Vertical Δz/Δr is added via the shared `add_vertical_metrics!`. The 2D horizontal metrics are wrapped in Fields so they
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

    add_vertical_metrics!(metrics, grid, indices, dim_name_generator)

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
since it gives users easy access to relevant immersed boundary parameters when opening the Zarr file.
"""
function gather_immersed_boundary(grid::PCBorGFBIBG, indices, dim_name_generator; grid_index=nothing)
    op_peripheral_nodes_ccc = KernelFunctionOperation{Center, Center, Center}(peripheral_node, grid, Center(), Center(), Center())
    op_peripheral_nodes_fcc = KernelFunctionOperation{Face, Center, Center}(peripheral_node, grid, Face(), Center(), Center())
    op_peripheral_nodes_cfc = KernelFunctionOperation{Center, Face, Center}(peripheral_node, grid, Center(), Face(), Center())
    op_peripheral_nodes_ccf = KernelFunctionOperation{Center, Center, Face}(peripheral_node, grid, Center(), Center(), Face())

    op_inactive_nodes_ccc = KernelFunctionOperation{Center, Center, Center}(inactive_node, grid, Center(), Center(), Center())
    op_inactive_nodes_fcc = KernelFunctionOperation{Face, Center, Center}(inactive_node, grid, Face(), Center(), Center())
    op_inactive_nodes_cfc = KernelFunctionOperation{Center, Face, Center}(inactive_node, grid, Center(), Face(), Center())
    op_inactive_nodes_ccf = KernelFunctionOperation{Center, Center, Face}(inactive_node, grid, Center(), Center(), Face())

    ib_vars = Dict("bottom_height" => Field(grid.immersed_boundary.bottom_height; indices),
                   "peripheral_nodes_ccc" => Field(op_peripheral_nodes_ccc; indices),
                   "peripheral_nodes_fcc" => Field(op_peripheral_nodes_fcc; indices),
                   "peripheral_nodes_cfc" => Field(op_peripheral_nodes_cfc; indices),
                   "peripheral_nodes_ccf" => Field(op_peripheral_nodes_ccf; indices),
                   "inactive_nodes_ccc" => Field(op_inactive_nodes_ccc; indices),
                   "inactive_nodes_fcc" => Field(op_inactive_nodes_fcc; indices),
                   "inactive_nodes_cfc" => Field(op_inactive_nodes_cfc; indices),
                   "inactive_nodes_ccf" => Field(op_inactive_nodes_ccf; indices))

    return suffix_grid_keys(ib_vars, grid_index)
end

const GFBoundaryIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:GridFittedBoundary}

function gather_immersed_boundary(grid::GFBoundaryIBG, indices, dim_name_generator; grid_index=nothing)
    op_peripheral_nodes_ccc = KernelFunctionOperation{Center, Center, Center}(peripheral_node, grid, Center(), Center(), Center())
    op_peripheral_nodes_fcc = KernelFunctionOperation{Face, Center, Center}(peripheral_node, grid, Face(), Center(), Center())
    op_peripheral_nodes_cfc = KernelFunctionOperation{Center, Face, Center}(peripheral_node, grid, Center(), Face(), Center())
    op_peripheral_nodes_ccf = KernelFunctionOperation{Center, Center, Face}(peripheral_node, grid, Center(), Center(), Face())

    op_inactive_nodes_ccc = KernelFunctionOperation{Center, Center, Center}(inactive_node, grid, Center(), Center(), Center())
    op_inactive_nodes_fcc = KernelFunctionOperation{Face, Center, Center}(inactive_node, grid, Face(), Center(), Center())
    op_inactive_nodes_cfc = KernelFunctionOperation{Center, Face, Center}(inactive_node, grid, Center(), Face(), Center())
    op_inactive_nodes_ccf = KernelFunctionOperation{Center, Center, Face}(inactive_node, grid, Center(), Center(), Face())

    ib_vars = Dict("mask" => Field(grid.immersed_boundary.mask; indices),
                   "peripheral_nodes_ccc" => Field(op_peripheral_nodes_ccc; indices),
                   "peripheral_nodes_fcc" => Field(op_peripheral_nodes_fcc; indices),
                   "peripheral_nodes_cfc" => Field(op_peripheral_nodes_cfc; indices),
                   "peripheral_nodes_ccf" => Field(op_peripheral_nodes_ccf; indices),
                   "inactive_nodes_ccc" => Field(op_inactive_nodes_ccc; indices),
                   "inactive_nodes_fcc" => Field(op_inactive_nodes_fcc; indices),
                   "inactive_nodes_cfc" => Field(op_inactive_nodes_cfc; indices),
                   "inactive_nodes_ccf" => Field(op_inactive_nodes_ccf; indices))

    return suffix_grid_keys(ib_vars, grid_index)
end


#####
##### Grid reconstruction
#####

zarr_grid_type_string(g) = string(typeof(g).name.wrapper)

# TODO: from netcdf, need this too?
# zarr_grid_type_string(::TripolarGrid) = "TripolarGrid"
# zarr_grid_type_string(::RotatedLatitudeLongitudeGrid) = "RotatedLatitudeLongitudeGrid"

add_conformal_mapping_info_to_kwargs!(kwargs, grid) = nothing

function add_conformal_mapping_info_to_kwargs!(kwargs, grid::OrthogonalSphericalShellGrid)
    cm_attrs = conformal_mapping_info(grid.conformal_mapping)
    kwargs[:conformal_mapping_attrs] = cm_attrs
    return nothing
end

add_conformal_mapping_info_to_kwargs!(kwargs, grid::ImmersedBoundaryGrid) =
    add_conformal_mapping_info_to_kwargs!(kwargs, grid.underlying_grid)

function zarr_grid_constructor_info(grid)
    args, kwargs = constructor_arguments(grid)
    metadata = Dict(:underlying_grid_type   => zarr_grid_type_string(grid),
                    :immersed_boundary_type => nothing)
    add_conformal_mapping_info_to_kwargs!(kwargs, grid)
    immersed_grid_args = Dict()
    return args, kwargs, immersed_grid_args, metadata
end

function zarr_grid_constructor_info(grid::ImmersedBoundaryGrid)
    underlying_args, underlying_kwargs, immersed_grid_args = constructor_arguments(grid)
    metadata = Dict(:underlying_grid_type   => zarr_grid_type_string(grid.underlying_grid),
                    :immersed_boundary_type => zarr_grid_type_string(grid.immersed_boundary))
    add_conformal_mapping_info_to_kwargs!(underlying_kwargs, grid.underlying_grid)
    # Immersed boundary fields (mask, bottom_height) need data write; deferred to a
    # follow-on phase. For now the grid still serializes the underlying portion.
    return underlying_args, underlying_kwargs, immersed_grid_args, metadata
end

function write_zarr_grid_reconstruction!(root_group, grids)
    single_grid = length(grids) == 1

    grid_groups = map(enumerate(grids)) do (i, grid)
    write_one_grid_reconstruction!(root_group, grid, single_grid ? "grid" : "grid_$i")
    end
    return grid_groups
end

function write_one_grid_reconstruction!(root_group, grid, subgroup_name)
    args, kwargs, immersed_grid_args, metadata = zarr_grid_constructor_info(grid)

    # Positional args: stored as a JSON array of [key, value] pairs so order survives
    # the round-trip through JSON (Zarr.jl parses attrs with `dicttype=Dict{String,Any}`
    # which does not preserve insertion order).
    args_json     = [[string(k), convert_for_zarr(v)] for (k, v) in pairs(args)]
    kwargs_json   = convert_for_zarr(kwargs)
    metadata_json = convert_for_zarr(metadata)

    attrs = Dict{String, Any}(
        "underlying_grid_reconstruction_args"   => args_json,
        "underlying_grid_reconstruction_kwargs" => kwargs_json,
        "grid_reconstruction_metadata"          => metadata_json,
    )
    grid_group = Zarr.zgroup(root_group, subgroup_name; attrs=attrs)
    return grid_group
end

function write_zarr_grid_coords!(group, grid, outputs, grid_suffix, indices, with_halos, dimension_name_generator)
    dims = gather_dimensions(outputs, grid, indices, with_halos, dimension_name_generator; grid_index=grid_suffix)
    
    for (var_name, entry) in dims
        isempty(var_name) && continue

        if entry isa NamedTuple
            arr = entry.array
            var_dims = entry.dims
        else
            arr = entry
            var_dims = (var_name,)
        end

        arr isa Nothing && continue

        arr = collect(arr)

        if !haskey(group, var_name)

            coord = Zarr.zcreate(
                eltype(arr),
                group,
                var_name,
                size(arr)...;
                chunks = size(arr),#TODO: Do we want grid chunking?
                # attrs = merge(
                #     Dict("_ARRAY_DIMENSIONS" => collect(var_dims)),
                #     get(attributes_dict, var_name, Dict{String,Any}())
                # ) TODO: Do we want to write attributes for coordinates? e.g. as netcdf?
            )

            coord .= arr

        else
            existing = collect(group[var_name])

            existing == arr || throw(
                ArgumentError(
                    "Variable '$var_name' already exists but values differ."
                )
            )
        end
    end

    return nothing
end

# function write_zarr_grid_metrics!(group, grid, indices, dimension_name_generator, grid_suffix)
#     metrics = gather_grid_metrics(grid, indices, dimension_name_generator; grid_index=grid_suffix)
    
#     for (name, field) in pairs(metrics)
#         if !haskey(group, name)

#             metric = Zarr.zcreate(
#                 eltype(field),
#                 group,
#                 name,
#                 size(field)...;
#                 chunks = size(field),#TODO: Do we want to chunk metrics?
#                 # attrs = merge(
#                 #     Dict("_ARRAY_DIMENSIONS" => collect(var_dims)),
#                 #     get(attributes_dict, var_name, Dict{String,Any}())
#                 # ) TODO: Do we want to write attributes for metrics? e.g. as netcdf?
#             )

#             metric .= field

#         else
#             existing = collect(group[name])

#             existing == field || throw(
#                 ArgumentError(
#                     "Variable '$name' already exists but values differ."
#                 )
#             )
#         end
#     end
#     return nothing
# end



#£££££££££££££££££££££££££££££££££££££££££££££££££££££££££££ BELOW just write immersed boundary data (mask or bottom_height) and reconstruction

# function write_immersed_boundary_data!(ds, grid::ImmersedBoundaryGrid, immersed_grid_args, prefix)
#     group_name = "$(prefix)immersed_grid_reconstruction_args"
#     if (grid.immersed_boundary isa GridFittedBottom) || (grid.immersed_boundary isa PartialCellBottom)
#         bottom_height = pop!(immersed_grid_args, :bottom_height)
#         ibg_group = defGroup(ds, group_name; attrib=convert_for_netcdf(immersed_grid_args))
#         defVar(ibg_group, "bottom_height", bottom_height)

#     elseif grid.immersed_boundary isa GridFittedBoundary
#         mask = pop!(immersed_grid_args, :mask)
#         ibg_group = defGroup(ds, group_name; attrib=convert_for_netcdf(immersed_grid_args))
#         defVar(ibg_group, "mask", mask)
#     end

#     return ds
# end

# write_immersed_boundary_data!(ds, grid, immersed_grid_args, prefix) = nothing


# function reconstruct_grid(filename::String; grid_index=1, architecture=nothing)
#     ds = NCDataset(filename, "r")
#     grid = reconstruct_grid(ds; grid_index, architecture)
#     close(ds)
#     return grid
# end

# function reconstruct_immersed_boundary(ds, ::Val{:GridFittedBoundary}, prefix)
#     ibg_group = ds.group["$(prefix)immersed_grid_reconstruction_args"]
#     mask = Array(ibg_group["mask"])
#     return GridFittedBoundary(mask)
# end

# function reconstruct_immersed_boundary(ds, ::Val{:GridFittedBottom}, prefix)
#     ibg_group = ds.group["$(prefix)immersed_grid_reconstruction_args"]
#     bottom_height = Array(ibg_group["bottom_height"])
#     immersed_condition = ibg_group.attrib["immersed_condition"] |> materialize_from_netcdf
#     return GridFittedBottom(bottom_height, immersed_condition)
# end

# function reconstruct_immersed_boundary(ds, ::Val{:PartialCellBottom}, prefix)
#     ibg_group = ds.group["$(prefix)immersed_grid_reconstruction_args"]
#     bottom_height = Array(ibg_group["bottom_height"])
#     minimum_fractional_cell_height = ibg_group.attrib["minimum_fractional_cell_height"] |> materialize_from_netcdf
#     return PartialCellBottom(bottom_height, minimum_fractional_cell_height)
# end

# reconstruct_immersed_boundary(ds, immersed_boundary_type, prefix) = error("Unsupported immersed boundary type: $immersed_boundary_type")

# function reconstruct_immersed_boundary(ds, prefix)
#     grid_reconstruction_metadata = ds.group["$(prefix)grid_reconstruction_metadata"].attrib
#     immersed_boundary_type = grid_reconstruction_metadata[:immersed_boundary_type]
#     immersed_boundary = reconstruct_immersed_boundary(ds, Val(Symbol(immersed_boundary_type)), prefix)
#     return immersed_boundary
# end

# function reconstruct_grid(ds; grid_index=1, architecture=nothing)
#     # Try prefixed format (multi-grid) first, fall back to unprefixed format (single-grid / legacy)
#     prefixed_key = "grid_$(grid_index)_underlying_grid_reconstruction_args"
#     prefix = haskey(ds.group, prefixed_key) ? "grid_$(grid_index)_" : ""

#     # Read back the grid reconstruction metadata
#     underlying_grid_reconstruction_args   = ds.group["$(prefix)underlying_grid_reconstruction_args"].attrib |> Dict
#     if !isnothing(architecture) # If architecture is specified, force it into the underlying grid reconstruction arguments before materializing
#         underlying_grid_reconstruction_args["architecture"] = architecture
#     end
#     underlying_grid_reconstruction_args   = underlying_grid_reconstruction_args |> materialize_from_netcdf
#     underlying_grid_reconstruction_kwargs = ds.group["$(prefix)underlying_grid_reconstruction_kwargs"].attrib |> materialize_from_netcdf
#     grid_reconstruction_metadata          = ds.group["$(prefix)grid_reconstruction_metadata"].attrib |> materialize_from_netcdf

#     # Pop out information about the underlying grid
#     underlying_grid_type = grid_reconstruction_metadata[:underlying_grid_type]

#     # OSSG (TripolarGrid, RotatedLatitudeLongitudeGrid, ConformalCubedSpherePanelGrid, …)
#     # is rebuilt directly from the saved λ/φ/Δx/Δy/Az/z arrays — bypassing the user-facing
#     # constructor of the original alias. The reconstructed grid is a generic
#     # `OrthogonalSphericalShellGrid` (with `conformal_mapping = nothing`), which is the
#     # most we can faithfully recover from on-disk state alone.
#     if underlying_grid_type <: OrthogonalSphericalShellGrid
#         underlying_grid = reconstruct_ossg_grid(ds, prefix,
#                                                 underlying_grid_reconstruction_args,
#                                                 underlying_grid_reconstruction_kwargs)
#     else
#         underlying_grid = underlying_grid_type(values(underlying_grid_reconstruction_args)...; underlying_grid_reconstruction_kwargs...)
#     end

#     # If this is an ImmersedBoundaryGrid, reconstruct the immersed boundary, otherwise underlying grid is the final grid
#     if isnothing(grid_reconstruction_metadata[:immersed_boundary_type])
#         grid = underlying_grid
#     else
#         immersed_boundary = reconstruct_immersed_boundary(ds, prefix)
#         immersed_boundary = on_architecture(Architectures.architecture(underlying_grid), immersed_boundary)
#         grid = ImmersedBoundaryGrid(underlying_grid, immersed_boundary)
#     end

#     return grid
# end

# #####
# ##### Metrics-based OrthogonalSphericalShellGrid reconstruction
# #####
# #
# # OSSG variants (TripolarGrid, RotatedLatitudeLongitudeGrid, ConformalCubedSpherePanelGrid)
# # are rebuilt directly from the eight λ/φ aux-coord arrays + twelve Δx/Δy/Az metric
# # arrays + the z vertical-coordinate scaffold — without going through the original
# # constructor. This means:
# #   - Reconstruction is uniform across OSSG aliases (one code path).
# #   - The reconstructed grid is a generic `OrthogonalSphericalShellGrid` with
# #     `conformal_mapping = nothing`. The type-alias identity (e.g. `TripolarGrid`)
# #     is *not* preserved — a faithful copy of the grid arrays is the contract.
# #   - Requires `include_grid_metrics = true` on the writer (the default).
# #
# # Halo regions: if the writer was run with `with_halos = true`, the saved arrays
# # already include halos and are copied in directly. Otherwise the file holds interior
# # values only, and the halo cells of the reconstructed arrays are left as zeros
# # (NaN-filling would be a kinder choice if floating-point hazards matter to consumers).

# """
#     reconstruct_ossg_grid(ds, prefix, args, kwargs)

# Rebuild an `OrthogonalSphericalShellGrid` from the metric arrays stored in `ds`. Used
# internally by `reconstruct_grid` for any underlying grid type that is a subtype of
# `OrthogonalSphericalShellGrid` (TripolarGrid, RotatedLatitudeLongitudeGrid, etc.).
# """
# function reconstruct_ossg_grid(ds, prefix, args, kwargs)
#     arch = args[:architecture]
#     FT   = args[:number_type]

#     # Size/halo come back from the file as `Int32`; normalize to `Int`.
#     Nx, Ny, Nz = map(Int, kwargs[:size])
#     Hx, Hy, Hz = map(Int, kwargs[:halo])
#     topo       = kwargs[:topology]
#     radius     = FT(kwargs[:radius])
#     TX, TY, TZ = topo
#     topo_instances = (TX(), TY(), TZ())

#     file_has_halos = haskey(ds.attrib, "output_includes_halos")

#     # Vertical: detect whether the file used "z" (Static) or "r" (Mutable) for the
#     # reference 1D coordinate. Read the Face nodes and let `generate_coordinate`
#     # rebuild the full halo-padded `StaticVerticalDiscretization`.
#     z_face_var = "$(prefix)z_aaf"
#     r_face_var = "$(prefix)r_aaf"
#     if r_face_var ∈ keys(ds)
#         face_var = r_face_var
#     elseif z_face_var ∈ keys(ds)
#         face_var = z_face_var
#     else
#         throw(ArgumentError("No vertical coordinate variable (z_aaf or r_aaf) found in dataset for OSSG reconstruction."))
#     end
#     z_face_data = collect(ds[face_var])
#     interior_z_faces = file_has_halos ? z_face_data[Hz+1:Hz+Nz+1] : z_face_data
#     Lz, z_disc = generate_coordinate(FT, TZ(), Nz, Hz, collect(interior_z_faces), :z, arch)

#     # Read 2D aux coords + metrics and pad with halos as needed.
#     read_2d(name, lx, ly) = read_ossg_halo_padded_array(ds, "$(prefix)$(name)",
#                                                        FT, arch, lx, ly,
#                                                        topo_instances, (Nx, Ny, Nz), (Hx, Hy, Hz),
#                                                        file_has_halos)

#     λcc = read_2d("λ_cca", Center(), Center())
#     λfc = read_2d("λ_fca", Face(),   Center())
#     λcf = read_2d("λ_cfa", Center(), Face())
#     λff = read_2d("λ_ffa", Face(),   Face())

#     φcc = read_2d("φ_cca", Center(), Center())
#     φfc = read_2d("φ_fca", Face(),   Center())
#     φcf = read_2d("φ_cfa", Center(), Face())
#     φff = read_2d("φ_ffa", Face(),   Face())

#     # Metrics may not be present if the writer ran with `include_grid_metrics=false`.
#     have_metrics = "$(prefix)Δx_cca" ∈ keys(ds)
#     if !have_metrics
#         throw(ArgumentError("OrthogonalSphericalShellGrid reconstruction requires grid metrics " *
#                             "(Δx_**, Δy_**, Az_**). Re-run the writer with `include_grid_metrics=true`."))
#     end

#     Δxcc = read_2d("Δx_cca", Center(), Center())
#     Δxfc = read_2d("Δx_fca", Face(),   Center())
#     Δxcf = read_2d("Δx_cfa", Center(), Face())
#     Δxff = read_2d("Δx_ffa", Face(),   Face())

#     Δycc = read_2d("Δy_cca", Center(), Center())
#     Δyfc = read_2d("Δy_fca", Face(),   Center())
#     Δycf = read_2d("Δy_cfa", Center(), Face())
#     Δyff = read_2d("Δy_ffa", Face(),   Face())

#     Azcc = read_2d("Az_cca", Center(), Center())
#     Azfc = read_2d("Az_fca", Face(),   Center())
#     Azcf = read_2d("Az_cfa", Center(), Face())
#     Azff = read_2d("Az_ffa", Face(),   Face())

#     # Reconstruct the conformal_mapping (if saved) so the resulting grid keeps its
#     # type-alias identity (TripolarGrid / RotatedLatitudeLongitudeGrid). This is what
#     # downstream code (boundary-condition defaults, kernel dispatch) keys on.
#     cm_group_key = "$(prefix)conformal_mapping"
#     conformal_mapping = haskey(ds.group, cm_group_key) ?
#         reconstruct_conformal_mapping(ds.group[cm_group_key].attrib, TY) : nothing

#     # Preliminary grid with unfilled metric halos. We use it as the "helper grid" for
#     # halo filling: building a Field on it picks up the correct BCs from the topology
#     # (e.g. the north fold for TripolarGrid), so `fill_halo_regions!` does the right
#     # thing across the fold.
#     preliminary = OrthogonalSphericalShellGrid{FT, TX, TY, TZ}(arch,
#                                                                 Nx, Ny, Nz, Hx, Hy, Hz,
#                                                                 FT(Lz),
#                                                                 λcc, λfc, λcf, λff,
#                                                                 φcc, φfc, φcf, φff,
#                                                                 z_disc,
#                                                                 Δxcc, Δxfc, Δxcf, Δxff,
#                                                                 Δycc, Δyfc, Δycf, Δyff,
#                                                                 Azcc, Azfc, Azcf, Azff,
#                                                                 radius,
#                                                                 conformal_mapping)

#     fill_metric_halos(arr, lx, ly) = halo_fill_2d_metric(arr, preliminary, lx, ly)

#     λcc = fill_metric_halos(λcc, Center, Center)
#     λfc = fill_metric_halos(λfc, Face,   Center)
#     λcf = fill_metric_halos(λcf, Center, Face)
#     λff = fill_metric_halos(λff, Face,   Face)

#     φcc = fill_metric_halos(φcc, Center, Center)
#     φfc = fill_metric_halos(φfc, Face,   Center)
#     φcf = fill_metric_halos(φcf, Center, Face)
#     φff = fill_metric_halos(φff, Face,   Face)

#     Δxcc = fill_metric_halos(Δxcc, Center, Center)
#     Δxfc = fill_metric_halos(Δxfc, Face,   Center)
#     Δxcf = fill_metric_halos(Δxcf, Center, Face)
#     Δxff = fill_metric_halos(Δxff, Face,   Face)

#     Δycc = fill_metric_halos(Δycc, Center, Center)
#     Δyfc = fill_metric_halos(Δyfc, Face,   Center)
#     Δycf = fill_metric_halos(Δycf, Center, Face)
#     Δyff = fill_metric_halos(Δyff, Face,   Face)

#     Azcc = fill_metric_halos(Azcc, Center, Center)
#     Azfc = fill_metric_halos(Azfc, Face,   Center)
#     Azcf = fill_metric_halos(Azcf, Center, Face)
#     Azff = fill_metric_halos(Azff, Face,   Face)

#     return OrthogonalSphericalShellGrid{FT, TX, TY, TZ}(arch,
#                                                          Nx, Ny, Nz, Hx, Hy, Hz,
#                                                          FT(Lz),
#                                                          λcc, λfc, λcf, λff,
#                                                          φcc, φfc, φcf, φff,
#                                                          z_disc,
#                                                          Δxcc, Δxfc, Δxcf, Δxff,
#                                                          Δycc, Δyfc, Δycf, Δyff,
#                                                          Azcc, Azfc, Azcf, Azff,
#                                                          radius,
#                                                          conformal_mapping)
# end

# # Wrap a bare 2D metric/coord `OffsetMatrix` as a `Field` on `grid`, fill its halos
# # using the grid's default BCs (e.g. the north fold on TripolarGrid), and return a
# # halo-filled `OffsetMatrix` matching the input layout. Necessary because derivatives
# # across the tripolar fold otherwise NaN out if the metric halos are zero.
# function halo_fill_2d_metric(old_data, grid, LX, LY)
#     TX, TY, _ = topology(grid)
#     Nx, Ny, _ = size(grid)
#     new_field = Field{LX, LY, Center}(grid)
#     Ni = Base.length(LX(), TX(), Nx)
#     Nj = Base.length(LY(), TY(), Ny)
#     cpu_old_data = on_architecture(CPU(), old_data)
#     # Broadcast the 2D metric into every z-level so we can slice any of them back out
#     # below; we need `LZ = Center` (not `Nothing`) so the TripolarGrid fold halo-fill
#     # — which dispatches on `Center` z-location — adds the 360° longitude wrap.
#     for k in axes(new_field.data, 3)
#         new_field.data[1:Ni, 1:Nj, k] .= cpu_old_data[1:Ni, 1:Nj]
#     end
#     fill_halo_regions!(new_field)
#     # The UPivot fold's "redundancy substitution" rewrites the j=Ny interior row for
#     # Center- and Face-Center y-locations (see fill_halo_regions_upivotzipper.jl) to
#     # mirror the right half of that row from the left half. That's correct for
#     # symmetric metrics (Δx/Δy/Az/φ) but wrong for λ, whose halves differ by 360°.
#     # Re-stamp the interior from the saved data so λ is preserved; for symmetric
#     # metrics this is a no-op.
#     for k in axes(new_field.data, 3)
#         new_field.data[1:Ni, 1:Nj, k] .= cpu_old_data[1:Ni, 1:Nj]
#     end
#     # Take z=1 instead of `dropdims`, which would require z to be singleton (it isn't —
#     # the field has Nz+2Hz cells in z). Any k works because we set every level identically.
#     return on_architecture(architecture(grid), deepcopy(view(new_field.data, :, :, 1)))
# end

# # Rebuild the `conformal_mapping` from its serialized attributes (see
# # `conformal_mapping_info` in `src/OrthogonalSphericalShellGrids/`). The `TY`
# # argument is the y-topology type — for `Tripolar`, that's the fold flavor
# # (`RightCenterFolded`/`RightFaceFolded`) which lives as a type-parameter on
# # the struct rather than a runtime field.
# reconstruct_conformal_mapping(attrib, TY) = reconstruct_conformal_mapping(attrib, Val(Symbol(attrib["type"])), TY)

# reconstruct_conformal_mapping(attrib, ::Val{:Nothing}, TY) = nothing

# function reconstruct_conformal_mapping(attrib, ::Val{:Tripolar}, TY)
#     return Tripolar(
#         attrib["north_poles_latitude"],
#         attrib["first_pole_longitude"],
#         attrib["southernmost_latitude"],
#         TY,
#     )
# end

# function reconstruct_conformal_mapping(attrib, ::Val{:LatitudeLongitudeRotation}, TY)
#     return LatitudeLongitudeRotation((attrib["north_pole_λ"], attrib["north_pole_φ"]))
# end

# # Unknown conformal-mapping types (e.g., `CubedSphereConformalMapping`) — leave as
# # `nothing`; the grid will still be usable as a generic OSSG.
# reconstruct_conformal_mapping(attrib, ::Val, TY) = nothing

# # Read a 2D OSSG metric/coord variable from the file and pad it out to the halo-included
# # shape that the OSSG constructor expects. Returns an OffsetMatrix indexed `[1-Hx:Nx+Hx, …]`.
# function read_ossg_halo_padded_array(ds, name, FT, arch, lx, ly, topo_instances, sz, halo_sz, file_has_halos)
#     Nx, Ny, Nz = sz
#     TX, TY, _  = topo_instances

#     # Allocate a halo-padded 2D array via `Oceananigans.Grids.new_data` (imported as
#     # `allocate_grid_data` to avoid colliding with `OutputReaders`' separate `new_data`),
#     # then drop the singleton vertical dim so we get an `OffsetMatrix` the OSSG
#     # constructor accepts.
#     full3d = allocate_grid_data(FT, arch, (lx, ly, nothing), topo_instances, sz, halo_sz)
#     full2d = OffsetArray(dropdims(parent(full3d), dims=3), full3d.offsets[1:2]...)

#     raw = collect(ds[name])

#     if file_has_halos
#         # The saved array already includes halos. Sanity-check the size and assign through.
#         expected_full = size(parent(full2d))
#         size(raw) == expected_full || throw(ArgumentError(
#             "Saved array '$name' has size $(size(raw)) but expected halo-included size $expected_full."))
#         parent(full2d) .= FT.(raw)
#     else
#         # The saved array is the interior only. Copy it into the interior of the halo array.
#         i_range = interior_indices(lx, TX, Nx)
#         j_range = interior_indices(ly, TY, Ny)
#         expected_interior = (length(i_range), length(j_range))
#         size(raw) == expected_interior || throw(ArgumentError(
#             "Saved array '$name' has size $(size(raw)) but expected interior size $expected_interior."))
#         full2d[i_range, j_range] .= FT.(raw)
#     end

#     return full2d
# end
