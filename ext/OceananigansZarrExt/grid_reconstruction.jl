#####
##### Zarr grid serialization and reconstruction
#####

#####
##### Grid reconstruction
#####

zarr_grid_type_string(g) = string(typeof(g).name.wrapper)

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
    args, kwargs, _, metadata = zarr_grid_constructor_info(grid)

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

function write_zarr_array!(group, name, data, dimensions, attributes=Dict())
    data = collect(data)
    attributes = zarr_attribute_dict(attributes)
    attributes["_ARRAY_DIMENSIONS"] = reverse(collect(string.(dimensions)))

    if !haskey(group, name)
        variable = Zarr.zcreate(eltype(data), group, name, size(data)...;
                                chunks=size(data), attrs=attributes)
        variable .= data
    else
        existing = collect(group[name])
        existing == data || throw(ArgumentError("Variable '$name' already exists but values differ."))
    end

    return nothing
end

zarr_field_data(field, data = interior(field)) =
    squeeze_reduced_dimensions(field, data; array_type = Array{eltype(field)})

function write_zarr_grid_coords!(group, grid, outputs, grid_suffix, indices,
                                 with_halos, dimension_name_generator,
                                 attributes, dimension_type)
    dims = gather_dimensions(outputs, grid, indices, with_halos, dimension_name_generator; grid_index=grid_suffix)

    for (var_name, entry) in dims
        isempty(var_name) && continue

        if entry isa NamedTuple
            arr = entry.array
            var_dims = entry.dims
        else
            arr = entry
            var_dims = tuple(var_name)
        end

        arr isa Nothing && continue

        arr = collect(dimension_type.(arr))
        variable_attributes = get(attributes, var_name, Dict())
        write_zarr_array!(group, var_name, arr, var_dims, variable_attributes)
    end

    return nothing
end

function write_zarr_grid_metrics!(group, grid, indices, with_halos, dimension_name_generator,
                                  grid_suffix, attributes)
    metrics = gather_grid_metrics(grid, indices, dimension_name_generator; grid_index=grid_suffix)

    for (name, field) in pairs(metrics)
        dimensions = filter(!isempty, field_dimensions(field, grid, dimension_name_generator;
                                                       grid_index=grid_suffix))
        data = with_halos ? zarr_field_data(field, parent(field)) : zarr_field_data(field)
        variable_attributes = zarr_attribute_dict(get(attributes, name, Dict()))
        coordinates = field_auxiliary_coordinates(field, grid, dimension_name_generator;
                                                  grid_index=grid_suffix)
        isempty(coordinates) || (variable_attributes["coordinates"] = join(coordinates, " "))
        write_zarr_array!(group, name, data, dimensions, variable_attributes)
    end
    return nothing
end

function write_zarr_grid_immersed_boundary!(group, grid::ImmersedBoundaryGrid, indices, dimension_name_generator, grid_suffix)

    _, _, ibg_args, _ = zarr_grid_constructor_info(grid)
    ib_vars = gather_immersed_boundary(grid, indices, dimension_name_generator)
    pop!(ibg_args, :mask, nothing)
    pop!(ibg_args, :bottom_height, nothing)

    ibg_group = Zarr.zgroup(
        group,
        "immersed_boundary";
        attrs = zarr_safe_dict(convert_for_zarr(ibg_args))
    )

    for (name, field) in pairs(ib_vars)
        dimensions = filter(!isempty, field_dimensions(field, grid, dimension_name_generator;
                                                       grid_index=grid_suffix))
        data = zarr_field_data(field)
        write_zarr_array!(ibg_group, name, data, dimensions)
    end
    return nothing
end

write_zarr_grid_immersed_boundary!(group, grid, indices, dimension_name_generator, grid_suffix) = nothing

#####
##### Grid reconstruction
#####

function reconstruct_immersed_boundary(grid_group, ::Val{:GridFittedBoundary})
    ibg_group = grid_group["immersed_boundary"]
    mask = Array(ibg_group["mask"])
    return GridFittedBoundary(mask)
end

function reconstruct_immersed_boundary(grid_group, ::Val{:GridFittedBottom})
    ibg_group = grid_group["immersed_boundary"]
    bottom_height = Array(ibg_group["bottom_height"])
    immersed_condition = ibg_group.attrs["immersed_condition"] |> materialize_from_zarr
    return GridFittedBottom(bottom_height, immersed_condition)
end

function reconstruct_immersed_boundary(grid_group, ::Val{:PartialCellBottom})
    ibg_group = grid_group["immersed_boundary"]
    bottom_height = Array(ibg_group["bottom_height"])
    minimum_fractional_cell_height = ibg_group.attrs["minimum_fractional_cell_height"] |> materialize_from_zarr
    return PartialCellBottom(bottom_height, minimum_fractional_cell_height)
end

reconstruct_immersed_boundary(grid_group, immersed_boundary_type, prefix) = error("Unsupported immersed boundary type: $immersed_boundary_type")

function reconstruct_immersed_boundary(grid_group)
    grid_reconstruction_metadata = grid_group.attrs["grid_reconstruction_metadata"]
    immersed_boundary_type = grid_reconstruction_metadata["immersed_boundary_type"]
    immersed_boundary = reconstruct_immersed_boundary(grid_group, Val(Symbol(immersed_boundary_type)))
    return immersed_boundary
end

"""
    reconstruct_zarr_grid(group; grid_index=1, architecture=nothing)

Read a grid back from a Zarr group written by `ZarrWriter`. Looks for `grid/` (single)
then `grid_<index>/` (multi).
"""
function reconstruct_zarr_grid(group; grid_index=1, architecture=nothing)
    subgroup_name = "grid" in keys(group.groups) ? "grid" : "grid_$grid_index"
    haskey(group.groups, subgroup_name) ||
        throw(ArgumentError("No grid reconstruction data found in this Zarr group (looked for `$subgroup_name`)."))

    grid_group = group.groups[subgroup_name]
    attrs = grid_group.attrs

    # Positional args: list of [key, value] pairs (preserves order across JSON).
    args_pairs  = attrs["underlying_grid_reconstruction_args"]
    args_ordered = [(Symbol(p[1]), materialize_from_zarr(p[2])) for p in args_pairs]
    if !isnothing(architecture)
        # Override architecture entry with the user-supplied one.
        args_ordered = [(k === :architecture ? :architecture => architecture : k => v)
                        for (k, v) in args_ordered]
    end
    args_values = [v for (_, v) in args_ordered]
    args_dict   = Dict(args_ordered)
    kwargs_dict = materialize_from_zarr(attrs["underlying_grid_reconstruction_kwargs"])
    metadata    = materialize_from_zarr(attrs["grid_reconstruction_metadata"])

    underlying_grid_type = metadata[:underlying_grid_type]

    if underlying_grid_type <: OrthogonalSphericalShellGrid
        grid_suffix = subgroup_name == "grid" ? nothing : grid_index
        underlying_grid = reconstruct_zarr_ossg_grid(group, args_dict, kwargs_dict, grid_suffix)
    else
        underlying_grid = underlying_grid_type(args_values...; kwargs_dict...)
    end

    if isnothing(metadata[:immersed_boundary_type])
        grid = underlying_grid
    else
        haskey(grid_group.groups, "immersed_boundary") ||
            throw(ArgumentError("No grid immersed boundary reconstruction data found in this Zarr group"))
        immersed_boundary = reconstruct_immersed_boundary(grid_group)
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
function reconstruct_zarr_ossg_grid(root_group, args, kwargs, grid_suffix)
    arch = args[:architecture]
    FT   = args[:number_type]

    # Size/halo come back from the file as `Int32`; normalize to `Int`.
    Nx, Ny, Nz = map(Int, kwargs[:size])
    Hx, Hy, Hz = map(Int, kwargs[:halo])
    topo       = kwargs[:topology]
    radius     = FT(kwargs[:radius])
    TX, TY, TZ = topo
    topo_instances = (TX(), TY(), TZ())

    file_has_halos = haskey(root_group.attrs, "output_includes_halos")

    # Vertical: detect whether the file used "z" (Static) or "r" (Mutable) for the
    # reference 1D coordinate. Read the Face nodes and let `generate_coordinate`
    # rebuild the full halo-padded vertical discretization.
    z_face_var = add_grid_suffix("z_aaf", grid_suffix)
    r_face_var = add_grid_suffix("r_aaf", grid_suffix)
    if haskey(root_group, r_face_var)
        face_var = r_face_var
    elseif haskey(root_group, z_face_var)
        face_var = z_face_var
    else
        throw(ArgumentError("No vertical coordinate variable (z_aaf or r_aaf) found in dataset for OSSG reconstruction."))
    end
    z_face_data = collect(root_group[face_var])
    interior_z_faces = file_has_halos ? z_face_data[Hz+1:Hz+Nz+1] : z_face_data
    Lz, z_disc = generate_coordinate(FT, TZ(), Nz, Hz, collect(interior_z_faces), :z, arch)

    # Read 2D aux coords + metrics and pad with halos as needed.
    read_2d(name, lx, ly) = read_ossg_halo_padded_array(root_group,
                                                        add_grid_suffix(name, grid_suffix),
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
    if !haskey(root_group, add_grid_suffix("Δx_cca", grid_suffix))
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
    cm_group_key = :conformal_mapping_attrs

    conformal_mapping = haskey(kwargs, cm_group_key) ?
        reconstruct_conformal_mapping(kwargs[cm_group_key], TY) : nothing

    # Preliminary grid with unfilled metric halos. We use it as the "helper grid" for
    # halo filling: building a Field on it picks up the correct BCs from the topology
    # (e.g. the north fold for TripolarGrid), so `fill_halo_regions!` does the right
    # thing across the fold.
    preliminary = OrthogonalSphericalShellGrid{FT, TX, TY, TZ}(arch,
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

    fill_metric_halos(arr, lx, ly) = halo_fill_2d_metric(arr, preliminary, lx, ly)

    λcc = fill_metric_halos(λcc, Center, Center)
    λfc = fill_metric_halos(λfc, Face,   Center)
    λcf = fill_metric_halos(λcf, Center, Face)
    λff = fill_metric_halos(λff, Face,   Face)

    φcc = fill_metric_halos(φcc, Center, Center)
    φfc = fill_metric_halos(φfc, Face,   Center)
    φcf = fill_metric_halos(φcf, Center, Face)
    φff = fill_metric_halos(φff, Face,   Face)

    Δxcc = fill_metric_halos(Δxcc, Center, Center)
    Δxfc = fill_metric_halos(Δxfc, Face,   Center)
    Δxcf = fill_metric_halos(Δxcf, Center, Face)
    Δxff = fill_metric_halos(Δxff, Face,   Face)

    Δycc = fill_metric_halos(Δycc, Center, Center)
    Δyfc = fill_metric_halos(Δyfc, Face,   Center)
    Δycf = fill_metric_halos(Δycf, Center, Face)
    Δyff = fill_metric_halos(Δyff, Face,   Face)

    Azcc = fill_metric_halos(Azcc, Center, Center)
    Azfc = fill_metric_halos(Azfc, Face,   Center)
    Azcf = fill_metric_halos(Azcf, Center, Face)
    Azff = fill_metric_halos(Azff, Face,   Face)

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
reconstruct_conformal_mapping(attrib, TY) = reconstruct_conformal_mapping(attrib, Val(nameof(attrib[:type])), TY)

reconstruct_conformal_mapping(attrib, ::Val{:Nothing}, TY) = nothing

function reconstruct_conformal_mapping(attrib, ::Val{:Tripolar}, TY)
    return Tripolar(
        attrib[:north_poles_latitude],
        attrib[:first_pole_longitude],
        attrib[:southernmost_latitude],
        TY,
    )
end

function reconstruct_conformal_mapping(attrib, ::Val{:LatitudeLongitudeRotation}, TY)
    return LatitudeLongitudeRotation((attrib[:north_pole_λ], attrib[:north_pole_φ]))
end

# Unknown conformal-mapping types (e.g., `CubedSphereConformalMapping`) — leave as
# `nothing`; the grid will still be usable as a generic OSSG.
reconstruct_conformal_mapping(attrib, ::Val, TY) = nothing

# Read a 2D OSSG metric/coord variable from the file and pad it out to the halo-included
# shape that the OSSG constructor expects. Returns an OffsetMatrix indexed `[1-Hx:Nx+Hx, …]`.
function read_ossg_halo_padded_array(grid_group, name, FT, arch, lx, ly, topo_instances, sz, halo_sz, file_has_halos)
    raw = collect(grid_group[name])

    # Older stores retained a singleton reduced dimension.
    if ndims(raw) == 3 && size(raw, 3) == 1
        raw = dropdims(raw; dims=3)
    end

    return construct_ossg_halo_padded_array(raw, name, FT, arch, lx, ly,
                                            topo_instances, sz, halo_sz, file_has_halos)
end
