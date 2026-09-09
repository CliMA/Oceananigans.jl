#####
##### NetCDF grid serialization and reconstruction
#####


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
        pop!(immersed_grid_args, :bottom_height)
        ibg_group = defGroup(ds, group_name; attrib=convert_for_netcdf(immersed_grid_args))
        defVar(ibg_group, "bottom_height", bottom_height_field(grid))

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
    raw = collect(ds[name])
    return construct_ossg_halo_padded_array(raw, name, FT, arch, lx, ly,
                                            topo_instances, sz, halo_sz, file_has_halos)
end
