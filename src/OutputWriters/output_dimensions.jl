#####
##### Output dimensions and coordinates
#####

function effective_reduced_dimensions(field)
    location_reduced_dimensions = reduced_dimensions(field)
    topology_reduced_dimensions = findall(==(Flat), topology(field))
    return Tuple(unique((location_reduced_dimensions..., topology_reduced_dimensions...)))
end

function drop_reduced_dimensions(field::AbstractField, values)
    reduced_dimensions = effective_reduced_dimensions(field)
    return Tuple(value for (dimension, value) in enumerate(values) if dimension ∉ reduced_dimensions)
end

drop_reduced_dimensions(output::WindowedTimeAverage{<:AbstractField}, values) =
    drop_reduced_dimensions(output.operand, values)

function squeeze_reduced_dimensions(field::AbstractField, data; array_type=identity)
    data = array_type(data)
    reduced_dimensions = effective_reduced_dimensions(field)
    selectors = ntuple(dimension -> dimension in reduced_dimensions ? 1 : Colon(), 3)
    return getindex(data, selectors...)
end

squeeze_reduced_dimensions(output, data; kw...) = data
squeeze_reduced_dimensions(output::WindowedTimeAverage{<:AbstractField}, data; kw...) =
    squeeze_reduced_dimensions(output.operand, data; kw...)
squeeze_reduced_dimensions(field::AbstractField; array_type=Array{eltype(field)}) =
    squeeze_reduced_dimensions(field, parent(field); array_type)
squeeze_reduced_dimensions(output::WindowedTimeAverage{<:AbstractField}; kw...) =
    squeeze_reduced_dimensions(output.operand; kw...)

function inflate_reduced_dimensions(data, location, grid)
    ndims(data) == 3 && return data

    reduced_dimensions = ntuple(3) do dimension
        loc = location[dimension]
        return loc === Nothing || loc === nothing || topology(grid, dimension) === Flat
    end

    any(reduced_dimensions) || return data

    data_shape = size(data)
    data_dimension = 1
    inflated_shape = ntuple(3) do dimension
        reduced_dimensions[dimension] && return 1
        size = data_shape[data_dimension]
        data_dimension += 1
        return size
    end

    data = data isa AbstractArray ? data : fill(data)
    return reshape(data, inflated_shape)
end

function collect_dim(coordinate, location, topology, size, halo, indices, with_halos)
    with_halos && return collect(coordinate)
    indices = validate_index(indices, location, topology, size, halo)
    indices = restrict_to_interior(indices, location, topology, size)
    return collect(coordinate[indices])
end

#####
##### Dimension attributes
#####

const base_dimension_attributes = Dict("time"        => Dict("long_name" => "Time", "units" => "s"),
                                       "particle_id" => Dict("long_name" => "Particle ID"))

suffix_grid_entry(entry, grid_index) = entry
suffix_grid_entry(entry::NamedTuple, grid_index) =
    (array=entry.array, dims=Tuple(add_grid_suffix(name, grid_index) for name in entry.dims))

suffix_grid_keys(dims, grid_index) =
    Dict(add_grid_suffix(key, grid_index) => suffix_grid_entry(value, grid_index)
         for (key, value) in dims)

function default_vertical_dimension_attributes(coordinate::StaticVerticalDiscretization, dim_name_generator; grid_index=nothing)
    z = vertical_coordinate_name(coordinate)
    zᵃᵃᶠ_name = dim_name_generator(z, coordinate, nothing, nothing, f, Val(:z))
    zᵃᵃᶜ_name = dim_name_generator(z, coordinate, nothing, nothing, c, Val(:z))

    Δzᵃᵃᶠ_name = dim_name_generator("Δz", coordinate, nothing, nothing, f, Val(:z))
    Δzᵃᵃᶜ_name = dim_name_generator("Δz", coordinate, nothing, nothing, c, Val(:z))

    zᵃᵃᶠ_attrs = Dict("long_name" => "Cell face locations in the z-direction.",
                         "standard_name" => "height", "units" => "m",
                         "axis" => "Z", "positive" => "up")
    zᵃᵃᶜ_attrs = Dict("long_name" => "Cell center locations in the z-direction.",
                         "standard_name" => "height", "units" => "m",
                         "axis" => "Z", "positive" => "up")

    Δzᵃᵃᶠ_attrs = Dict("long_name" => "Spacings between cell centers (located at cell faces) in the z-direction.", "units" => "m")
    Δzᵃᵃᶜ_attrs = Dict("long_name" => "Spacings between cell faces (located at cell centers) in the z-direction.", "units" => "m")

    vertical_dimension_attributes = Dict(zᵃᵃᶠ_name  => zᵃᵃᶠ_attrs,
                                         zᵃᵃᶜ_name  => zᵃᵃᶜ_attrs,
                                         Δzᵃᵃᶠ_name => Δzᵃᵃᶠ_attrs,
                                         Δzᵃᵃᶜ_name => Δzᵃᵃᶜ_attrs)

    return suffix_grid_keys(vertical_dimension_attributes, grid_index)
end

function default_vertical_dimension_attributes(coordinate::AbstractVerticalCoordinate, dim_name_generator; grid_index=nothing)
    # `r` is the reference (Lagrangian) coordinate. Physical depth `z = z(r, η, …)` is
    # reconstructible at read time from `r` and the time-varying free-surface `η`
    # written separately (see grid_reconstruction.jl).
    r = vertical_coordinate_name(coordinate)
    rᵃᵃᶠ_name = dim_name_generator(r, coordinate, nothing, nothing, f, Val(:z))
    rᵃᵃᶜ_name = dim_name_generator(r, coordinate, nothing, nothing, c, Val(:z))

    Δrᵃᵃᶠ_name = dim_name_generator("Δr", coordinate, nothing, nothing, f, Val(:z))
    Δrᵃᵃᶜ_name = dim_name_generator("Δr", coordinate, nothing, nothing, c, Val(:z))

    long_face   = "Reference cell-face locations in the vertical (Lagrangian r). Physical depth is reconstructible from r and η."
    long_center = "Reference cell-center locations in the vertical (Lagrangian r). Physical depth is reconstructible from r and η."

    rᵃᵃᶠ_attrs = Dict("long_name" => long_face,   "units" => "m", "axis" => "Z", "positive" => "up")
    rᵃᵃᶜ_attrs = Dict("long_name" => long_center, "units" => "m", "axis" => "Z", "positive" => "up")

    Δrᵃᵃᶠ_attrs = Dict("long_name" => "Reference spacings between cell centers (at cell faces) in the vertical.", "units" => "m")
    Δrᵃᵃᶜ_attrs = Dict("long_name" => "Reference spacings between cell faces (at cell centers) in the vertical.", "units" => "m")

    vertical_dimension_attributes = Dict(rᵃᵃᶠ_name  => rᵃᵃᶠ_attrs,
                                         rᵃᵃᶜ_name  => rᵃᵃᶜ_attrs,
                                         Δrᵃᵃᶠ_name => Δrᵃᵃᶠ_attrs,
                                         Δrᵃᵃᶜ_name => Δrᵃᵃᶜ_attrs)

    return suffix_grid_keys(vertical_dimension_attributes, grid_index)
end

function default_dimension_attributes(grid::RectilinearGrid, dim_name_generator; grid_index=nothing)
    xᶠᵃᵃ_name = dim_name_generator("x", grid, f, nothing, nothing, Val(:x))
    xᶜᵃᵃ_name = dim_name_generator("x", grid, c, nothing, nothing, Val(:x))
    yᵃᶠᵃ_name = dim_name_generator("y", grid, nothing, f, nothing, Val(:y))
    yᵃᶜᵃ_name = dim_name_generator("y", grid, nothing, c, nothing, Val(:y))

    Δxᶠᵃᵃ_name = dim_name_generator("Δx", grid, f, nothing, nothing, Val(:x))
    Δxᶜᵃᵃ_name = dim_name_generator("Δx", grid, c, nothing, nothing, Val(:x))
    Δyᵃᶠᵃ_name = dim_name_generator("Δy", grid, nothing, f, nothing, Val(:y))
    Δyᵃᶜᵃ_name = dim_name_generator("Δy", grid, nothing, c, nothing, Val(:y))

    xᶠᵃᵃ_attrs = Dict("long_name" => "Cell face locations in the x-direction.",
                         "standard_name" => "projection_x_coordinate", "units" => "m",
                         "axis" => "X", "location" => "Face")
    xᶜᵃᵃ_attrs = Dict("long_name" => "Cell center locations in the x-direction.",
                         "standard_name" => "projection_x_coordinate", "units" => "m",
                         "axis" => "X", "location" => "Center")
    yᵃᶠᵃ_attrs = Dict("long_name" => "Cell face locations in the y-direction.",
                         "standard_name" => "projection_y_coordinate", "units" => "m",
                         "axis" => "Y", "location" => "Face")
    yᵃᶜᵃ_attrs = Dict("long_name" => "Cell center locations in the y-direction.",
                         "standard_name" => "projection_y_coordinate", "units" => "m",
                         "axis" => "Y", "location" => "Center")

    Δxᶠᵃᵃ_attrs = Dict("long_name" => "Spacings between cell centers (located at the cell faces) in the x-direction.", "units" => "m", "location" => "Face")
    Δxᶜᵃᵃ_attrs = Dict("long_name" => "Spacings between cell faces (located at the cell centers) in the x-direction.", "units" => "m", "location" => "Center")
    Δyᵃᶠᵃ_attrs = Dict("long_name" => "Spacings between cell centers (located at cell faces) in the y-direction.",     "units" => "m", "location" => "Face")
    Δyᵃᶜᵃ_attrs = Dict("long_name" => "Spacings between cell faces (located at cell centers) in the y-direction.",     "units" => "m", "location" => "Center")

    horizontal_dimension_attributes = Dict(xᶠᵃᵃ_name  => xᶠᵃᵃ_attrs,
                                           xᶜᵃᵃ_name  => xᶜᵃᵃ_attrs,
                                           yᵃᶠᵃ_name  => yᵃᶠᵃ_attrs,
                                           yᵃᶜᵃ_name  => yᵃᶜᵃ_attrs,
                                           Δxᶠᵃᵃ_name => Δxᶠᵃᵃ_attrs,
                                           Δxᶜᵃᵃ_name => Δxᶜᵃᵃ_attrs,
                                           Δyᵃᶠᵃ_name => Δyᵃᶠᵃ_attrs,
                                           Δyᵃᶜᵃ_name => Δyᵃᶜᵃ_attrs)

    horizontal_dimension_attributes = suffix_grid_keys(horizontal_dimension_attributes, grid_index)
    vertical_dimension_attributes   = default_vertical_dimension_attributes(grid.z, dim_name_generator; grid_index)

    return merge(base_dimension_attributes,
                 horizontal_dimension_attributes,
                 vertical_dimension_attributes)
end

function default_dimension_attributes(grid::LatitudeLongitudeGrid, dim_name_generator; grid_index=nothing)
    λᶠᵃᵃ_name = dim_name_generator("λ", grid, f, nothing, nothing, Val(:x))
    λᶜᵃᵃ_name = dim_name_generator("λ", grid, c, nothing, nothing, Val(:x))

    λᶠᵃᵃ_attrs = Dict("long_name" => "Cell face locations in the zonal direction.",
                         "standard_name" => "longitude", "units" => "degrees_east",
                         "axis" => "X")
    λᶜᵃᵃ_attrs = Dict("long_name" => "Cell center locations in the zonal direction.",
                         "standard_name" => "longitude", "units" => "degrees_east",
                         "axis" => "X")

    φᵃᶠᵃ_name = dim_name_generator("φ", grid, nothing, f, nothing, Val(:y))
    φᵃᶜᵃ_name = dim_name_generator("φ", grid, nothing, c, nothing, Val(:y))

    φᵃᶠᵃ_attrs = Dict("long_name" => "Cell face locations in the meridional direction.",
                         "standard_name" => "latitude", "units" => "degrees_north",
                         "axis" => "Y")
    φᵃᶜᵃ_attrs = Dict("long_name" => "Cell center locations in the meridional direction.",
                         "standard_name" => "latitude", "units" => "degrees_north",
                         "axis" => "Y")

    Δλᶠᵃᵃ_name = dim_name_generator("Δλ", grid, f, nothing, nothing, Val(:x))
    Δλᶜᵃᵃ_name = dim_name_generator("Δλ", grid, c, nothing, nothing, Val(:x))

    Δλᶠᵃᵃ_attrs = Dict("long_name" => "Angular spacings between cell faces in the zonal direction.",   "units" => "degrees")
    Δλᶜᵃᵃ_attrs = Dict("long_name" => "Angular spacings between cell centers in the zonal direction.", "units" => "degrees")

    Δφᵃᶠᵃ_name = dim_name_generator("Δφ", grid, nothing, f, nothing, Val(:y))
    Δφᵃᶜᵃ_name = dim_name_generator("Δφ", grid, nothing, c, nothing, Val(:y))

    Δφᵃᶠᵃ_attrs = Dict("long_name" => "Angular spacings between cell faces in the meridional direction.",   "units" => "degrees")
    Δφᵃᶜᵃ_attrs = Dict("long_name" => "Angular spacings between cell centers in the meridional direction.", "units" => "degrees")

    Δxᶠᶠᵃ_name = dim_name_generator("Δx", grid, f, f, nothing, Val(:x))
    Δxᶠᶜᵃ_name = dim_name_generator("Δx", grid, f, c, nothing, Val(:x))
    Δxᶜᶠᵃ_name = dim_name_generator("Δx", grid, c, f, nothing, Val(:x))
    Δxᶜᶜᵃ_name = dim_name_generator("Δx", grid, c, c, nothing, Val(:x))

    Δxᶠᶠᵃ_attrs = Dict("long_name" => "Geodesic spacings in the zonal direction between the cell located at (Face, Face).",
                       "units" => "m")

    Δxᶠᶜᵃ_attrs = Dict("long_name" => "Geodesic spacings in the zonal direction between the cell located at (Face, Center).",
                       "units" => "m")

    Δxᶜᶠᵃ_attrs = Dict("long_name" => "Geodesic spacings in the zonal direction between the cell located at (Center, Face).",
                       "units" => "m")

    Δxᶜᶜᵃ_attrs = Dict("long_name" => "Geodesic spacings in the zonal direction between the cell located at (Center, Center).",
                       "units" => "m")

    Δyᶠᶠᵃ_name = dim_name_generator("Δy", grid, f, f, nothing, Val(:y))
    Δyᶠᶜᵃ_name = dim_name_generator("Δy", grid, f, c, nothing, Val(:y))
    Δyᶜᶠᵃ_name = dim_name_generator("Δy", grid, c, f, nothing, Val(:y))
    Δyᶜᶜᵃ_name = dim_name_generator("Δy", grid, c, c, nothing, Val(:y))

    Δyᶠᶠᵃ_attrs = Dict("long_name" => "Geodesic spacings in the meridional direction between the cell located at (Face, Face).",
                       "units" => "m")

    Δyᶠᶜᵃ_attrs = Dict("long_name" => "Geodesic spacings in the meridional direction between the cell located at (Face, Center).",
                       "units" => "m")

    Δyᶜᶠᵃ_attrs = Dict("long_name" => "Geodesic spacings in the meridional direction between the cell located at (Center, Face).",
                       "units" => "m")

    Δyᶜᶜᵃ_attrs = Dict("long_name" => "Geodesic spacings in the meridional direction between the cell located at (Center, Center).",
                       "units" => "m")

    horizontal_dimension_attributes = Dict(λᶠᵃᵃ_name  => λᶠᵃᵃ_attrs,
                                           λᶜᵃᵃ_name  => λᶜᵃᵃ_attrs,
                                           φᵃᶠᵃ_name  => φᵃᶠᵃ_attrs,
                                           φᵃᶜᵃ_name  => φᵃᶜᵃ_attrs,
                                           Δλᶠᵃᵃ_name => Δλᶠᵃᵃ_attrs,
                                           Δλᶜᵃᵃ_name => Δλᶜᵃᵃ_attrs,
                                           Δφᵃᶠᵃ_name => Δφᵃᶠᵃ_attrs,
                                           Δφᵃᶜᵃ_name => Δφᵃᶜᵃ_attrs,
                                           Δxᶠᶠᵃ_name => Δxᶠᶠᵃ_attrs,
                                           Δxᶠᶜᵃ_name => Δxᶠᶜᵃ_attrs,
                                           Δxᶜᶠᵃ_name => Δxᶜᶠᵃ_attrs,
                                           Δxᶜᶜᵃ_name => Δxᶜᶜᵃ_attrs,
                                           Δyᶠᶠᵃ_name => Δyᶠᶠᵃ_attrs,
                                           Δyᶠᶜᵃ_name => Δyᶠᶜᵃ_attrs,
                                           Δyᶜᶠᵃ_name => Δyᶜᶠᵃ_attrs,
                                           Δyᶜᶜᵃ_name => Δyᶜᶜᵃ_attrs)

    horizontal_dimension_attributes = suffix_grid_keys(horizontal_dimension_attributes, grid_index)
    vertical_dimension_attributes   = default_vertical_dimension_attributes(grid.z, dim_name_generator; grid_index)

    return merge(base_dimension_attributes,
                 horizontal_dimension_attributes,
                 vertical_dimension_attributes)
end

function default_dimension_attributes(grid::OrthogonalSphericalShellGrid, dim_name_generator; grid_index=nothing)
    horizontal_dimension_attributes = Dict{String, Any}()

    # Bare logical-index dimension attributes.
    for (loc, label) in ((c, "Center"), (f, "Face"))
        xi_name = ossg_xi_name(grid, loc, dim_name_generator)
        eta_name = ossg_eta_name(grid, loc, dim_name_generator)
        horizontal_dimension_attributes[xi_name]  = Dict("long_name" => "Logical x-index ($label)")
        horizontal_dimension_attributes[eta_name] = Dict("long_name" => "Logical y-index ($label)")
    end

    # 2D auxiliary coordinate variables — λ and φ at each Arakawa-C stagger location.
    # Also pre-register attribute defaults for the grid-metric variables that
    # `gather_grid_metrics(::OrthogonalSphericalShellGrid)` emits at each stagger.
    for (lx, ly) in ((c, c), (f, c), (c, f), (f, f))
        λ_name  = dim_name_generator("λ",  grid, lx, ly, nothing, Val(:x))
        φ_name  = dim_name_generator("φ",  grid, lx, ly, nothing, Val(:y))
        Δx_name = dim_name_generator("Δx", grid, lx, ly, nothing, Val(:x))
        Δy_name = dim_name_generator("Δy", grid, lx, ly, nothing, Val(:y))
        Az_name = dim_name_generator("Az", grid, lx, ly, nothing, Val(:x))

        loc_label = "($(lx isa Center ? "Center" : "Face"), $(ly isa Center ? "Center" : "Face"))"
        horizontal_dimension_attributes[λ_name] = Dict("long_name"     => "Longitude at $(loc_label)",
                                                       "standard_name" => "longitude",
                                                       "units"         => "degrees_east")
        horizontal_dimension_attributes[φ_name] = Dict("long_name"     => "Latitude at $(loc_label)",
                                                       "standard_name" => "latitude",
                                                       "units"         => "degrees_north")
        horizontal_dimension_attributes[Δx_name] = Dict("long_name" => "Curvilinear x-spacing at $(loc_label).", "units" => "m")
        horizontal_dimension_attributes[Δy_name] = Dict("long_name" => "Curvilinear y-spacing at $(loc_label).", "units" => "m")
        horizontal_dimension_attributes[Az_name] = Dict("long_name" => "Horizontal cell area at $(loc_label).",  "units" => "m^2")
    end

    horizontal_dimension_attributes = suffix_grid_keys(horizontal_dimension_attributes, grid_index)
    vertical_dimension_attributes   = default_vertical_dimension_attributes(grid.z, dim_name_generator; grid_index)

    return merge(base_dimension_attributes,
                 horizontal_dimension_attributes,
                 vertical_dimension_attributes)
end

default_dimension_attributes(grid::ImmersedBoundaryGrid, dim_name_generator; kw...) =
    default_dimension_attributes(grid.underlying_grid, dim_name_generator; kw...)


#####
##### Gathering of grid dimensions
#####

function maybe_add_particle_dims!(dims, outputs)
    if "particles" in keys(outputs)
        dims["particle_id"] = collect(1:length(outputs["particles"]))
    end
    return dims
end

#####
##### Vertical dimensions
#####

function gather_vertical_dimensions(coordinate::StaticVerticalDiscretization, TZ, Nz, Hz, z_indices, with_halos, dim_name_generator)
    zᵃᵃᶠ_name = dim_name_generator("z", coordinate, nothing, nothing, f, Val(:z))
    zᵃᵃᶜ_name = dim_name_generator("z", coordinate, nothing, nothing, c, Val(:z))

    zᵃᵃᶠ_data = collect_dim(coordinate.cᵃᵃᶠ, f, TZ(), Nz, Hz, z_indices, with_halos)
    zᵃᵃᶜ_data = collect_dim(coordinate.cᵃᵃᶜ, c, TZ(), Nz, Hz, z_indices, with_halos)

    return Dict(zᵃᵃᶠ_name => zᵃᵃᶠ_data,
                zᵃᵃᶜ_name => zᵃᵃᶜ_data)
end

# For MutableVerticalDiscretization, the saved 1D coordinate is the *reference* (Lagrangian)
# coordinate `r`. The physical `z = z(r, η, …)` is reconstructible at read time from `r` and
# the time-varying `η` (output separately).
function gather_vertical_dimensions(coordinate::AbstractVerticalCoordinate, TZ, Nz, Hz, z_indices, with_halos, dim_name_generator)
    rᵃᵃᶠ_name = dim_name_generator("r", coordinate, nothing, nothing, f, Val(:z))
    rᵃᵃᶜ_name = dim_name_generator("r", coordinate, nothing, nothing, c, Val(:z))

    rᵃᵃᶠ_data = collect_dim(coordinate.cᵃᵃᶠ, f, TZ(), Nz, Hz, z_indices, with_halos)
    rᵃᵃᶜ_data = collect_dim(coordinate.cᵃᵃᶜ, c, TZ(), Nz, Hz, z_indices, with_halos)

    return Dict(rᵃᵃᶠ_name => rᵃᵃᶠ_data,
                rᵃᵃᶜ_name => rᵃᵃᶜ_data)
end

#####
##### Horizontal dimensions (per grid type)
#####

function gather_dimensions(outputs, grid::RectilinearGrid, indices, with_halos, dim_name_generator; grid_index=nothing)
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

    return suffix_grid_keys(dims, grid_index)
end

function gather_dimensions(outputs, grid::LatitudeLongitudeGrid, indices, with_halos, dim_name_generator; grid_index=nothing)
    TΛ, TΦ, TZ = topology(grid)
    Nλ, Nφ, Nz = size(grid)
    Hλ, Hφ, Hz = halo_size(grid)

    dims = Dict()

    if TΛ != Flat
        λᶠᵃᵃ_name = dim_name_generator("λ", grid, f, nothing, nothing, Val(:x))
        λᶜᵃᵃ_name = dim_name_generator("λ", grid, c, nothing, nothing, Val(:x))

        λᶠᵃᵃ_data = collect_dim(grid.λᶠᵃᵃ, f, TΛ(), Nλ, Hλ, indices[1], with_halos)
        λᶜᵃᵃ_data = collect_dim(grid.λᶜᵃᵃ, c, TΛ(), Nλ, Hλ, indices[1], with_halos)

        dims[λᶠᵃᵃ_name] = λᶠᵃᵃ_data
        dims[λᶜᵃᵃ_name] = λᶜᵃᵃ_data
    end

    if TΦ != Flat
        φᵃᶠᵃ_name = dim_name_generator("φ", grid, nothing, f, nothing, Val(:y))
        φᵃᶜᵃ_name = dim_name_generator("φ", grid, nothing, c, nothing, Val(:y))

        φᵃᶠᵃ_data = collect_dim(grid.φᵃᶠᵃ, f, TΦ(), Nφ, Hφ, indices[2], with_halos)
        φᵃᶜᵃ_data = collect_dim(grid.φᵃᶜᵃ, c, TΦ(), Nφ, Hφ, indices[2], with_halos)

        dims[φᵃᶠᵃ_name] = φᵃᶠᵃ_data
        dims[φᵃᶜᵃ_name] = φᵃᶜᵃ_data
    end

    if TZ != Flat
        vertical_dims = gather_vertical_dimensions(grid.z, TZ, Nz, Hz, indices[3], with_halos, dim_name_generator)
        dims = merge(dims, vertical_dims)
    end

    maybe_add_particle_dims!(dims, outputs)

    return suffix_grid_keys(dims, grid_index)
end

#####
##### OrthogonalSphericalShellGrid (CF §5.2: 2D auxiliary coordinates)
#####
#
# OSSG (TripolarGrid, RotatedLatitudeLongitudeGrid, ConformalCubedSpherePanelGrid) stores
# 2D arrays for `λ` and `φ` at each Arakawa-C stagger location. We follow CF §5.2:
#   - Logical-index dimensions `i_caa`/`i_faa`/`j_aca`/`j_afa` are bare dimensions
#     with no coordinate variable.
#   - The eight 2D `λ_**` and `φ_**` arrays are written as auxiliary coordinate variables
#     dimensioned `(i_*, j_*)`.
#   - Each data field carries a `coordinates = "λ_** φ_** z_aac"` attribute so that
#     CF-aware tools (xarray, ncview, Panoply, CDO) pick up the right lat/lon pair.
#

# 2D analogue of `collect_dim`: take a 2D coordinate or metric array, optionally trim halos,
# and return a plain CPU `Array{T,2}`. Indices for OSSG are 2-tuples `(i_range, j_range)`.
function collect_2d(arr, ℓx, ℓy, Tx, Ty, Nx, Ny, Hx, Hy, indices, with_halos)
    if with_halos
        return collect(arr)
    else
        i_range = validate_index(indices[1], ℓx, Tx, Nx, Hx)
        j_range = validate_index(indices[2], ℓy, Ty, Ny, Hy)
        i_range = restrict_to_interior(i_range, ℓx, Tx, Nx)
        j_range = restrict_to_interior(j_range, ℓy, Ty, Ny)
        return collect(view(arr, i_range, j_range))
    end
end

# Bare horizontal index dim names — used by `field_dimensions` and as the `dims` of the
# 2D aux coords. The names go through the same `dim_name_generator` machinery as
# everything else, so a user-supplied custom generator can override them.
ossg_xi_name(grid, LX, gen) = gen("i", grid, LX, nothing, nothing, Val(:x))
ossg_eta_name(grid, LY, gen) = gen("j", grid, nothing, LY, nothing, Val(:y))

function gather_dimensions(outputs, grid::OrthogonalSphericalShellGrid, indices, with_halos, dim_name_generator; grid_index=nothing)
    TX, TY, TZ = topology(grid)
    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    # OSSG horizontal axes cannot be flat.
    (TX == Flat || TY == Flat) && error("Flat horizontal topology is not supported on OrthogonalSphericalShellGrid output.")

    dims = Dict()

    # 2D auxiliary coordinate variables — one λ and one φ per Arakawa-C stagger location.
    for (lx, ly) in ((c, c), (f, c), (c, f), (f, f))
        λ_name = dim_name_generator("λ", grid, lx, ly, nothing, Val(:x))
        φ_name = dim_name_generator("φ", grid, lx, ly, nothing, Val(:y))

        xi_name  = ossg_xi_name(grid, lx, dim_name_generator)
        eta_name = ossg_eta_name(grid, ly, dim_name_generator)

        λ_data = collect_2d(λnodes(grid, lx, ly; with_halos=true), lx, ly, TX(), TY(), Nx, Ny, Hx, Hy, (indices[1], indices[2]), with_halos)
        φ_data = collect_2d(φnodes(grid, lx, ly; with_halos=true), lx, ly, TX(), TY(), Nx, Ny, Hx, Hy, (indices[1], indices[2]), with_halos)

        dims[λ_name] = (array = λ_data, dims = (xi_name, eta_name))
        dims[φ_name] = (array = φ_data, dims = (xi_name, eta_name))
    end

    if TZ != Flat
        vertical_dims = gather_vertical_dimensions(grid.z, TZ, Nz, Hz, indices[3], with_halos, dim_name_generator)
        dims = merge(dims, vertical_dims)
    end

    maybe_add_particle_dims!(dims, outputs)

    return suffix_grid_keys(dims, grid_index)
end

gather_dimensions(outputs, grid::ImmersedBoundaryGrid, args...; kw...) =
    gather_dimensions(outputs, grid.underlying_grid, args...; kw...)


#####
##### Mapping outputs/fields to dimensions
#####

function field_dimensions(fd::AbstractField, grid::RectilinearGrid, dim_name_generator; grid_index=nothing)
    LX, LY, LZ = location(fd)

    z = vertical_coordinate_name(grid)
    x_dim_name = LX == Nothing ? "" : dim_name_generator("x", grid, LX(), nothing, nothing, Val(:x))
    y_dim_name = LY == Nothing ? "" : dim_name_generator("y", grid, nothing, LY(), nothing, Val(:y))
    z_dim_name = LZ == Nothing ? "" : dim_name_generator(z,   grid, nothing, nothing, LZ(), Val(:z))

    return Tuple(add_grid_suffix(dim_name, grid_index) for dim_name in (x_dim_name, y_dim_name, z_dim_name))
end

function field_dimensions(fd::AbstractField, grid::LatitudeLongitudeGrid, dim_name_generator; grid_index=nothing)
    LΛ, LΦ, LZ = location(fd)

    z = vertical_coordinate_name(grid)
    λ_dim_name = LΛ == Nothing ? "" : dim_name_generator("λ", grid, LΛ(), nothing, nothing, Val(:x))
    φ_dim_name = LΦ == Nothing ? "" : dim_name_generator("φ", grid, nothing, LΦ(), nothing, Val(:y))
    z_dim_name = LZ == Nothing ? "" : dim_name_generator(z,   grid, nothing, nothing, LZ(), Val(:z))

    return Tuple(add_grid_suffix(dim_name, grid_index) for dim_name in (λ_dim_name, φ_dim_name, z_dim_name))
end

function field_dimensions(fd::AbstractField, grid::OrthogonalSphericalShellGrid, dim_name_generator; grid_index=nothing)
    LX, LY, LZ = location(fd)

    # On OSSG, field dimensions are the bare horizontal index dimensions (i_*, j_*)
    # — *not* the 2D λ/φ aux coords. Physical position comes from `coordinates` attribute
    # added per-field elsewhere.
    z = vertical_coordinate_name(grid)
    x_dim_name = LX == Nothing ? "" : ossg_xi_name(grid, LX(), dim_name_generator)
    y_dim_name = LY == Nothing ? "" : ossg_eta_name(grid, LY(), dim_name_generator)
    z_dim_name = LZ == Nothing ? "" : dim_name_generator(z, grid, nothing, nothing, LZ(), Val(:z))

    return Tuple(add_grid_suffix(dim_name, grid_index) for dim_name in (x_dim_name, y_dim_name, z_dim_name))
end

field_dimensions(fd::AbstractField, grid::ImmersedBoundaryGrid, dim_name_generator; kw...) =
    field_dimensions(fd, grid.underlying_grid, dim_name_generator; kw...)

field_dimensions(fd::AbstractField, dim_name_generator; kw...) =
    field_dimensions(fd, grid(fd), dim_name_generator; kw...)

field_auxiliary_coordinates(fd::AbstractField, dim_name_generator; kw...) =
    field_auxiliary_coordinates(fd, grid(fd), dim_name_generator; kw...)

field_auxiliary_coordinates(fd, grid, dim_name_generator; grid_index=nothing) = String[]

function field_auxiliary_coordinates(fd, grid::OrthogonalSphericalShellGrid, dim_name_generator; grid_index=nothing)
    LX, LY, LZ = location(fd)
    coordinates = String[]

    if LX !== Nothing && LY !== Nothing
        λ_name = dim_name_generator("λ", grid, LX(), LY(), nothing, Val(:x))
        φ_name = dim_name_generator("φ", grid, LX(), LY(), nothing, Val(:y))
        push!(coordinates, add_grid_suffix(λ_name, grid_index), add_grid_suffix(φ_name, grid_index))
    end

    if LZ !== Nothing
        vertical_name = vertical_coordinate_name(grid)
        coordinate_name = dim_name_generator(vertical_name, grid, nothing, nothing, LZ(), Val(:z))
        push!(coordinates, add_grid_suffix(coordinate_name, grid_index))
    end

    return coordinates
end

field_auxiliary_coordinates(fd, grid::ImmersedBoundaryGrid, dim_name_generator; kw...) =
    field_auxiliary_coordinates(fd, grid.underlying_grid, dim_name_generator; kw...)
