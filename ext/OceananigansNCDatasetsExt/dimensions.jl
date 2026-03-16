#####
##### NetCDF Dimension Management
#####

# This file contains utilities for managing dimensions in NetCDF files,
# including default conventions and quality of life functions for Oceananigans outputs.
#
# 1. DIMENSION NAMING CONVENTIONS
#    - NetCDF dimensions use names based on grid direction: x, y, z
#      (or λ, φ, z for latitude-longitude grids).
#    - By default, Oceananigans conventions for staggered fields are followed,
#      but Unicode is avoided for NetCDF compatibility:
#         - e.g., xᶜᵃᵃ becomes x_caa.
#    - Users may provide a custom dimension naming function to override these defaults.
#
# 2. DIMENSION VALIDATION
#    - When appending to existing NetCDF files, dimensions are checked for compatibility.
#    - Ensures dimension sizes match expectations and coordinate values agree.
#    - Errors are raised if a mismatch is detected, protecting files from silent corruption.
#

"""
    create_field_dimensions!(ds, fd::AbstractField, dimension_name_generator; time_dependent=false, with_halos=false, array_type=Array{eltype(fd)})

Creates all dimensions for the given field `fd` in the NetCDF dataset `ds`. If the dimensions
already exist, they are validated to match the expected dimensions.

Arguments:
- `ds`: NetCDF dataset
- `fd`: AbstractField being written
- `dim_names`: Tuple of dimension names to create/validate
- `dimension_name_generator`: Function to generate dimension names
"""
function create_field_dimensions!(ds, fd::AbstractField, dimension_name_generator; time_dependent=false, with_halos=false, array_type=Array{eltype(fd)}, dimension_type=Float64)
    # Assess and create the dimensions for the field

    dimension_attributes = default_dimension_attributes(fd.grid, dimension_name_generator)
    spatial_dim_names = field_dimensions(fd, dimension_name_generator)
    spatial_dim_data = nodes(fd; with_halos)

    # Create dictionary of spatial dimensions and their data. Using OrderedDict to ensure the order of the dimensions is preserved.
    spatial_dim_names_dict = OrderedDict(spatial_dim_name => spatial_dim_array for (spatial_dim_name, spatial_dim_array) in zip(spatial_dim_names, spatial_dim_data))
    effective_spatial_dim_names = create_spatial_dimensions!(ds, spatial_dim_names_dict, dimension_attributes; dimension_type)

    # Create time dimension if needed
    if time_dependent
        "time" ∉ keys(ds.dim) && create_time_dimension!(ds, dimension_type=dimension_type)
        return (effective_spatial_dim_names..., "time") # Add "time" dimension if the field is time-dependent
    else
        return effective_spatial_dim_names
    end
end

"""
    create_spatial_dimensions!(dataset, dims, attributes_dict; array_type=Array{Float32}, kwargs...)

Create spatial dimensions in the NetCDF dataset and define corresponding variables to store
their coordinate values. Each dimension variable has itself as its sole dimension (e.g., the
`x` variable has dimension `x`). The dimensions are created if they don't exist, and validated
against provided arrays if they do exist. An error is thrown if the dimension already exists
but is different from the provided array.
"""
function create_spatial_dimensions!(dataset, dims, attributes_dict; dimension_type=Float64, kwargs...)
    effective_dim_names = []
    for (i, (dim_name, dim_array)) in enumerate(dims)
        dim_array isa Nothing && continue # Don't create anything if dim_array is Nothing
        dim_name == "" && continue # Don't create anything if dim_name is an empty string
        push!(effective_dim_names, dim_name)

        # Transform dim_array to the correct float type and ensure it's on the CPU
        dim_array = collect(dimension_type.(dim_array))

        if dim_name ∉ keys(dataset.dim)
            # Create missing dimension
            defVar(dataset, dim_name, dim_array, (dim_name,), attrib=attributes_dict[dim_name]; kwargs...)
        else
            # Validate existing dimension
            dataset_dim_array = collect(dataset[dim_name])
            if dataset_dim_array != collect(dim_array)
                throw(ArgumentError("Dimension '$dim_name' already exists in dataset but is different from expected.\n" *
                                    "  Actual:   $(dataset_dim_array) (length=$(length(dataset_dim_array)))\n" *
                                    "  Expected: $(dim_array) (length=$(length(dim_array)))"))
            end
        end
    end
    return tuple(effective_dim_names...)
end

#####
##### Gathering of grid dimensions
#####

function maybe_add_particle_dims!(dims, outputs)
    if "particles" in keys(outputs)  # TODO: Change this to look for ::LagrangianParticles in outputs?
        dims["particle_id"] = collect(1:length(outputs["particles"]))
    end
    return dims
end

function gather_vertical_dimensions(coordinate::StaticVerticalDiscretization, TZ, Nz, Hz, z_indices, with_halos, dim_name_generator)
    zᵃᵃᶠ_name = dim_name_generator("z", coordinate, nothing, nothing, f, Val(:z))
    zᵃᵃᶜ_name = dim_name_generator("z", coordinate, nothing, nothing, c, Val(:z))

    zᵃᵃᶠ_data = collect_dim(coordinate.cᵃᵃᶠ, f, TZ(), Nz, Hz, z_indices, with_halos)
    zᵃᵃᶜ_data = collect_dim(coordinate.cᵃᵃᶜ, c, TZ(), Nz, Hz, z_indices, with_halos)

    return Dict(zᵃᵃᶠ_name => zᵃᵃᶠ_data,
                zᵃᵃᶜ_name => zᵃᵃᶜ_data)
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

function gather_dimensions(outputs, grid::LatitudeLongitudeGrid, indices, with_halos, dim_name_generator)
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

    return dims
end

gather_dimensions(outputs, grid::ImmersedBoundaryGrid, args...) =
    gather_dimensions(outputs, grid.underlying_grid, args...)

#####
##### Mapping outputs/fields to dimensions
#####

function field_dimensions(fd::AbstractField, grid::RectilinearGrid, dim_name_generator)
    LX, LY, LZ = location(fd)
    TX, TY, TZ = topology(grid)

    x_dim_name = LX == Nothing ? "" : dim_name_generator("x", grid, LX(), nothing, nothing, Val(:x))
    y_dim_name = LY == Nothing ? "" : dim_name_generator("y", grid, nothing, LY(), nothing, Val(:y))
    z_dim_name = LZ == Nothing ? "" : dim_name_generator("z", grid, nothing, nothing, LZ(), Val(:z))

    return tuple(x_dim_name, y_dim_name, z_dim_name)
end

function field_dimensions(fd::AbstractField, grid::LatitudeLongitudeGrid, dim_name_generator)
    LΛ, LΦ, LZ = location(fd)
    TΛ, TΦ, TZ = topology(grid)

    λ_dim_name = LΛ == Nothing ? "" : dim_name_generator("λ", grid, LΛ(), nothing, nothing, Val(:x))
    φ_dim_name = LΦ == Nothing ? "" : dim_name_generator("φ", grid, nothing, LΦ(), nothing, Val(:y))
    z_dim_name = LZ == Nothing ? "" : dim_name_generator("z", grid, nothing, nothing, LZ(), Val(:z))

    return tuple(λ_dim_name, φ_dim_name, z_dim_name)
end

field_dimensions(fd::AbstractField, grid::ImmersedBoundaryGrid, dim_name_generator) =
    field_dimensions(fd, grid.underlying_grid, dim_name_generator)

field_dimensions(fd::AbstractField, dim_name_generator) =
    field_dimensions(fd, fd.grid, dim_name_generator)

#####
##### Dimension attributes
#####

const base_dimension_attributes = Dict("time"        => Dict("long_name" => "Time", "units" => "s"),
                                       "particle_id" => Dict("long_name" => "Particle ID"))

function default_vertical_dimension_attributes(coordinate::StaticVerticalDiscretization, dim_name_generator)
    zᵃᵃᶠ_name = dim_name_generator("z", coordinate, nothing, nothing, f, Val(:z))
    zᵃᵃᶜ_name = dim_name_generator("z", coordinate, nothing, nothing, c, Val(:z))

    Δzᵃᵃᶠ_name = dim_name_generator("Δz", coordinate, nothing, nothing, f, Val(:z))
    Δzᵃᵃᶜ_name = dim_name_generator("Δz", coordinate, nothing, nothing, c, Val(:z))

    zᵃᵃᶠ_attrs = Dict("long_name" => "Cell face locations in the z-direction.",   "units" => "m")
    zᵃᵃᶜ_attrs = Dict("long_name" => "Cell center locations in the z-direction.", "units" => "m")

    Δzᵃᵃᶠ_attrs = Dict("long_name" => "Spacings between cell centers (located at cell faces) in the z-direction.", "units" => "m")
    Δzᵃᵃᶜ_attrs = Dict("long_name" => "Spacings between cell faces (located at cell centers) in the z-direction.", "units" => "m")

    return Dict(zᵃᵃᶠ_name => zᵃᵃᶠ_attrs,
                zᵃᵃᶜ_name => zᵃᵃᶜ_attrs,
                Δzᵃᵃᶠ_name => Δzᵃᵃᶠ_attrs,
                Δzᵃᵃᶜ_name => Δzᵃᵃᶜ_attrs)
end

function default_dimension_attributes(grid::RectilinearGrid, dim_name_generator)
    xᶠᵃᵃ_name = dim_name_generator("x", grid, f, nothing, nothing, Val(:x))
    xᶜᵃᵃ_name = dim_name_generator("x", grid, c, nothing, nothing, Val(:x))
    yᵃᶠᵃ_name = dim_name_generator("y", grid, nothing, f, nothing, Val(:y))
    yᵃᶜᵃ_name = dim_name_generator("y", grid, nothing, c, nothing, Val(:y))

    Δxᶠᵃᵃ_name = dim_name_generator("Δx", grid, f, nothing, nothing, Val(:x))
    Δxᶜᵃᵃ_name = dim_name_generator("Δx", grid, c, nothing, nothing, Val(:x))
    Δyᵃᶠᵃ_name = dim_name_generator("Δy", grid, nothing, f, nothing, Val(:y))
    Δyᵃᶜᵃ_name = dim_name_generator("Δy", grid, nothing, c, nothing, Val(:y))

    xᶠᵃᵃ_attrs = Dict("long_name" => "Cell face locations in the x-direction.",   "units" => "m", "location" => "Face")
    xᶜᵃᵃ_attrs = Dict("long_name" => "Cell center locations in the x-direction.", "units" => "m", "location" => "Center")
    yᵃᶠᵃ_attrs = Dict("long_name" => "Cell face locations in the y-direction.",   "units" => "m", "location" => "Face")
    yᵃᶜᵃ_attrs = Dict("long_name" => "Cell center locations in the y-direction.", "units" => "m", "location" => "Center")

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

    vertical_dimension_attributes = default_vertical_dimension_attributes(grid.z, dim_name_generator)

    return merge(base_dimension_attributes,
                 horizontal_dimension_attributes,
                 vertical_dimension_attributes)
end

function default_dimension_attributes(grid::LatitudeLongitudeGrid, dim_name_generator)
    λᶠᵃᵃ_name = dim_name_generator("λ", grid, f, nothing, nothing, Val(:x))
    λᶜᵃᵃ_name = dim_name_generator("λ", grid, c, nothing, nothing, Val(:x))

    λᶠᵃᵃ_attrs = Dict("long_name" => "Cell face locations in the zonal direction.",   "units" => "degrees east")
    λᶜᵃᵃ_attrs = Dict("long_name" => "Cell center locations in the zonal direction.", "units" => "degrees east")

    φᵃᶠᵃ_name = dim_name_generator("φ", grid, nothing, f, nothing, Val(:y))
    φᵃᶜᵃ_name = dim_name_generator("φ", grid, nothing, c, nothing, Val(:y))

    φᵃᶠᵃ_attrs = Dict("long_name" => "Cell face locations in the meridional direction.",   "units" => "degrees north")
    φᵃᶜᵃ_attrs = Dict("long_name" => "Cell center locations in the meridional direction.", "units" => "degrees north")

    Δλᶠᵃᵃ_name = dim_name_generator("Δλ", grid, f, nothing, nothing, Val(:x))
    Δλᶜᵃᵃ_name = dim_name_generator("Δλ", grid, c, nothing, nothing, Val(:x))

    Δλᶠᵃᵃ_attrs = Dict("long_name" => "Angular spacings between cell faces in the zonal direction.",   "units" => "degrees")
    Δλᶜᵃᵃ_attrs = Dict("long_name" => "Angular spacings between cell centers in the zonal direction.", "units" => "degrees")

    Δφᵃᶠᵃ_name = dim_name_generator("Δλ", grid, nothing, f, nothing, Val(:y))
    Δφᵃᶜᵃ_name = dim_name_generator("Δλ", grid, nothing, c, nothing, Val(:y))

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

    vertical_dimension_attributes = default_vertical_dimension_attributes(grid.z, dim_name_generator)

    return merge(base_dimension_attributes,
                 horizontal_dimension_attributes,
                 vertical_dimension_attributes)
end

default_dimension_attributes(grid::ImmersedBoundaryGrid, dim_name_generator) =
    default_dimension_attributes(grid.underlying_grid, dim_name_generator)
