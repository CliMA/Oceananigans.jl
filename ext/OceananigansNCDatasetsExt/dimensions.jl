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
Base.@nospecializeinfer function create_field_dimensions!(ds, @nospecialize(fd::AbstractField), dimension_name_generator; time_dependent=false, with_halos=false, array_type=Array{eltype(fd)}, dimension_type=Float64, grid_index=nothing)
    # `field_dimensions` returns a 3-tuple with `""` in slots where the field has a
    # Nothing location or the grid axis is Flat. The "effective" dim names are what
    # actually go into the variable's NetCDF signature.
    spatial_dim_names = field_dimensions(fd, dimension_name_generator; grid_index)

    # `create_field_coord_variables!` may rename windowed axes (see `resolve_windowed_dim_name`),
    # so the variable's signature must use the resolved names it returns.
    resolved_dim_names = create_field_coord_variables!(ds, fd, grid(fd), spatial_dim_names, dimension_name_generator;
                                                       with_halos, dimension_type, grid_index)

    effective_dim_names = tuple(filter(!isempty, resolved_dim_names)...)

    if time_dependent
        "time" ∉ keys(ds.dim) && create_time_dimension!(ds, dimension_type=dimension_type)
        return (effective_dim_names..., "time")
    else
        return effective_dim_names
    end
end

# A "windowed" field (e.g. the free-surface displacement, or a user `view` that pins an axis
# to a sub-range) shares a location with full-extent fields but covers only part of the
# corresponding axis. Because a NetCDF coordinate variable maps a dimension name to a single
# set of values, such a field needs its own dimension whenever its coordinate values differ
# from the canonical full-extent dimension already in the dataset. Resolve the base name to
# itself when it is unused or its values already match, otherwise to the first free
# `name_2`, `name_3`, … whose values match (deduping fields that share the same window).
#
# This only applies to axes the field actually windows (a non-`Colon` index). Axes with the
# default `Colon` index keep the strict create-or-validate in `create_spatial_dimensions!`,
# so a dimension pre-created with mismatched values still errors rather than being aliased.
function resolve_windowed_dim_name(ds, base_name, index, data, dimension_type)
    (base_name == "" || data === nothing || index === Colon()) && return base_name
    target = collect(dimension_type.(data))
    candidate = base_name
    n = 1
    while candidate ∈ keys(ds)
        collect(ds[candidate]) == target && return candidate
        n += 1
        candidate = string(base_name, "_", n)
    end
    return candidate
end

# Default (1D-coordinate) path — RectilinearGrid and LatitudeLongitudeGrid: zip the
# field's (windowing-resolved) NetCDF dim names with the field's `nodes(fd)` 1D arrays and
# pass them through `create_spatial_dimensions!`, which creates missing coord vars or
# validates existing ones against the field's nodes. Returns the resolved dim names.
Base.@nospecializeinfer function create_field_coord_variables!(ds, @nospecialize(fd), @nospecialize(grid), spatial_dim_names, dim_name_generator;
                                        with_halos, dimension_type, grid_index)
    dimension_attributes = default_dimension_attributes(grid, dim_name_generator; grid_index)
    spatial_dim_data = nodes(fd; with_halos)

    resolved_dim_names = map(spatial_dim_names, indices(fd), spatial_dim_data) do name, index, data
        resolve_windowed_dim_name(ds, name, index, data, dimension_type)
    end

    # A renamed windowed axis inherits the attributes of the base coordinate it derives from.
    for (base, resolved) in zip(spatial_dim_names, resolved_dim_names)
        if resolved != base && haskey(dimension_attributes, base) && !haskey(dimension_attributes, resolved)
            dimension_attributes[resolved] = dimension_attributes[base]
        end
    end

    spatial_dim_names_dict = OrderedDict(name => data
                                         for (name, data) in zip(resolved_dim_names, spatial_dim_data))
    create_spatial_dimensions!(ds, spatial_dim_names_dict, dimension_attributes; dimension_type)

    return resolved_dim_names
end

# OrthogonalSphericalShellGrid path: dimensions are 1D bare `i_*`/`j_*` indices plus the
# vertical, and the lat/lon are 2D auxiliary coord variables that don't correspond
# positionally to `nodes(fd)`. The common path is that `gather_dimensions` has already
# created these dimensions at file init, so this is a no-op. But fields can also be
# written into subgroups (e.g. `bottom_height` for an `ImmersedBoundaryGrid`'s
# reconstruction record) whose dimension scope is local — for those, we need to
# `defDim` the missing bare dims here on the fly, using the field's interior shape.
function create_field_coord_variables!(ds, fd, grid::OrthogonalSphericalShellGrid,
                                        spatial_dim_names, dim_name_generator;
                                        with_halos, dimension_type, grid_index)
    field_sizes = size(interior(fd))
    for (dname, dsize) in zip(spatial_dim_names, field_sizes)
        isempty(dname) && continue
        dname ∈ keys(ds.dim) && continue
        defDim(ds, dname, dsize)
    end
    return spatial_dim_names
end

# Defer through ImmersedBoundaryGrid to the underlying grid's dispatch.
create_field_coord_variables!(ds, fd, grid::ImmersedBoundaryGrid, spatial_dim_names, gen; kw...) =
    create_field_coord_variables!(ds, fd, grid.underlying_grid, spatial_dim_names, gen; kw...)

"""
    create_spatial_dimensions!(dataset, dims, attributes_dict; array_type=Array{Float32}, kwargs...)

Create spatial dimensions in the NetCDF dataset and define corresponding variables to store
their coordinate values. Each dimension variable has itself as its sole dimension (e.g., the
`x` variable has dimension `x`). The dimensions are created if they don't exist, and validated
against provided arrays if they do exist. An error is thrown if the dimension already exists
but is different from the provided array.
"""
#
# Entries in the `dims` dict passed to `create_spatial_dimensions!` are either:
#   - a `NamedTuple` `(array, dims)` where `dims` is a tuple of the NetCDF dimension names
#     the variable spans. For 2D auxiliary coordinates (e.g. λ/φ on an
#     `OrthogonalSphericalShellGrid`) `dims` is a pair of *other* dimension names like
#     `("i_caa", "j_aca")`; those underlying dimensions are created with `defDim` here.
#   - a plain `AbstractArray`, which is treated as a 1D coordinate variable whose
#     dimension is itself (the variable's name `var_name` doubles as the dim name).
#
# A `nothing` array (or a `(array = nothing, dims = …)` entry) skips creation
# (used when a topology is `Flat`).
#

function create_spatial_dimensions!(dataset, dims, attributes_dict; dimension_type=Float64, kwargs...)
    effective_dim_names = String[]
    for (var_name, entry) in dims
        var_name == "" && continue # Skip empty names

        # Normalize to (array, var_dims). A bare `AbstractArray` is interpreted as a 1D
        # coordinate variable; explicit `NamedTuple` entries are taken as-is.
        if entry isa NamedTuple
            arr = entry.array
            var_dims = entry.dims
        else
            arr = entry
            var_dims = (var_name,)
        end
        arr isa Nothing && continue

        # Convert to the requested float type and collect to a plain CPU array
        arr = collect(dimension_type.(arr))

        # Ensure each NetCDF dimension referenced by this variable exists.
        for (axis, dname) in enumerate(var_dims)
            if dname ∉ keys(dataset.dim)
                defDim(dataset, dname, size(arr, axis))
            end
        end

        if var_name ∉ keys(dataset)
            defVar(dataset, var_name, arr, var_dims,
                   attrib=get(attributes_dict, var_name, Dict{String, Any}()); kwargs...)
        else
            # The variable already exists in the dataset. Validate that the existing values
            # match what we'd write — applies equally to 1D coordinate variables (a NetCDF
            # "coordinate variable", same name as its dimension) and to 2D auxiliary
            # coordinates such as λ_cca/φ_cca on an OrthogonalSphericalShellGrid. Without
            # this, an inconsistent reused dataset could pass silently.
            existing_array = collect(dataset[var_name])
            if existing_array != collect(arr)
                throw(ArgumentError("Variable '$var_name' already exists in dataset but its values differ from expected.\n" *
                                    "  Actual:   $(existing_array) (size=$(size(existing_array)))\n" *
                                    "  Expected: $(arr) (size=$(size(arr)))"))
            end
        end

        # Effective dim names list: track NetCDF dimensions consumed (deduped)
        for dname in var_dims
            dname ∈ effective_dim_names || push!(effective_dim_names, dname)
        end
    end
    return tuple(effective_dim_names...)
end
