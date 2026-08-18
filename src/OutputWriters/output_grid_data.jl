#####
##### Orthogonal spherical shell grid reconstruction
#####

function construct_ossg_halo_padded_array(stored_data, variable_name, FT, arch, lx, ly,
                                          topology_instances, grid_sizes, halo_sizes,
                                          file_has_halos)
    Nx, Ny, _ = grid_sizes
    TX, TY, _ = topology_instances

    three_dimensional_data = allocate_grid_data(FT, arch, (lx, ly, nothing),
                                                topology_instances, grid_sizes, halo_sizes)
    halo_array = OffsetArray(dropdims(parent(three_dimensional_data), dims=3),
                             three_dimensional_data.offsets[1:2]...)

    if file_has_halos
        destination = parent(halo_array)
        region = "halo-included"
    else
        i_range = interior_indices(lx, TX, Nx)
        j_range = interior_indices(ly, TY, Ny)
        destination = view(halo_array, i_range, j_range)
        region = "interior"
    end

    size(stored_data) == size(destination) || throw(ArgumentError(
        "Saved array '$variable_name' has size $(size(stored_data)) but expected $region size $(size(destination))."))

    stored_data = on_architecture(arch, FT.(stored_data))
    destination .= stored_data
    return halo_array
end

# Wrap 2D OSSG data in a Field so that the grid's boundary conditions fill its halos,
# including the north fold on TripolarGrid.
function halo_fill_2d_metric(old_data, grid, LX, LY)
    TX, TY, _ = topology(grid)
    Nx, Ny, _ = size(grid)
    new_field = Field{LX, LY, Center}(grid)
    Ni = Base.length(LX(), TX(), Nx)
    Nj = Base.length(LY(), TY(), Ny)

    # Use Center in z so the TripolarGrid fold dispatch adds the longitude wrap.
    for k in axes(new_field.data, 3)
        new_field.data[1:Ni, 1:Nj, k] .= old_data[1:Ni, 1:Nj]
    end

    fill_halo_regions!(new_field)

    # The UPivot redundancy substitution rewrites the j=Ny interior row. That is
    # correct for symmetric metrics but not longitude, so restore all interiors.
    for k in axes(new_field.data, 3)
        new_field.data[1:Ni, 1:Nj, k] .= old_data[1:Ni, 1:Nj]
    end

    # Every z level is identical; retain one while preserving the halo offsets.
    data = deepcopy(view(new_field.data, :, :, 1))
    return on_architecture(architecture(grid), data)
end

#####
##### Gathering of grid metrics
#####

"""
    gather_grid_metrics(grid, indices, dim_name_generator)

Gather grid metrics for output. These are not strictly necessary for grid reconstruction,
but give users direct access to the grid geometry in analysis tools.
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

# `rspacings` (rather than `zspacings`) gives the *reference* spacing, which is what the
# saved 1D vertical coordinate refers to for a `MutableVerticalDiscretization`, and which
# coincides with the physical spacing for a `StaticVerticalDiscretization`.
vertical_spacing_field(grid, lz, indices) = Field(rspacings(grid, lz); indices)

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
        Δφᵃᶠᵃ_name = dim_name_generator("Δφ", grid, nothing, f, nothing, Val(:y))
        Δφᵃᶜᵃ_name = dim_name_generator("Δφ", grid, nothing, c, nothing, Val(:y))

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

# OSSG horizontal metrics are written at the four Arakawa-C stagger locations.
# Vertical Δz/Δr is added by `add_vertical_metrics!`.
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

const PCBorGFBIBG = Union{GFBIBG, PCBIBG}

# `peripheral_node` and `inactive_node` masks, at the four locations of the C-grid variables.
function immersed_node_fields(grid, indices)
    node_fields = Dict()

    for (lx, ly, lz) in ((c, c, c), (f, c, c), (c, f, c), (c, c, f))
        LX, LY, LZ = map(typeof, (lx, ly, lz))
        letters = loc2letter(lx) * loc2letter(ly) * loc2letter(lz)

        peripheral_nodes = KernelFunctionOperation{LX, LY, LZ}(peripheral_node, grid, lx, ly, lz)
        inactive_nodes = KernelFunctionOperation{LX, LY, LZ}(inactive_node, grid, lx, ly, lz)

        node_fields["peripheral_nodes_" * letters] = Field(peripheral_nodes; indices)
        node_fields["inactive_nodes_" * letters] = Field(inactive_nodes; indices)
    end

    return node_fields
end

"""
    gather_immersed_boundary(grid, indices, dim_name_generator)

Gather immersed-boundary data for output and grid reconstruction.
"""
function gather_immersed_boundary(grid::PCBorGFBIBG, indices, dim_name_generator; grid_index=nothing)
    ib_vars = merge(Dict("bottom_height" => Field(bottom_height_field(grid); indices)),
                    immersed_node_fields(grid, indices))

    return suffix_grid_keys(ib_vars, grid_index)
end

const GFBoundaryIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:GridFittedBoundary}

function gather_immersed_boundary(grid::GFBoundaryIBG, indices, dim_name_generator; grid_index=nothing)
    ib_vars = merge(Dict("mask" => Field(grid.immersed_boundary.mask; indices)),
                    immersed_node_fields(grid, indices))

    return suffix_grid_keys(ib_vars, grid_index)
end
