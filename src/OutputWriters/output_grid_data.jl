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

"""
    gather_immersed_boundary(grid, indices, dim_name_generator)

Gather immersed-boundary data for output and grid reconstruction.
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

    ib_vars = Dict("bottom_height" => Field(bottom_height_field(grid); indices),
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
