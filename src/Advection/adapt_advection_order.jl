

function adapt_advection_order_x(advection::Centered{H}, grid::AbstractGrid, N::Int) where H
    if N == 1
        return nothing
    elseif N >= H
        return advection
    else
        return Centered(; order = N)
    end
end

function adapt_advection_order_x(advection::UpwindBiased{H}, grid::AbstractGrid, N::Int) where H
    if N == 1
        return nothing
    elseif N >= H
        return advection
    else
        return UpwindBiased(; order = N)
    end
end

adapt_advection_order_y(advection, grid::AbstractGrid, N::Int) = adapt_advection_order_x(advection, grid, N)
adapt_advection_order_z(advection, grid::AbstractGrid, N::Int) = adapt_advection_order_x(advection, grid, N)

"""
    new_weno_scheme(grid, order, bounds, T)

Constructs a new WENO scheme based on the given parameters. `T` is the type of the weno coefficients. 
A _non-stretched_ WENO scheme has `T` equal to `Nothing`. In case of a non-stretched WENO scheme, 
we rebuild the advection without passing the grid information, otherwise we use the grid to account for stretched directions.
"""
new_weno_scheme(grid, order, bounds, T) = ifelse(T == Nothing, WENO(; order, bounds), WENO(grid; order, bounds))

function adapt_advection_order_x(advection::WENO{H, FT, XT, YT, ZT}, grid::AbstractGrid) where {H, FT, XT, YT, ZT}
    
    if N == 1
        return nothing
    elseif N >= H
        return advection
    else
        return new_weno_scheme(grid, N, advection.bounds, XT)
    end
end

function adapt_advection_order_y(advection::WENO{H, FT, XT, YT, ZT}, grid::AbstractGrid, N::Int) where {H, FT, XT, YT, ZT}
    
    if N > H
        return advection
    else
        return new_weno_scheme(grid, N, advection.bounds, YT)
    end
end

function adapt_advection_order_z(advection::WENO{H, FT, XT, YT, ZT}, grid::AbstractGrid, N::Int) where {H, FT, XT, YT, ZT}
    
    if N == 1
        return nothing
    elseif N >= H
        return advection
    else
        return new_weno_scheme(grid, N, advection.bounds, XT)
    end
end

"""
    adapt_advection_order(advection, grid::AbstractGrid)

Adapts the advection operator `advection` based on the grid `grid` by adjusting the order of advection in each direction.
For example, if the grid has only one point in the x-direction, the advection operator in the x-direction is set to `nothing`.
A high order advection sheme is reduced to a lower order advection scheme if the grid has fewer points in that direction.

# Arguments
- `advection`: The original advection scheme.
- `grid::AbstractGrid`: The grid on which the advection scheme is applied.

The adapted advection scheme with adjusted advection order returned by this function is a `FluxFormAdvection`.
"""
function adapt_advection_order(advection, grid::AbstractGrid)
    advection_x = adapt_advection_order_x(advection, grid, grid.Nx)
    advection_y = adapt_advection_order_y(advection, grid, grid.Ny)
    advection_z = adapt_advection_order_z(advection, grid, grid.Nz)

    # Check that we indeed changed the advection operator
    changed_x = advection_x != advection
    changed_y = advection_y != advection
    changed_z = advection_z != advection

    new_advection = FluxFormAdvection(advection_x, advection_y, advection_z)
    changed_advection = any((changed_x, changed_y, changed_z))

    if changed_advection
        @info "User-defined advection scheme $(advection) reduced to $(new_advection) to comply with grid-size limitations."
    end

    return ifelse(changed_advection, new_advection, advection)
end

function adapt_advection_order(advection::FluxFormAdvection, grid::AbstractGrid)
    advection_x = adapt_advection_order_x(advection.x, grid, grid.Nx)
    advection_y = adapt_advection_order_y(advection.y, grid, grid.Ny)
    advection_z = adapt_advection_order_z(advection.z, grid, grid.Nz)

    # Check that we indeed changed the advection operator
    changed_x = advection_x != advection.x
    changed_y = advection_y != advection.y
    changed_z = advection_z != advection.z

    new_advection = FluxFormAdvection(advection_x, advection_y, advection_z)
    changed_advection = any((changed_x, changed_y, changed_z))

    if changed_advection
        @info "User-defined advection scheme $(advection) reduced to $(new_advection) to comply with grid-size limitations."
    end

    return ifelse(changed_advection, new_advection, advection)
end

# For the moment, we do not adapt the advection order for the VectorInvariant advection scheme
adapt_advection_order(advection::VectorInvariant, grid::AbstractGrid) = advection