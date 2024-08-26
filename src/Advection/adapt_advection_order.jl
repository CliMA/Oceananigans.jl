using Oceananigans.Grids: topology

"""
    adapt_advection_order(advection, grid::AbstractGrid)

Adapts the advection operator `advection` based on the grid `grid` by adjusting the order of advection in each direction.
For example, if the grid has only one point in the x-direction, the advection operator in the x-direction is set to `nothing`.
A high order advection sheme is reduced to a lower order advection scheme if the grid has fewer points in that direction.

# Arguments
- `advection`: The original advection scheme.
- `grid::AbstractGrid`: The grid on which the advection scheme is applied.

If the order of advection is changed in at least one direction, the adapted advection scheme with adjusted advection order returned 
by this function is a `FluxFormAdvection`.
"""
function adapt_advection_order(advection, grid::AbstractGrid)
    advection_x = adapt_advection_order_x(advection.x, topology(grid, 1), size(grid, 1), grid)
    advection_y = adapt_advection_order_y(advection.y, topology(grid, 2), size(grid, 2), grid)
    advection_z = adapt_advection_order_z(advection.z, topology(grid, 3), size(grid, 3), grid)

    # Check that we indeed changed the advection operator
    changed_x = advection_x != advection
    changed_y = advection_y != advection
    changed_z = advection_z != advection

    new_advection = FluxFormAdvection(advection_x, advection_y, advection_z)
    changed_advection = any((changed_x, changed_y, changed_z))

    if changed_x
        @info "User-defined advection scheme $(summary(advection)) reduced to $(summary(new_advection.x)) in the x-direction to comply with grid-size limitations."
    end
    if changed_y
        @info "User-defined advection scheme $(summary(advection)) reduced to $(summary(new_advection.y)) in the y-direction to comply with grid-size limitations."
    end
    if changed_z
        @info "User-defined advection scheme $(summary(advection)) reduced to $(summary(new_advection.z)) in the z-direction to comply with grid-size limitations."
    end

    return ifelse(changed_advection, new_advection, advection)
end

function adapt_advection_order(advection::FluxFormAdvection, grid::AbstractGrid)
    advection_x = adapt_advection_order_x(advection.x, topology(grid, 1), size(grid, 1), grid)
    advection_y = adapt_advection_order_y(advection.y, topology(grid, 2), size(grid, 2), grid)
    advection_z = adapt_advection_order_z(advection.z, topology(grid, 3), size(grid, 3), grid)

    # Check that we indeed changed the advection operator
    changed_x = advection_x != advection.x
    changed_y = advection_y != advection.y
    changed_z = advection_z != advection.z

    new_advection = FluxFormAdvection(advection_x, advection_y, advection_z)
    changed_advection = any((changed_x, changed_y, changed_z))

    if changed_x
        @info "User-defined advection scheme $(summary(advection)) reduced to $(summary(new_advection.x)) in the x-direction to comply with grid-size limitations."
    end
    if changed_y
        @info "User-defined advection scheme $(summary(advection)) reduced to $(summary(new_advection.y)) in the y-direction to comply with grid-size limitations."
    end
    if changed_z
        @info "User-defined advection scheme $(summary(advection)) reduced to $(summary(new_advection.z)) in the z-direction to comply with grid-size limitations."
    end

    return ifelse(changed_advection, new_advection, advection)
end

# For the moment, we do not adapt the advection order for the VectorInvariant advection scheme
adapt_advection_order(advection::VectorInvariant, grid::AbstractGrid) = advection

# We only need one halo in bounded directions!
adapt_advection_order_x(advection, topo, N, grid) = advection
adapt_advection_order_y(advection, topo, N, grid) = advection
adapt_advection_order_z(advection, topo, N, grid) = advection

#####
##### Directional adapt advection order
#####

function adapt_advection_order_x(advection::Centered{H}, topology, N::Int, grid::AbstractGrid) where H
    if N == 1
        return nothing
    elseif N >= H
        return advection
    else
        return Centered(; order = N * 2)
    end
end

function adapt_advection_order_x(advection::UpwindBiased{H}, topology, grid::AbstractGrid, N::Int) where H
    if N == 1
        return nothing
    elseif N >= H
        return advection
    else
        return UpwindBiased(; order = N * 2 - 1)
    end
end

adapt_advection_order_y(advection, topology, grid::AbstractGrid, N::Int) = adapt_advection_order_x(advection, topology, grid, N)
adapt_advection_order_z(advection, topology, grid::AbstractGrid, N::Int) = adapt_advection_order_x(advection, topology, grid, N)

"""
    new_weno_scheme(grid, order, bounds, T)

Constructs a new WENO scheme based on the given parameters. `T` is the type of the weno coefficients. 
A _non-stretched_ WENO scheme has `T` equal to `Nothing`. In case of a non-stretched WENO scheme, 
we rebuild the advection without passing the grid information, otherwise we use the grid to account for stretched directions.
"""
new_weno_scheme(grid, order, bounds, T) = ifelse(T == Nothing, WENO(; order, bounds), WENO(grid; order, bounds))

function adapt_advection_order_x(advection::WENO{H, FT, XT, YT, ZT}, topology, grid::AbstractGrid, N::Int) where {H, FT, XT, YT, ZT}
    
    if N == 1
        return nothing
    elseif N >= H
        return advection
    else
        return new_weno_scheme(grid, N * 2 - 1, advection.bounds, XT)
    end
end

function adapt_advection_order_y(advection::WENO{H, FT, XT, YT, ZT}, topology, grid::AbstractGrid, N::Int) where {H, FT, XT, YT, ZT}
        
    if N == 1
        return nothing
    elseif N > H
        return advection
    else
        return new_weno_scheme(grid, N * 2 - 1, advection.bounds, YT)
    end
end

function adapt_advection_order_z(advection::WENO{H, FT, XT, YT, ZT}, topology, grid::AbstractGrid, N::Int) where {H, FT, XT, YT, ZT}
    
    if N == 1
        return nothing
    elseif N >= H
        return advection
    else
        return new_weno_scheme(grid, N * 2 - 1, advection.bounds, XT)
    end
end
