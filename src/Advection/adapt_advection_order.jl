using Oceananigans.Grids: topology

"""
    adapt_advection_order(advection, grid::AbstractGrid)

Adapts the advection operator `advection` based on the grid `grid` by adjusting the order of advection in each direction.
For example, if the grid has only one point in the x-direction, the advection operator in the x-direction is set to first order
upwind or 2nd order centered scheme, depending on the original user-specified advection scheme. A high order advection sheme 
is reduced to a lower order advection scheme if the grid has fewer points in that direction.

# Arguments
- `advection`: The original advection scheme.
- `grid::AbstractGrid`: The grid on which the advection scheme is applied.

If the order of advection is changed in at least one direction, the adapted advection scheme with adjusted advection order returned 
by this function is a `FluxFormAdvection`.
"""
function adapt_advection_order(advection, grid::AbstractGrid)
    advection_x = adapt_advection_order(advection, size(grid, 1), grid)
    advection_y = adapt_advection_order(advection, size(grid, 2), grid)
    advection_z = adapt_advection_order(advection, size(grid, 3), grid)

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
    advection_x = adapt_advection_order(advection.x, size(grid, 1), grid)
    advection_y = adapt_advection_order(advection.y, size(grid, 2), grid)
    advection_z = adapt_advection_order(advection.z, size(grid, 3), grid)

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
adapt_advection_order(advection::Nothing, grid::AbstractGrid) = nothing
adapt_advection_order(advection::Nothing, N::Int, grid::AbstractGrid) = nothing

#####
##### Directional adapt advection order
#####

function adapt_advection_order(advection::Centered{H}, N::Int, grid::AbstractGrid) where H
    if N >= H
        return advection
    else
        return Centered(; order = N * 2)
    end
end

function adapt_advection_order(advection::UpwindBiased{H}, N::Int, grid::AbstractGrid) where H
    if N >= H
        return advection
    else
        return UpwindBiased(; order = N * 2 - 1)
    end
end

"""
    new_weno_scheme(grid, order, bounds, XT, YT, ZT)

Constructs a new WENO scheme based on the given parameters. `XT`, `YT`, and `ZT` is the type of the precomputed weno coefficients in the 
x-direction, y-direction and z-direction. A _non-stretched_ WENO scheme has `T` equal to `Nothing` everywhere. In case of a non-stretched WENO scheme, 
we rebuild the advection without passing the grid information, otherwise we use the grid to account for stretched directions.
"""
new_weno_scheme(::WENO, grid, order, bounds, ::Type{Nothing}, ::Type{Nothing}, ::Type{Nothing},) = WENO(; order, bounds)
new_weno_scheme(::WENO, grid, order, bounds, XT, YT, ZT)                                         = WENO(grid; order, bounds)

function adapt_advection_order(advection::WENO{H, FT, XT, YT, ZT}, N::Int, grid::AbstractGrid) where {H, FT, XT, YT, ZT}
    if N >= H
        return advection
    else
        return new_weno_scheme(advection, grid, N * 2 - 1, advection.bounds, XT, YT, ZT)
    end
end