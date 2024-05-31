using Oceananigans.Operators: ∂xᶜᶜᶜ

"""
    ZeroGradient

Zero gradient perepndicular velocity boundary condition.
"""
struct ZeroGradient end

const ZGOBC = BoundaryCondition{<:Open{<:ZeroGradient}}

function ZeroGradientOpenBoundaryCondition()
    classifcation = Open(ZeroGradient())
    
    return BoundaryCondition(classifcation, nothing)
end

@inline function _fill_west_open_halo!(j, k, grid, c, bc::ZGOBC, loc, clock, model_fields)
    @inbounds c[0, j, k] = c[1, j, k]
end

@inline function _fill_east_open_halo!(j, k, grid, c, bc::ZGOBC, loc, clock, model_fields)
    i = grid.Nx + 1

    @inbounds c[i, j, k] =  c[i - 1, j, k]
end

@inline function _fill_south_open_halo!(i, k, grid, c, bc::ZGOBC, loc, clock, model_fields)
    @inbounds c[i, 0, k] = c[i, 1, k]
end

@inline function _fill_north_open_halo!(i, k, grid, c, bc::ZGOBC, loc, clock, model_fields)
    j = grid.Ny + 1

    @inbounds c[i, j, k] = c[i, j - 1, k]
end

@inline function _fill_bottom_open_halo!(i, j, grid, c, bc::ZGOBC, loc, clock, model_fields)
    @inbounds c[i, j, 0] = c[i, j, 1]
end

@inline function _fill_top_open_halo!(i, j, grid, c, bc::ZGOBC, loc, clock, model_fields)
    k = grid.Nz + 1

    @inbounds c[i, j, k] = c[i, j, k - 1]
end