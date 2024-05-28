"""
    ZeroGradient

Zero gradient perepndicular velocity boundary condition. This is compatible 
with the pressure solve (???) and implies that there is a pressure gradient
on the boundary equal to the pressure gradient at the boundary adjacent face
node.

```math
u_b^\\star = u_{b-1}^\\star,\\\\
u_b^{n+1} = u_{b-1}^{n+1} = u_{b-1}^\\star - \\Delta t\\partial_x P^{fcc}_{b-1} 
= u_{b}^\\star - \\Delta t\\partial_x P^{fcc}_{b},\\\\
\\to \\partial_x P^{fcc}_{b} = \\partial_x P^{fcc}_{b-1} \\therefore \\partial_x^2 P^{ccc}_{b-1} = 0.
```
"""
struct ZeroGradient end

const ZGOBC = BoundaryCondition{<:Open{<:ZeroGradient}}

function ZeroGradientOpenBoundaryCondition()
    classifcation = Open(ZeroGradient())
    
    return BoundaryCondition(classifcation, nothing)
end

@inline function _fill_west_halo!(j, k, grid, c, bc::ZGOBC, loc, clock, model_fields)
    @inbounds c[0, j, k] = c[1, j, k]
end

@inline function _fill_east_halo!(j, k, grid, c, bc::ZGOBC, loc, clock, model_fields)
    i = grid.Nx + 1

    @inbounds c[i, j, k] =  c[i - 1, j, k] #2 * c[i - 1, j, k] - c[i - 2, j, k]
end

@inline function _fill_south_halo!(i, k, grid, c, bc::ZGOBC, loc, clock, model_fields)
    @inbounds c[i, 0, k] = c[i, 1, k]
end

@inline function _fill_north_halo!(i, k, grid, c, bc::ZGOBC, loc, clock, model_fields)
    j = grid.Ny + 1

    @inbounds c[i, j, k] = c[i, j - 1, k]
end

@inline function _fill_bottom_halo!(i, j, grid, c, bc::ZGOBC, loc, clock, model_fields)
    @inbounds c[i, j, 0] = c[i, j, 1]
end

@inline function _fill_top_halo!(i, j, grid, c, bc::ZGOBC, loc, clock, model_fields)
    k = grid.Nz + 1

    @inbounds c[i, j, k] = c[i, j, k - 1]
end