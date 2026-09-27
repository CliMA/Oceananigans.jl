using Oceananigans
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Grids: znode

#####
##### Volume-weighted diagnostics for conservation tests
#####

"""
    volume_integral(c)

Return the volume integral `∫ c dV` of the field or operation `c` over the active cells of its grid, computed with
`Integral`. Under a `ZStarCoordinate` the cell volumes are the current ones, so this is `∫ σ c dV` in static volumes.
"""
volume_integral(c) = sum(Array(interior(Field(Integral(c)))))

@inline z_times_tracer(i, j, k, grid, c) = @inbounds znode(i, j, k, grid, Center(), Center(), Center()) * c[i, j, k]

"""
    center_of_mass_height(c)

Return the volume-weighted mean height `∫ z c dV / ∫ c dV` of the tracer `c`.
"""
function center_of_mass_height(c)
    zc = KernelFunctionOperation{Center, Center, Center}(z_times_tracer, c.grid, c)
    return volume_integral(zc) / volume_integral(c)
end
