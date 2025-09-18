module DimensionalityUtils

using Oceananigans
using Oceananigans.Grids: OrthogonalSphericalShellGrid
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

export deduce_dimensionality, drop_singleton_indices, axis_str, LLGOrIBLLG

function drop_singleton_indices(N)
    N == 1 ? 1 : Colon()
end

"""
    deduce_dimensionality(f)

Deduce the dimensionality of the field `f` and return a 3-tuple `d1, d2, D`, where `d1` is the first dimension along
which `f` varies, `d2` is the second dimension (if any), and `D` is the total dimensionality of `f`.
"""
function deduce_dimensionality(f)
    # Find indices of the dimensions along which `f` varies.
    d1 = findfirst(n -> n > 1, size(f))
    d2 =  findlast(n -> n > 1, size(f))
    
    # Deduce total dimensionality.
    Nx, Ny, Nz = size(f)
    D = (Nx > 1) + (Ny > 1) + (Nz > 1)

    return d1, d2, D
end

axis_str(::RectilinearGrid, dim) = ("x", "y", "z")[dim]
axis_str(::LatitudeLongitudeGrid, dim) = ("Longitude (deg)", "Latitude (deg)", "z")[dim]
axis_str(::OrthogonalSphericalShellGrid, dim) = ""
axis_str(grid::ImmersedBoundaryGrid, dim) = axis_str(grid.underlying_grid, dim)

const LLGOrIBLLG = Union{LatitudeLongitudeGrid, ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:LatitudeLongitudeGrid}}

end # module DimensionalityUtils
