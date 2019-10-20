""" Calculates δx_caa(Ax * δx_faa(f)) -> fcc. """
@inline δx²A_caa(i, j, k, grid::Grid, f::AbstractArray) =
    Ax(i+1, j, k, grid) * δx_faa(i+1, j, k, grid, f) -
    Ax(i,   j, k, grid) * δx_faa(i,   j, k, grid, f)

""" Calculates δy_aca(Ay * δy_afa(f)) -> cfc. """
@inline δy²A_aca(i, j, k, grid::Grid, f::AbstractArray) =
    Ay(i, j+1, k, grid) * δy_afa(i, j+1, k, grid, f) -
    Ay(i,   j, k, grid) * δy_afa(i,   j, k, grid, f)

""" Calculates δz_aac(Az * δz_aaf(f)) -> ccf. """
@inline function δz²A_aac(i, j, k, grid::Grid, f::AbstractArray)
    if k == grid.Nz
        return Az(i, j, k, grid) * δz_aaf(i, j, k, grid, f)
    else
        return Az(i, j,   k, grid) * δz_aaf(i, j,   k, grid, f) -
               Az(i, j, k+1, grid) * δz_aaf(i, j, k+1, grid, f)
    end
end

"""
    ∇²(i, j, k, grid::Grid, f::AbstractArray)

Calculates the Laplacian of f via

    1/V * [δx_caa(Ax * δx_faa(f)) + δy_aca(Ay * δy_afa(f)) + δz_aac(Az * δz_aaf(f))]

which will end up at the location `ccc`.
"""
@inline function ∇²(i, j, k, grid::Grid, f::AbstractArray)
    1/V(i, j, k, grid) * (δx²A_caa(i, j, k, grid, f) +
                          δy²A_aca(i, j, k, grid, f) +
                          δz²A_aac(i, j, k, grid, f))
end
