""" Calculates δx_caa(Ax²/Vᵘ * δx_faa(f)) """
@inline function δx²A_caa(i, j, k, grid::Grid, f::AbstractArray)
    Ax(i+1, j, k, grid)^2 * ϊx_V⁻¹(i+1, j, k, grid) * δx_faa(i+1, j, k, grid, f) -
    Ax(i,   j, k, grid)^2 * ϊx_V⁻¹(i,   j, k, grid) * δx_faa(i,   j, k, grid, f)
end

""" Calculates δy_aca(Ay²/Vᵛ * δy_afa(f)) """
@inline function δy²A_aca(i, j, k, grid::Grid, f::AbstractArray)
    Ay(i, j+1, k, grid)^2 * ϊy_V⁻¹(i, j+1, k, grid) * δy_afa(i, j+1, k, grid, f) -
    Ay(i,   j, k, grid)^2 * ϊy_V⁻¹(i,   j, k, grid) * δy_afa(i,   j, k, grid, f)
end

""" Calculates δz_aac(Az²/Vʷ * δz_aaf(f)) """
@inline function δz²A_aac(i, j, k, grid::Grid, f::AbstractArray)
    if k == grid.Nz
        return Az(i, j, k, grid)^2 * ϊz_V⁻¹(i, j, k, grid) * δz_aaf(i, j, k, grid, f)
    else
        return Az(i, j,   k, grid)^2 * ϊz_V⁻¹(i, j,   k, grid) * δz_aaf(i, j,   k, grid, f) -
               Az(i, j, k+1, grid)^2 * ϊz_V⁻¹(i, j, k+1, grid) * δz_aaf(i, j, k+1, grid, f)
    end
end

"""
Calculates the Laplacian of f via

    V⁻¹ * [δx_caa(Ax²/Vᵘ * δx_faa(f)) + δy_aca(Ay²/Vᵛ * δy_afa(f)) + δz_aac(Az²/Vʷ * δz_aaf(f))]

which will end up at the location `ccc`.
"""
@inline function ∇²_ppn(i, j, k, grid::Grid, f::AbstractArray)
    V⁻¹(i, j, k, grid) * (δx²A_caa(i, j, k, grid, f) + δy²A_aca(i, j, k, grid, f) + δz²A_aac(i, j, k, grid, f))
end
