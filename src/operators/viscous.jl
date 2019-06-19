""" Calculates δx_faa(Ax * ν * ϊx_caa(Ax) * δx_caa(u)). For LES, ν should be ν_cff(i, j, k) and ν_cff(i-1, j, k). """
@inline function δx²Aν_faa(i, j, k, grid::Grid, u::AbstractArray, ν::AbstractFloat)
    Ax(i,   j, k, grid) * ν * ϊAx_caa(i,   j, k, grid) * δx_caa(i,   j, k, grid, u) -
    Ax(i-1, j, k, grid) * ν * ϊAx_caa(i-1, j, k, grid) * δx_caa(i-1, j, k, grid, u)
end

""" Calculates δy_aca(Ay * ν * ϊy_afa(Ay) * δy_afa(u)). For LES, ν should be ν_cff(i, j, k) and ν_cff(i, j+1, k). """
@inline function δy²Aν_aca(i, j, k, grid::Grid, u::AbstractArray, ν::AbstractFloat)
    Ay(i, j+1, k, grid) * ν * ϊAy_afa(i, j+1, k, grid) * δy_afa(i, j+1, k, grid, u) -
    Ay(i,   j, k, grid) * ν * ϊAy_afa(i,   j, k, grid) * δy_afa(i,   j, k, grid, u)
end

""" Calculates δz_aac(Az * ν * ϊz_aaf(Az) * δz_aaf(u)). For LES, ν should be ν_cff(i, j, k) and ν_cff(i, j, k+1). """
@inline function δz²Aν_aac(i, j, k, grid::Grid, u::AbstractArray, ν::AbstractFloat)
    if k == grid.Nz
        return Az(i, j, k, grid) * ν * ϊAz_aaf(i, j, k, grid) * δz_aaf(i, j, k, grid, u)
    else
        return Az(i, j,   k, grid) * ν * ϊAz_aaf(i, j,   k, grid) * δz_aaf(i, j,   k, grid, u) -
               Az(i, j, k+1, grid) * ν * ϊAz_aaf(i, j, k+1, grid) * δz_aaf(i, j, k+1, grid, u)
    end
end

"""
Calculates viscous dissipation for the u-velocity via

    (V * Vᵘ)⁻¹ * [δx_faa(Ax * ν * ϊx_caa(Ax) * δx_caa(u)) + δy_aca(Ay * ν * ϊy_afa(Ay) * δy_afa(u)) + δz_aac(Az * ν * ϊz_aaf(Az) * δz_aaf(u))]

which will end up at the location `fcc`.
"""
@inline function ∇ν∇u(i, j, k, grid::Grid, u::AbstractArray, ν::AbstractFloat)
    V⁻¹(i, j, k, grid) * ϊx_V⁻¹(i, j, k, grid) * (δx²Aν_faa(i, j, k, grid, u, ν) + δy²Aν_aca(i, j, k, grid, u, ν) + δz²Aν_aac(i, j, k, grid, u, ν))
end

""" Calculates δx_caa(Ax * ν * ϊx_faa(Ax) * δx_faa(u)). For LES, ν should be ν_fcf(i, j, k) and ν_fcf(i+1, j, k). """
@inline function δx²Aν_caa(i, j, k, grid::Grid, u::AbstractArray, ν::AbstractFloat)
    Ax(i+1, j, k, grid) * ν * ϊAx_faa(i+1, j, k, grid) * δx_faa(i+1, j, k, grid, u) -
    Ax(i,   j, k, grid) * ν * ϊAx_faa(i,   j, k, grid) * δx_faa(i,   j, k, grid, u)
end

""" Calculates δy_afa(Ay * ν * ϊx_aca(Ay) * δy_afa(u)). For LES, ν should be ν_fcf(i, j, k) and ν_fcf(i-1, j, k). """
@inline function δy²Aν_afa(i, j, k, grid::Grid, u::AbstractArray, ν::AbstractFloat)
    Ay(i,   j, k, grid) * ν * ϊAy_aca(i,   j, k, grid) * δy_aca(i,   j, k, grid, u) -
    Ay(i, j-1, k, grid) * ν * ϊAy_aca(i, j-1, k, grid) * δy_aca(i, j-1, k, grid, u)
end

"""
Calculates viscous dissipation for the v-velocity via

    (V * Vᵛ)⁻¹ * [δx_caa(Ax * ν * ϊx_faa(Ax) * δx_faa(v)) + δy_afa(Ay * ν * ϊy_aca(Ay) * δy_aca(v)) + δz_aac(Az * ν * ϊz_aaf(Az) * δz_aaf(v))]

which will end up at the location `cfc`.
"""
@inline function ∇ν∇v(i, j, k, grid::Grid, v::AbstractArray, ν::AbstractFloat)
    V⁻¹(i, j, k, grid) * ϊy_V⁻¹(i, j, k, grid) * (δx²Aν_caa(i, j, k, grid, v, ν) + δy²Aν_afa(i, j, k, grid, v, ν) + δz²Aν_aac(i, j, k, grid, v, ν))
end

""" Calculates δz_aaf(Az * ν * ϊz_aac(Az) * δz_aac(u)).  """
@inline function δz²Aν_aaf(i, j, k, grid::Grid{T}, u::AbstractArray, ν::AbstractFloat) where T
    if k == 1
        return -zero(T)
    else
        return Az(i, j, k-1, grid) * ν * ϊAz_aac(i, j, k-1, grid) * δz_aac(i, j, k-1, grid, u) -
               Az(i, j,   k, grid) * ν * ϊAz_aac(i, j,   k, grid) * δz_aac(i, j,   k, grid, u)
    end
end

"""
Calculates viscous dissipation for the w-velocity via

    (V * Vʷ)⁻¹ * [δx_caa(Ax * ν * ϊx_faa(Ax) * δx_faa(w)) + δy_aca(Ay * ν * ϊy_afa(Ay) * δy_afa(w)) + δz_aaf(Az * ν * ϊz_aac(Az) * δz_aac(w))]

which will end up at the location `ccf`.
"""
@inline function ∇ν∇w(i, j, k, grid::Grid, w::AbstractArray, ν::AbstractFloat)
    V⁻¹(i, j, k, grid) * ϊz_V⁻¹(i, j, k, grid) * (δx²Aν_caa(i, j, k, grid, v, ν) + δy²Aν_aca(i, j, k, grid, v, ν) + δz²Aν_aaf(i, j, k, grid, v, ν))
end
