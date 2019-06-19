""" Calculates δx_faa(ϊx_caa(Ax * u) * ϊx_caa(u)). """
@inline function δx_faa_Aūˣūˣ(i, j, k, grid::Grid, u::AbstractArray)
    ϊxAx_caa(i,   j, k, grid, u) * ϊx_caa(i,   j, k, grid, u) -
    ϊxAx_caa(i-1, j, k, grid, u) * ϊx_caa(i-1, j, k, grid, u)
end

""" Calculates δy_fca(ϊx_faa(Ay * v) * ϊy_afa(u)). """
@inline function δy_fca_Av̄ˣūʸ(i, j, k, grid::Grid, u::AbstractArray, v::AbstractArray)
    ϊxAy_faa(i, j+1, k, grid, v) * ϊy_afa(i, j+1, k, grid, u) -
    ϊxAy_faa(i,   j, k, grid, v) * ϊy_afa(i,   j, k, grid, u)
end

""" Calculates δz_fac(ϊx_faa(Az * w) * ϊz_aaf(u)). """
@inline function δz_fac_Aw̄ˣūᶻ(i, j, k, grid::Grid, u::AbstractArray, w::AbstractArray)
    if k == grid.Nz
        @inbounds return ϊxAz_faa(i, j, k, grid, w) * ϊz_aaf(i, j, k, grid, u)
    else
        @inbounds return ϊxAz_faa(i, j,   k, grid, w) * ϊz_aaf(i, j,   k, grid, u) -
                         ϊxAz_faa(i, j, k+1, grid, w) * ϊz_aaf(i, j, k+1, grid, u)
    end
end

"""
Calculates the advection of momentum in the x-direction V·∇u with a velocity field V = (u, v, w) via

    (v̅ˣ)⁻¹ * [δx_faa(ϊx_caa(Ax * u) * ϊx_caa(u)) + δy_fca(ϊx_faa(Ay * v) * ϊy_afa(u)) + δz_fac(ϊx_faa(Az * w) * ϊz_aaf(u))]

which will end up at the location `fcc`.
"""
@inline function u∇u(i, j, k, grid::Grid, u::AbstractArray, v::AbstractArray, w::AbstractArray)
    ϊx_V⁻¹(i, j, k, grid) * (δx_faa_Aūˣūˣ(i, j, k, grid, u) + δy_fca_Av̄ˣūʸ(i, j, k, grid, u, v) + δz_fac_Aw̄ˣūᶻ(i, j, k, grid, u, w))
end

""" Calculates δx_cfa(ϊy_afa(Ax * u) * ϊx_faa(v)). """
@inline function δx_cfa_Aūʸv̄ˣ(i, j, k, grid::RegularCartesianGrid, u::AbstractArray, v::AbstractArray)
    ϊyAx_afa(i+1, j, k, grid, u) * ϊx_faa(i+1, j, k, grid, v) -
    ϊyAx_afa(i,   j, k, grid, u) * ϊx_faa(i,   j, k, grid, v)
end

""" Calculates δy_afa(ϊy_aca(Ay * v) * ϊy_aca(v)). """
@inline function δy_afa_Av̄ʸv̄ʸ(i, j, k, grid::Grid, v::AbstractArray)
    ϊyAy_aca(i,   j, k, grid, v) * ϊy_aca(i,   j, k, grid, v) -
    ϊyAy_aca(i, j-1, k, grid, v) * ϊy_aca(i, j-1, k, grid, v)
end

""" Calculates δz_afc(ϊx_faa(Az * w) * ϊz_aaf(w)) """
@inline function δz_afc_Aw̄ʸv̄ᶻ(i, j, k, grid::Grid, v::AbstractArray, w::AbstractArray)
    if k == grid.Nz
        @inbounds return ϊyAz_afa(i, j, k, grid, w) * ϊz_aaf(i, j, k, grid, v)
    else
        @inbounds return ϊyAz_afa(i, j,   k, grid, w) * ϊz_aaf(i, j,   k, grid, v) -
                         ϊyAz_afa(i, j, k+1, grid, w) * ϊz_aaf(i, j, k+1, grid, v)
    end
end

"""
Calculates the advection of momentum in the y-direction V·∇v with a velocity field V = (u, v, w) via

    (v̅ʸ)⁻¹ * [δx_cfa(ϊy_afa(Ax * u) * ϊx_faa(v)) + δy_afa(ϊy_aca(Ay * v) * ϊy_aca(v)) + δz_afc(ϊx_faa(Az * w) * ϊz_aaf(w))]

which will end up at the location `cfc`.
"""
@inline function u∇v(i, j, k, grid::Grid, u::AbstractArray, v::AbstractArray, w::AbstractArray)
    ϊy_V⁻¹(i, j, k, grid) * (δx_cfa_Aūʸv̄ˣ(i, j, k, grid, u, v) + δy_afa_Av̄ʸv̄ʸ(i, j, k, grid, v) + δz_afc_Aw̄ʸv̄ᶻ(i, j, k, grid, v, w))
end

""" Calculates δx_caf(ϊz_aaf(Ax * u) * ϊx_faa(w)). """
@inline function δx_caf_Aūᶻw̄ˣ(i, j, k, grid::Grid, u::AbstractArray, w::AbstractArray)
    ϊzAx_aaf(i+1, j, k, grid, u) * ϊx_faa(i+1, j, k, grid, w) -
    ϊzAx_aaf(i,   j, k, grid, u) * ϊx_faa(i,   j, k, grid, w)
end

""" Calculates δy_acf(ϊz_aaf(Ay * v) * ϊy_afa(w)). """
@inline function δy_acf_Av̄ᶻw̄ʸ(i, j, k, grid::Grid, v::AbstractArray, w::AbstractArray)
    ϊzAy_aaf(i, j+1, k, grid, v) * ϊy_afa(i, j+1, k, grid, w) -
    ϊzAy_aaf(i, j,   k, grid, v) * ϊy_afa(i, j,   k, grid, w)
end

""" Calculates δz_aaf(ϊz_aac(Az * w) * ϊz_aac(w)). """
@inline function δz_aaf_Aw̄ᶻw̄ᶻ(i, j, k, grid::Grid{T}, w::AbstractArray) where T
    if k == 1
        return -zero(T)
    else
        return ϊzAz_aac(i, j, k-1, grid, w) * ϊz_aac(i, j, k-1, grid, w) - ϊzAz_aac(i, j, k, grid, w) * ϊz_aac(i, j, k, grid, w)
    end
end

"""
Calculates the advection of momentum in the z-direction V·∇w with a velocity field V = (u, v, w) via

    (v̅ᶻ)⁻¹ * [δx_caf(ϊz_aaf(Ax * u) * ϊx_faa(w)) + δy_acf(ϊz_aaf(Ay * v) * ϊy_afa(w)) + δz_aaf(ϊz_aac(Az * w) * ϊz_aac(w))]

which will end up at the location `ccf`.
"""
@inline function u∇w(i, j, k, grid::Grid, u::AbstractArray, v::AbstractArray, w::AbstractArray)
    ϊz_V⁻¹(i, j, k, grid) * (δx_caf_Aūᶻw̄ˣ(i, j, k, grid, u, w) + δy_acf_Av̄ᶻw̄ʸ(i, j, k, grid, v, w) + δz_aaf_Aw̄ᶻw̄ᶻ(i, j, k, grid, w))
end
