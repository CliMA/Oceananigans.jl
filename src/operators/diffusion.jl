""" Calculates δx_caa(Ax² * κ * δx_faa(f)). For LES, κ should be κ_fcc(i+1, j, k) and κ_fcc(i, j, k). """
@inline function δx²Aκ_caa(i, j, k, grid::Grid, Q::AbstractArray, κ::AbstractFloat)
    Ax(i+1, j, k, grid)^2 * κ * δx_faa(i+1, j, k, Q) -
    Ax(i,   j, k, grid)^2 * κ * δx_faa(i,   j, k, Q)
end

""" Calculates δy_aca(Ay² * κ * δy_afa(f)). For LES, κ should be κ_cfc(i, j+1, k) and κ_cfc(i, j, k). """
@inline function δy²Aκ_aca(i, j, k, grid::Grid, Q::AbstractArray, κ::AbstractFloat)
    Ay(i, j+1, k, grid)^2 * κ * δy_afa(i, j+1, k, Q) -
    Ay(i,   j, k, grid)^2 * κ * δy_afa(i,   j, k, Q)
end

""" Calculates δz_aac(Az² * κ * δz_aaf(f)). For LES, κ should be κ_ccf(i, j, k+1) and κ_ccf(i, j, k). """
@inline function δz²Aκ_aac(i, j, k, grid::Grid, Q::AbstractArray, κ::AbstractFloat)
    if k == grid.Nz
        Az(i, j, k, grid)^2 * κ * δz_aaf(i, j, k, Q)
    else
        Az(i, j,   k, grid)^2 * κ * δz_aaf(i, j,   k, Q) -
        Az(i, j, k+1, grid)^2 * κ * δz_aaf(i, j, k+1, Q)
    end
end

"""
Calculates diffusion for a tracer Q via

    V⁻² * [δx_caa(Ax² * κ * δx_faa(Q)) + δy_aca(Ay² * κ * δy_afa(Q)) + δz_aac(Az² * κ * δz_aaf(Q))]

which will end up at the location `ccc`.
"""
@inline function ∇κ∇Q(i, j, k, grid::Grid, Q::AbstractArray, κ::AbstractFloat)
    V⁻¹(i, j, k, grid)^2 * (δx²Aκ_caa(i, j, k, grid, Q, κ) + δy²Aκ_aca(i, j, k, grid, Q, κ) + δz²Aκ_aac(i, j, k, grid, Q, κ))
end
