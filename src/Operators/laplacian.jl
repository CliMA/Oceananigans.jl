"""
    ∇²(i, j, k, grid, c)

Calculates the Laplacian of c via

    1/V * [δx_caa(Ax * δx_faa(c)) + δy_aca(Ay * δy_afa(c)) + δz_aac(Az * δz_aaf(c))]

which will end up at the location `ccc`.
"""
@inline function ∇²(i, j, k, grid, c)
    1/VC(i, j, k, grid) * (δx_caa(i, j, k, grid, δFx_faa, c) +
                           δy_aca(i, j, k, grid, δFy_afa, c) +
                           δz_aac(i, j, k, grid, δFz_aaf, c))
end

