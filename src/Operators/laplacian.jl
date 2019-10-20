"""
    ∇²(i, j, k, grid, c)

Calculates the Laplacian of c via

    1/V * [δx_caa(Ax * δx_faa(c)) + δy_aca(Ay * δy_afa(c)) + δz_aac(Az * δz_aaf(c))]

which will end up at the location `ccc`.
"""
@inline function ∇²(i, j, k, grid, c)
    1/V(i, j, k, grid) * (δx_caa(i, j, k, grid, δxA_faa, c) +
                          δy_aca(i, j, k, grid, δyA_afa, c) +
                          δz_aac(i, j, k, grid, δzA_aaf, c))
end

