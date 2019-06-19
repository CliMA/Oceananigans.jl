"""
Calculates the horizontal divergence of the velocity (u, v) via

    V⁻¹ * [δx_caa(Ax * u) + δy_aca(Ay * v)]

which will end up at the location `cca`.
"""
@inline function ∇h_u(i, j, k, grid::Grid, u::AbstractArray, v::AbstractArray)
    V⁻¹(i, j, k, grid) * (δxA_caa(i, j, k, grid, u) + δyA_aca(i, j, k, grid, v))
end

"""
Calculates the divergence ∇·f of a vector field f = (fx, fy, fz) via the discrete operation

    V⁻¹ * [δx_caa(Ax * fx) + δx_aca(Ay * fy) + δz_aac(Az * fz)]

which will end up at the cell centers `ccc`.
"""
@inline function div_ccc(i, j, k, grid::Grid, fx::AbstractArray, fy::AbstractArray, fz::AbstractArray)
    V⁻¹(i, j, k, grid) * (δxA_caa(i, j, k, grid, fx) + δyA_aca(i, j, k, grid, fy) + δzA_aac(i, j, k, grid, fz))
end

""" Calculates δx_caa(Ax * a * ϊx_faa(b)). """
@inline function δxA_caa_ab̄ˣ(i, j, k, grid::Grid, a::AbstractArray, b::AbstractArray)
    @inbounds (Ax(i, j, k, grid) * a[i+1, j, k] * ϊx_faa(i+1, j, k, grid, b) -
               Ax(i, j, k, grid) * a[i,   j, k] * ϊx_faa(i,   j, k, grid, b))
end

""" Calculates δy_aca(Ay * a * ϊy_afa(b)). """
@inline function δyA_aca_ab̄ʸ(i, j, k, grid::Grid, a::AbstractArray, b::AbstractArray)
    @inbounds (Ay(i, j, k, grid) * a[i, j+1, k] * ϊy_afa(i, j+1, k, grid, b) -
               Ay(i, j, k, grid) * a[i,   j, k] * ϊy_afa(i, j,   k, grid, b))
end

""" Calculates δz_aac(Az * a * ϊz_aaf(b)). """
@inline function δzA_aac_ab̄ᶻ(i, j, k, grid::Grid, a::AbstractArray, b::AbstractArray)
    if k == grid.Nz
        @inbounds return Az(i, j, k, grid) * a[i, j, k] * ϊz_aaf(i, j, k, grid, b)
    else
        @inbounds return (Az(i, j, k, grid) * a[i, j,   k] * ϊz_aaf(i, j,   k, grid, b) -
                          Az(i, j, k, grid) * a[i, j, k+1] * ϊz_aaf(i, j, k+1, grid, b))
    end
end

"""
Calculates the divergence of a flux ∇·(VQ) with a velocity field V = (u, v, w) and scalar quantity Q via

    V⁻¹ * [δx_caa(Ax * u * ϊx_faa(Q)) + δy_aca(Ay * v * ϊy_afa(Q)) + δz_aac(Az * w * ϊz_aaf(Q))]

which will end up at the location `ccc`.
"""
@inline function div_flux(i, j, k, grid::Grid, u::AbstractArray, v::AbstractArray, w::AbstractArray, Q::AbstractArray)
    if k == 1
        @inbounds return V⁻¹(i, j, k, grid) * (δxA_caa_ab̄ˣ(i, j, k, grid, u, Q) + δyA_aca_ab̄ʸ(i, j, k, grid, v, Q) - Az(i, j, k, grid) * (w[i, j, 2] * ϊz_aaf(i, j, 2, grid, Q))
    else
        return V⁻¹(i, j, k, grid) * (δxA_caa_ab̄ˣ(i, j, k, grid, u, Q) + δyA_aca_ab̄ʸ(i, j, k, grid, v, Q) + δzA_aac_ab̄ᶻ(i, j, k, grid, w, Q))
    end
end
