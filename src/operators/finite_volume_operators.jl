using Oceananigans:
    Grid, RegularCartesianGrid, VerticallyStretchedCartesianGrid

@inline Δx(i, j, k, grid::RegularCartesianGrid) = grid.Δx
@inline Δx(i, j, k, grid::VerticallyStretchedCartesianGrid) = grid.Δx
@inline Δx(i, j, k, grid::Grid) = @inbounds grid.Δx[i, j, k]

@inline Δy(i, j, k, grid::RegularCartesianGrid) = grid.Δy
@inline Δy(i, j, k, grid::VerticallyStretchedCartesianGrid) = grid.Δy
@inline Δy(i, j, k, grid::Grid) = @inbounds grid.Δy[i, j, k]

@inline Δz(i, j, k, grid::RegularCartesianGrid) = grid.Δz
@inline Δz(i, j, k, grid::VerticallyStretchedCartesianGrid) = @inbounds grid.Δz[k]
@inline Δz(i, j, k, grid::Grid) = @inbounds grid.Δz[i, j, k]

@inline Ax(i, j, k, grid::Grid) = Δy(i, j, k, grid) * Δz(i, j, k, grid)
@inline Ay(i, j, k, grid::Grid) = Δx(i, j, k, grid) * Δz(i, j, k, grid)
@inline Az(i, j, k, grid::Grid) = Δx(i, j, k, grid) * Δy(i, j, k, grid)

@inline V(i, j, k, grid::Grid) = Δx(i, j, k, grid) * Δy(i, j, k, grid) * Δz(i, j, k, grid)
@inline V⁻¹(i, j, k, grid::Grid) = 1 / V(i, j, k, grid)

@inline ϊx_V(i, j, k, grid::Grid{T}) where T = T(0.5) * (V(i+1, j, k, grid) + V(i, j, k, grid))
@inline ϊy_V(i, j, k, grid::Grid{T}) where T = T(0.5) * (V(i, j+1, k, grid) + V(i, j, k, grid))
@inline ϊz_V(i, j, k, grid::Grid{T}) where T = T(0.5) * (V(i, j, k+1, grid) + V(i, j, k, grid))

@inline ϊx_V⁻¹(i, j, k, grid::Grid{T}) = 1 / ϊx_V(i, j, k, grid)
@inline ϊy_V⁻¹(i, j, k, grid::Grid{T}) = 1 / ϊy_V(i, j, k, grid)
@inline ϊz_V⁻¹(i, j, k, grid::Grid{T}) = 1 / ϊz_V(i, j, k, grid)

#=
Differentiation and interpolation operators for functions

The geometry of the staggerd grid used by Oceananigans (the Arakawa C-grid)
is (in one dimension) shown below

face   cell   face   cell   face

        i-1            i
         ↓             ↓
  |      ×      |      ×      |
  ↑             ↑             ↑
 i-1            i            i+1

Difference operators are denoted by a `δ` (`\delta`). Calculating the difference
of a cell-centered quantity ϕ at cell i will return the difference at face i

δϕᵢ = ϕᵢ - ϕᵢ₋₁

and so this operation, if applied along the x-dimension, is denoted by `δx_faa`.

The difference of a face-centered quantity u at face i will return the difference
at cell i

δuᵢ = uᵢ₊₁ - uᵢ

and is thus denoted `δx_caa` when applied along the x-dimension.

The three characters at the end of the function name, `faa` for example, indicates that
the output will lie on the cell faces in the x-dimension but will remain at their original
positions in the y- and z-dimensions. Thus we further identify this operator by `_faa`
where the `a` stands for any as the location is unchanged by the operator and is determined
by the input.

As a result the interpolation of a quantity ϕ from a cell i to face i
(this operation is denoted "ϊx_faa" in the code below) is

ϊx_faa(ϕ)ᵢ = (ϕᵢ + ϕᵢ₋₁) / 2

Conversely, the interpolation of a quantity u from a face i to cell i is given by

ϊx_caa(u)ᵢ = (uᵢ₊₁ + uᵢ) / 2

The `ϊ` (`\iota\ddot`) symbol indicates that an interpolation is being performed.
`ϊx`, for example, indicates that the interpolation is performed along the x-dimension.
The three following characters in the interpolation function name indicate the
position that is being interpolated to following the same convention used for difference
operators.
=#

@inline δx_caa(i, j, k, grid::Grid, f::AbstractArray) = @inbounds f[i+1, j, k] - f[i,   j, k]
@inline δx_faa(i, j, k, grid::Grid, f::AbstractArray) = @inbounds f[i,   j, k] - f[i-1, j, k]

@inline δy_aca(i, j, k, grid::Grid, f::AbstractArray) = @inbounds f[i, j+1, k] - f[i, j,   k]
@inline δy_afa(i, j, k, grid::Grid, f::AbstractArray) = @inbounds f[i, j,   k] - f[i, j-1, k]

@inline function δz_aac(i, j, k, g::Grid{T}, f::AbstractArray) where T
    if k == grid.Nz
        @inbounds return f[i, j, k]
    else
        @inbounds return f[i, j, k] - f[i, j, k+1]
    end
end

@inline function δz_aaf(i, j, k, g::Grid{T}, f::AbstractArray) where T
    if k == 1
        return -zero(T)
    else
        @inbounds return f[i, j, k-1] - f[i, j, k]
    end
end

@inline δxA_caa(i, j, k, grid::Grid, f::AbstractArray) = @inbounds Ax(i+1, j, k, grid) * f[i+1, j, k] - Ax(i,   j, k) * f[i,   j, k]
@inline δxA_faa(i, j, k, grid::Grid, f::AbstractArray) = @inbounds Ax(i,   j, k, grid) * f[i,   j, k] - Ax(i-1, j, k) * f[i-1, j, k]

@inline δyA_aca(i, j, k, grid::Grid, f::AbstractArray) = @inbounds Ay(i, j+1, k, grid) * f[i, j+1, k] - Ay(i, j,   k, grid) * f[i, j,   k]
@inline δyA_afa(i, j, k, grid::Grid, f::AbstractArray) = @inbounds Ay(i,   j, k, grid) * f[i, j,   k] - Ay(i, j-1, k, grid) * f[i, j-1, k]

@inline function δzA_aac(i, j, k, g::Grid{T}, f::AbstractArray) where T
    if k == grid.Nz
        @inbounds return Az(i, j, k, grid) * f[i, j, k]
    else
        @inbounds return Az(i, j, k, grid) * f[i, j, k] - Az(i, j, k+1, grid) * f[i, j, k+1]
    end
end

@inline function δzA_aaf(i, j, k, g::Grid{T}, f::AbstractArray) where T
    if k == 1
        return -zero(T)
    else
        @inbounds return Az(i, j, k-1, grid) * f[i, j, k-1] - Az(i, j, k, grid) * f[i, j, k]
    end
end

@inline ϊx_caa(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (f[i+1, j, k] + f[i,    j, k])
@inline ϊx_faa(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (f[i,   j, k] + f[i-1,  j, k])

@inline ϊy_aca(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (f[i, j+1, k] + f[i,    j, k])
@inline ϊy_afa(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (f[i,   j, k] + f[i,  j-1, k])

@inline fv(i, j, k, grid::Grid{T}, v::AbstractArray, f::AbstractFloat) where T = T(0.5) * f * (avgy_aca(i-1,  j, k, grid, v) + avgy_aca(i, j, k, grid, v))
@inline fu(i, j, k, grid::Grid{T}, u::AbstractArray, f::AbstractFloat) where T = T(0.5) * f * (avgx_caa(i,  j-1, k, grid, u) + avgx_caa(i, j, k, grid, u))

@inline function ϊz_aac(i, j, k, grid::Grid{T}, f::AbstractArray) where T
    if k == grid.Nz
        @inbounds return T(0.5) * f[i, j, k]
    else
        @inbounds return T(0.5) * (f[i, j, k+1] + f[i, j, k])
    end
end

@inline function ϊz_aaf(i, j, k, grid::Grid{T}, f::AbstractArray) where T
    if k == 1
        @inbounds return f[i, j, k]
    else
        @inbounds return T(0.5) * (f[i, j, k] + f[i, j, k-1])
    end
end

@inline ϊxAx_caa(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (Ax(i+1, j, k, grid) * f[i+1, j, k] + Ax(i,   j, k, grid) * f[i,    j, k])
@inline ϊxAx_faa(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (Ax(i,   j, k, grid) * f[i,   j, k] + Ax(i-1, j, k, grid) * f[i-1,  j, k])
@inline ϊxAy_faa(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (Ay(i,   j, k, grid) * f[i,   j, k] + Ay(i-1, j, k, grid) * f[i-1,  j, k])
@inline ϊxAz_faa(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (Az(i,   j, k, grid) * f[i,   j, k] + Az(i-1, j, k, grid) * f[i-1,  j, k])

@inline ϊyAy_aca(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (Ay(i, j+1, k, grid) * f[i, j+1, k] + Ay(i,   j, k, grid) * f[i,    j, k])
@inline ϊyAx_afa(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (Ax(i,   j, k, grid) * f[i,   j, k] + Ax(i, j-1, k, grid) * f[i,  j-1, k])
@inline ϊyAy_afa(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (Ay(i,   j, k, grid) * f[i,   j, k] + Ay(i, j-1, k, grid) * f[i,  j-1, k])
@inline ϊyAz_afa(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (Az(i,   j, k, grid) * f[i,   j, k] + Az(i, j-1, k, grid) * f[i,  j-1, k])

@inline function ϊzAz_aac(i, j, k, grid::Grid{T}, f::AbstractArray) where T
    if k == grid.Nz
        @inbounds return T(0.5) * Az(i, j, k, grid) * f[i, j, k]
    else
        @inbounds return T(0.5) * (Az(i, j, k+1, grid) * f[i, j, k+1] + Az(i, j, k, grid) * f[i, j, k])
    end
end

@inline function ϊzAx_aaf(i, j, k, grid::Grid{T}, f::AbstractArray) where T
    if k == 1
        @inbounds return Ax(i, j, k, grid) * f[i, j, k]
    else
        @inbounds return T(0.5) * (Ax(i, j, k, grid) * f[i, j, k] + Ax(i, j, k-1, grid) * f[i, j, k-1])
    end
end

@inline function ϊzAy_aaf(i, j, k, grid::Grid{T}, f::AbstractArray) where T
    if k == 1
        @inbounds return Ay(i, j, k, grid) * f[i, j, k]
    else
        @inbounds return T(0.5) * (Ay(i, j, k, grid) * f[i, j, k] + Ay(i, j, k-1, grid) * f[i, j, k-1])
    end
end

@inline function ϊzAz_aaf(i, j, k, grid::Grid{T}, f::AbstractArray) where T
    if k == 1
        @inbounds return Az(i, j, k, grid) * f[i, j, k]
    else
        @inbounds return T(0.5) * (Az(i, j, k, grid) * f[i, j, k] + Az(i, j, k-1, grid) * f[i, j, k-1])
    end
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
