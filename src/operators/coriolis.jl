# Operators that calculate the Coriolis terms.
@inline fv(i, j, k, grid::Grid{T}, v::AbstractArray, f::AbstractFloat) where T = T(0.5) * f * (ϊy_aca(i-1,  j, k, grid, v) + ϊy_aca(i, j, k, grid, v))
@inline fu(i, j, k, grid::Grid{T}, u::AbstractArray, f::AbstractFloat) where T = T(0.5) * f * (ϊx_caa(i,  j-1, k, grid, u) + ϊx_caa(i, j, k, grid, u))
