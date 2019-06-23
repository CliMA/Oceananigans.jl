""" Calculate fVv̅ʸ -> ccc. """
@inline fVv̅ʸ(i, j, k, grid::Grid, v::AbstractArray, f::AbstractFloat) =
    f * V(i, j, k, grid) * ϊy_aca(i, j, k, grid, v)

"""
    fv(i, j, k, grid::Grid{T}, v::AbstractArray, f::AbstractFloat)

Calculate the Coriolis term in the u-momentum equation, 1/Vᵘ * (fVv̅ʸ)ˣ -> fcc.
"""
@inline fv(i, j, k, grid::Grid{T}, v::AbstractArray, f::AbstractFloat) where T =
    1/Vᵘ(i, j, k, grid) * T(0.5) * (fVv̅ʸ(i-1, j, k, grid, v) + fVv̅ʸ(i, j, k, grid, v))


""" Calculate fVu̅ˣ -> ccc. """
@inline fVu̅ˣ(i, j, k, grid::Grid, u::AbstractArray, f::AbstractFloat) =
    f * V(i, j, k, grid) * ϊx_caa(i, j, k, grid, u)

"""
    fu(i, j, k, grid::Grid{T}, u::AbstractArray, f::AbstractFloat)

Calculate the Coriolis term in the v-momentum equation, 1/Vᵛ * (fVu̅ˣ)ʸ.

Note that the minus sign is not included in here as this operator just computes
the "magnitude of fu". The reasoning behind this is so that the minus sign can
show up in the Gv calculation making the calculation of the right-hand-side
look like the continuous equations of motion where -fu is more familiar.
"""
@inline fu(i, j, k, grid::Grid{T}, u::AbstractArray, f::AbstractFloat) where T =
    1/Vᵛ(i, j, k, grid) * T(0.5) * (fVu̅ˣ(i, j-1, k, grid, u) + fVu̅ˣ(i, j, k, grid, u))
