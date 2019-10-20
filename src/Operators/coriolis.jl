####
#### Coriolis terms following Marshall et al. (1997) equation (25).
####

@inline fVv̅ʸ(i, j, k, grid, v, f) = f * V(i, j, k, grid) * ℑy_aca(i, j, k, grid, v)
@inline fVu̅ˣ(i, j, k, grid, u, f) = f * V(i, j, k, grid) * ℑx_caa(i, j, k, grid, u)

@inline fv(i, j, k, grid, v, f) = 1/Vᵘ(i, j, k, grid) * ℑx_faa(i, j, k, grid, fVv̅ʸ, v, f)
@inline fu(i, j, k, grid, u, f) = 1/Vᵛ(i, j, k, grid) * ℑy_afa(i, j, k, grid, fVu̅ˣ, u, f)

