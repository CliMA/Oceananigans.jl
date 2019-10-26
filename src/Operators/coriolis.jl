####
#### Coriolis terms following Marshall et al. (1997) equation (25).
####

@inline fv̅ʸ(i, j, k, grid, v, f) = f * ℑy_aca(i, j, k, grid, v)
@inline fu̅ˣ(i, j, k, grid, u, f) = f * ℑx_caa(i, j, k, grid, u)

@inline fv(i, j, k, grid, v, f) = ℑx_faa(i, j, k, grid, fv̅ʸ, v, f)
@inline fu(i, j, k, grid, u, f) = ℑy_afa(i, j, k, grid, fu̅ˣ, u, f)

