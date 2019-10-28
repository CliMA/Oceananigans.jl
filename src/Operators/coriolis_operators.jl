####
#### Coriolis terms following Marshall et al. (1997) equation (25).
####

@inline fv(i, j, k, grid, v, f) = f * ℑxy_fca(i, j, k, grid, v)
@inline fu(i, j, k, grid, u, f) = f * ℑxy_cfa(i, j, k, grid, u)

