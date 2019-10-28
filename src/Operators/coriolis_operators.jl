####
#### Coriolis terms following Marshall et al. (1997) equation (25).
####

@inline fv(i, j, k, grid, v, f) = f * ℑxyᶠᶜᵃ(i, j, k, grid, v)
@inline fu(i, j, k, grid, u, f) = f * ℑxyᶜᶠᵃ(i, j, k, grid, u)

