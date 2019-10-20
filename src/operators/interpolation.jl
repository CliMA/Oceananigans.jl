####
#### Base interpolation operators
####

@inline ℑx_caa(i, j, k, grid::AbstractGrid{FT}, u) where FT = @inbounds FT(0.5) * (u[i,   j, k] + u[i+1, j, k])
@inline ℑx_faa(i, j, k, grid::AbstractGrid{FT}, c) where FT = @inbounds FT(0.5) * (c[i-1, j, k] + c[i,   j, k])

@inline ℑy_aca(i, j, k, grid::AbstractGrid{FT}, v) where FT = @inbounds FT(0.5) * (v[i, j,   k] + v[i,  j+1, k])
@inline ℑy_afa(i, j, k, grid::AbstractGrid{FT}, c) where FT = @inbounds FT(0.5) * (c[i, j-1, k] + c[i,  j,   k])

@inline ℑz_aac(i, j, k, grid::AbstractGrid{FT}, w) where FT = @inbounds FT(0.5) * (w[i, j,   k] + w[i, j, k+1])
@inline ℑz_aaf(i, j, k, grid::AbstractGrid{FT}, c) where FT = @inbounds FT(0.5) * (c[i, j, k-1] + c[i, j,   k])

