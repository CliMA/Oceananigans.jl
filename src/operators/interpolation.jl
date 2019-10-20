const RCG = RegularCartesianGrid

####
#### Base interpolation operators
####

@inline ℑx_caa(i, j, k, grid::RCG{FT}, u) where FT = @inbounds FT(0.5) * (u[i,   j, k] + u[i+1, j, k])
@inline ℑx_faa(i, j, k, grid::RCG{FT}, c) where FT = @inbounds FT(0.5) * (c[i-1, j, k] + c[i,   j, k])

@inline ℑy_aca(i, j, k, grid::RCG{FT}, v) where FT = @inbounds FT(0.5) * (v[i, j,   k] + v[i,  j+1, k])
@inline ℑy_afa(i, j, k, grid::RCG{FT}, c) where FT = @inbounds FT(0.5) * (c[i, j-1, k] + c[i,  j,   k])

@inline ℑz_aac(i, j, k, grid::RCG{FT}, w) where FT = @inbounds FT(0.5) * (w[i, j,   k] + w[i, j, k+1])
@inline ℑz_aaf(i, j, k, grid::RCG{FT}, c) where FT = @inbounds FT(0.5) * (c[i, j, k-1] + c[i, j,   k])

####
#### Interpolation operators acting on functions
####

@inline ℑx_caa(i, j, k, grid::RCG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i,   j, k, grid, args...) + f(i+1, j, k, grid, args...))
@inline ℑx_faa(i, j, k, grid::RCG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i-1, j, k, grid, args...) + f(i,   j, k, grid, args...))

@inline ℑy_aca(i, j, k, grid::RCG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i, j,   k, grid, args...) + f(i, j+1, k, grid, args...))
@inline ℑy_afa(i, j, k, grid::RCG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i, j-1, k, grid, args...) + f(i, j,   k, grid, args...))

@inline ℑz_aac(i, j, k, grid::RCG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i, j, k,   grid, args...) + f(i, j, k+1, grid, args...))
@inline ℑz_aaf(i, j, k, grid::RCG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i, j, k-1, grid, args...) + f(i, j, k,   grid, args...))

####
#### Convenience operators for "interpolating constants"
####

@inline ℑx_faa(i, j, k, grid, f::Number, args...) = f
@inline ℑx_caa(i, j, k, grid, f::Number, args...) = f
@inline ℑy_afa(i, j, k, grid, f::Number, args...) = f
@inline ℑy_aca(i, j, k, grid, f::Number, args...) = f
@inline ℑz_aaf(i, j, k, grid, f::Number, args...) = f
@inline ℑz_aac(i, j, k, grid, f::Number, args...) = f
