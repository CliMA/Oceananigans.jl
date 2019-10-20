####
#### Convinient aliases
####

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

####
#### Double interpolation
####

@inline ℑxy_cca(i, j, k, grid, f, args...) = ℑy_aca(i, j, k, grid, ℑx_caa, f, args...)
@inline ℑxy_fca(i, j, k, grid, f, args...) = ℑy_aca(i, j, k, grid, ℑx_faa, f, args...)
@inline ℑxy_ffa(i, j, k, grid, f, args...) = ℑy_afa(i, j, k, grid, ℑx_faa, f, args...)
@inline ℑxy_cfa(i, j, k, grid, f, args...) = ℑy_afa(i, j, k, grid, ℑx_caa, f, args...)
@inline ℑxz_cac(i, j, k, grid, f, args...) = ℑz_aac(i, j, k, grid, ℑx_caa, f, args...)
@inline ℑxz_fac(i, j, k, grid, f, args...) = ℑz_aac(i, j, k, grid, ℑx_faa, f, args...)
@inline ℑxz_faf(i, j, k, grid, f, args...) = ℑz_aaf(i, j, k, grid, ℑx_faa, f, args...)
@inline ℑxz_caf(i, j, k, grid, f, args...) = ℑz_aaf(i, j, k, grid, ℑx_caa, f, args...)
@inline ℑyz_acc(i, j, k, grid, f, args...) = ℑz_aac(i, j, k, grid, ℑy_aca, f, args...)
@inline ℑyz_afc(i, j, k, grid, f, args...) = ℑz_aac(i, j, k, grid, ℑy_afa, f, args...)
@inline ℑyz_aff(i, j, k, grid, f, args...) = ℑz_aaf(i, j, k, grid, ℑy_afa, f, args...)
@inline ℑyz_acf(i, j, k, grid, f, args...) = ℑz_aaf(i, j, k, grid, ℑy_aca, f, args...)

####
#### Triple interpolation 
####

@inline ℑxyz_ffc(i, j, k, grid, f, args...) = ℑx_faa(i, j, k, grid, ℑy_afa, ℑz_aac, f, args...)
@inline ℑxyz_ccf(i, j, k, grid, f, args...) = ℑx_caa(i, j, k, grid, ℑy_aca, ℑz_aaf, f, args...)

