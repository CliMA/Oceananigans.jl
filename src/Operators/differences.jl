####
#### Base difference operators
####

@inline δx_caa(i, j, k, grid, u) = @inbounds u[i+1, j, k] - u[i,   j, k]
@inline δx_faa(i, j, k, grid, c) = @inbounds c[i,   j, k] - c[i-1, j, k]

@inline δy_aca(i, j, k, grid, v) = @inbounds v[i, j+1, k] - v[i, j,   k]
@inline δy_afa(i, j, k, grid, c) = @inbounds c[i, j,   k] - c[i, j-1, k]

@inline δz_aac(i, j, k, grid, w) = @inbounds w[i, j, k+1] - w[i, j,   k]
@inline δz_aaf(i, j, k, grid, c) = @inbounds c[i, j,   k] - c[i, j, k-1]

####
#### Difference operators acting on functions
####

@inline δx_caa(i, j, k, grid, f::F, args...) where F<:Function = f(i+1, j, k, grid, args...) - f(i,   j, k, grid, args...)
@inline δx_faa(i, j, k, grid, f::F, args...) where F<:Function = f(i,   j, k, grid, args...) - f(i-1, j, k, grid, args...)

@inline δy_aca(i, j, k, grid, f::F, args...) where F<:Function = f(i, j+1, k, grid, args...) - f(i, j,   k, grid, args...)
@inline δy_afa(i, j, k, grid, f::F, args...) where F<:Function = f(i, j,   k, grid, args...) - f(i, j-1, k, grid, args...)

@inline δz_aac(i, j, k, grid, f::F, args...) where F<:Function = f(i, j, k+1, grid, args...) - f(i, j, k,   grid, args...)
@inline δz_aaf(i, j, k, grid, f::F, args...) where F<:Function = f(i, j, k,   grid, args...) - f(i, j, k-1, grid, args...)

####
#### Operators that calculate A*q where A is an area and q is some quantity.
#### Useful for defining flux difference operators and other flux operators.
####

@inline Ax_u(i, j, k, grid, u) = @inbounds AxF(i, j, k, grid) * u[i, j, k]
@inline Ax_c(i, j, k, grid, c) = @inbounds AxC(i, j, k, grid) * c[i, j, k]
@inline Ay_v(i, j, k, grid, v) = @inbounds AyF(i, j, k, grid) * v[i, j, k]
@inline Ay_c(i, j, k, grid, c) = @inbounds AyC(i, j, k, grid) * c[i, j, k]
@inline Az_w(i, j, k, grid, w) = @inbounds  Az(i, j, k, grid) * w[i, j, k]
@inline Az_c(i, j, k, grid, c) = @inbounds  Az(i, j, k, grid) * c[i, j, k]

####
#### "Flux difference" operators of the form δ(A*f) where A is an area and f is an array.
####

@inline δFx_caa(i, j, k, grid, u) = δx_caa(i, j, k, grid, Ax_u, u)
@inline δFx_faa(i, j, k, grid, c) = δx_faa(i, j, k, grid, Ax_c, c)
@inline δFy_aca(i, j, k, grid, v) = δy_aca(i, j, k, grid, Ay_v, v)
@inline δFy_afa(i, j, k, grid, c) = δy_afa(i, j, k, grid, Ay_c, c)
@inline δFz_aac(i, j, k, grid, w) = δz_aac(i, j, k, grid, Az_w, w)
@inline δFz_aaf(i, j, k, grid, c) = δz_aaf(i, j, k, grid, Az_c, c)

