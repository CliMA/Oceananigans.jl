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
#### Difference operators of the form δ(A*f) where A is an area and f is an array.
####

@inline δxA_caa(i, j, k, grid, u) = @inbounds AxC(i+1, j, k, grid) * u[i+1, j, k] - AxC(i,   j, k, grid) * u[i,   j, k]
@inline δxA_faa(i, j, k, grid, c) = @inbounds AxF(i,   j, k, grid) * c[i,   j, k] - AxF(i-1, j, k, grid) * c[i-1, j, k]

@inline δyA_aca(i, j, k, grid, v) = @inbounds AyC(i, j+1, k, grid) * v[i, j+1, k] - AyC(i, j,   k, grid) * v[i, j,   k]
@inline δyA_afa(i, j, k, grid, c) = @inbounds AyF(i,   j, k, grid) * c[i, j,   k] - AyF(i, j-1, k, grid) * c[i, j-1, k]

@inline δzA_aac(i, j, k, grid, w) = @inbounds Az(i, j, k+1, grid) * w[i, j, k+1] - Az(i, j, k,   grid) * w[i, j,   k]
@inline δzA_aaf(i, j, k, grid, c) = @inbounds Az(i, j,   k, grid) * c[i, j,   k] - Az(i, j, k-1, grid) * c[i, j, k-1]

####
#### Derivative operators
####

@inline ∂x_caa(i, j, k, grid, u) = δx_caa(i, j, k, grid, u) / Δx(i, j, k, grid)
@inline ∂x_faa(i, j, k, grid, c) = δx_faa(i, j, k, grid, c) / Δx(i, j, k, grid)

@inline ∂y_aca(i, j, k, grid, v) = δy_aca(i, j, k, grid, u) / Δy(i, j, k, grid)
@inline ∂y_afa(i, j, k, grid, c) = δy_afa(i, j, k, grid, c) / Δy(i, j, k, grid)

@inline ∂z_aac(i, j, k, grid, w) = δx_aac(i, j, k, grid, w) / ΔzF(i, j, k, grid)
@inline ∂z_aaf(i, j, k, grid, c) = δx_aaf(i, j, k, grid, c) / ΔzC(i, j, k, grid)

####
#### Derivative operators acting on functions
####

@inline ∂x_caa(f::F, i, j, k, grid, args...) where F<:Function = (f(i+1, j, k, grid, args...) - f(i,   j, k, grid, args...)) / Δx(i, j, k, grid)
@inline ∂x_faa(f::F, i, j, k, grid, args...) where F<:Function = (f(i,   j, k, grid, args...) - f(i-1, j, k, grid, args...)) / Δx(i, j, k, grid)

@inline ∂y_aca(f::F, i, j, k, grid, args...) where F<:Function = (f(i, j+1, k, grid, args...) - f(i, j,   k, grid, args...)) / Δy(i, j, k, grid)
@inline ∂y_afa(f::F, i, j, k, grid, args...) where F<:Function = (f(i, j,   k, grid, args...) - f(i, j-1, k, grid, args...)) / Δy(i, j, k, grid)

@inline ∂z_aac(f::F, i, j, k, grid, args...) where F<:Function = (f(i, j, k+1, grid, args...) - f(i, j, k,   grid, args...)) / ΔzF(i, j, k, grid)
@inline ∂z_aaf(f::F, i, j, k, grid, args...) where F<:Function = (f(i, j, k,   grid, args...) - f(i, j, k-1, grid, args...)) / ΔzC(i, j, k, grid)

####
#### Second derivatives
####

@inline ∂x²_caa(i, j, k, grid, c) = ∂x_caa(∂x_faa, i, j, k, grid, c)
@inline ∂x²_faa(i, j, k, grid, u) = ∂x_faa(∂x_caa, i, j, k, grid, u)

@inline ∂y²_aca(i, j, k, grid, c) = ∂y_aca(∂y_afa, i, j, k, grid, c)
@inline ∂y²_afa(i, j, k, grid, v) = ∂y_afa(∂y_aca, i, j, k, grid, v)

@inline ∂z²_aac(i, j, k, grid, c) = ∂z_aac(∂z_aaf, i, j, k, grid, c)
@inline ∂z²_aaf(i, j, k, grid, w) = ∂z_aaf(∂z_aac, i, j, k, grid, w)

