####
#### Derivative operators
####

@inline ∂x_caa(i, j, k, grid, u) = δx_caa(i, j, k, grid, u) / Δx(i, j, k, grid)
@inline ∂x_faa(i, j, k, grid, c) = δx_faa(i, j, k, grid, c) / Δx(i, j, k, grid)

@inline ∂y_aca(i, j, k, grid, v) = δy_aca(i, j, k, grid, u) / Δy(i, j, k, grid)
@inline ∂y_afa(i, j, k, grid, c) = δy_afa(i, j, k, grid, c) / Δy(i, j, k, grid)

@inline ∂z_aac(i, j, k, grid, w) = δz_aac(i, j, k, grid, w) / ΔzF(i, j, k, grid)
@inline ∂z_aaf(i, j, k, grid, c) = δz_aaf(i, j, k, grid, c) / ΔzC(i, j, k, grid)

####
#### Derivative operators acting on functions
####

@inline ∂x_caa(i, j, k, grid, f::F, args...) where F<:Function = (f(i+1, j, k, grid, args...) - f(i,   j, k, grid, args...)) / Δx(i, j, k, grid)
@inline ∂x_faa(i, j, k, grid, f::F, args...) where F<:Function = (f(i,   j, k, grid, args...) - f(i-1, j, k, grid, args...)) / Δx(i, j, k, grid)

@inline ∂y_aca(i, j, k, grid, f::F, args...) where F<:Function = (f(i, j+1, k, grid, args...) - f(i, j,   k, grid, args...)) / Δy(i, j, k, grid)
@inline ∂y_afa(i, j, k, grid, f::F, args...) where F<:Function = (f(i, j,   k, grid, args...) - f(i, j-1, k, grid, args...)) / Δy(i, j, k, grid)

@inline ∂z_aac(i, j, k, grid, f::F, args...) where F<:Function = (f(i, j, k+1, grid, args...) - f(i, j, k,   grid, args...)) / ΔzF(i, j, k, grid)
@inline ∂z_aaf(i, j, k, grid, f::F, args...) where F<:Function = (f(i, j, k,   grid, args...) - f(i, j, k-1, grid, args...)) / ΔzC(i, j, k, grid)

####
#### Second derivatives
####

@inline ∂x²_caa(i, j, k, grid, c) = ∂x_caa(i, j, k, grid, ∂x_faa, c)
@inline ∂x²_faa(i, j, k, grid, u) = ∂x_faa(i, j, k, grid, ∂x_caa, u)

@inline ∂y²_aca(i, j, k, grid, c) = ∂y_aca(i, j, k, grid, ∂y_afa, c)
@inline ∂y²_afa(i, j, k, grid, v) = ∂y_afa(i, j, k, grid, ∂y_aca, v)

@inline ∂z²_aac(i, j, k, grid, c) = ∂z_aac(i, j, k, grid, ∂z_aaf, c)
@inline ∂z²_aaf(i, j, k, grid, w) = ∂z_aaf(i, j, k, grid, ∂z_aac, w)

