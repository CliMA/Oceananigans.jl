using Oceananigans.Grids: Flat

#####
##### Base difference operators
#####

@inline δxᶜᵃᵃ(i, j, k, grid, u) = @inbounds u[i+1, j, k] - u[i,   j, k]
@inline δxᶠᵃᵃ(i, j, k, grid, c) = @inbounds c[i,   j, k] - c[i-1, j, k]

@inline δyᵃᶜᵃ(i, j, k, grid, v) = @inbounds v[i, j+1, k] - v[i, j,   k]
@inline δyᵃᶠᵃ(i, j, k, grid, c) = @inbounds c[i, j,   k] - c[i, j-1, k]

@inline δzᵃᵃᶜ(i, j, k, grid, w) = @inbounds w[i, j, k+1] - w[i, j,   k]
@inline δzᵃᵃᶠ(i, j, k, grid, c) = @inbounds c[i, j,   k] - c[i, j, k-1]

#####
##### Difference operators acting on functions
#####

@inline δxᶜᵃᵃ(i, j, k, grid, f::F, args...) where F<:Function = f(i+1, j, k, grid, args...) - f(i,   j, k, grid, args...)
@inline δxᶠᵃᵃ(i, j, k, grid, f::F, args...) where F<:Function = f(i,   j, k, grid, args...) - f(i-1, j, k, grid, args...)

@inline δyᵃᶜᵃ(i, j, k, grid, f::F, args...) where F<:Function = f(i, j+1, k, grid, args...) - f(i, j,   k, grid, args...)
@inline δyᵃᶠᵃ(i, j, k, grid, f::F, args...) where F<:Function = f(i, j,   k, grid, args...) - f(i, j-1, k, grid, args...)

@inline δzᵃᵃᶜ(i, j, k, grid, f::F, args...) where F<:Function = f(i, j, k+1, grid, args...) - f(i, j, k,   grid, args...)
@inline δzᵃᵃᶠ(i, j, k, grid, f::F, args...) where F<:Function = f(i, j, k,   grid, args...) - f(i, j, k-1, grid, args...)

#####
##### Support for Flat Earths
#####

@inline δxᶜᵃᵃ(i, j, k, grid::AG{FT, Flat, TY, TZ}, u) where {FT, TY, TZ} = zero(FT)
@inline δxᶠᵃᵃ(i, j, k, grid::AG{FT, Flat, TY, TZ}, c) where {FT, TY, TZ} = zero(FT)

@inline δyᵃᶜᵃ(i, j, k, grid::AG{FT, TX, Flat, TZ}, v) where {FT, TX, TZ} = zero(FT)
@inline δyᵃᶠᵃ(i, j, k, grid::AG{FT, TX, Flat, TZ}, c) where {FT, TX, TZ} = zero(FT)

@inline δzᵃᵃᶜ(i, j, k, grid::AG{FT, TX, TY, Flat}, w) where {FT, TX, TY} = zero(FT)
@inline δzᵃᵃᶠ(i, j, k, grid::AG{FT, TX, TY, Flat}, c) where {FT, TX, TY} = zero(FT)

@inline δxᶜᵃᵃ(i, j, k, grid::AG{FT, Flat, TY, TZ}, f::F, args...) where {FT, TY, TZ, F<:Function} = zero(FT)
@inline δxᶠᵃᵃ(i, j, k, grid::AG{FT, Flat, TY, TZ}, f::F, args...) where {FT, TY, TZ, F<:Function} = zero(FT)

@inline δyᵃᶜᵃ(i, j, k, grid::AG{FT, TX, Flat, TZ}, f::F, args...) where {FT, TX, TZ, F<:Function} = zero(FT)
@inline δyᵃᶠᵃ(i, j, k, grid::AG{FT, TX, Flat, TZ}, f::F, args...) where {FT, TX, TZ, F<:Function} = zero(FT)

@inline δzᵃᵃᶜ(i, j, k, grid::AG{FT, TX, TY, Flat}, f::F, args...) where {FT, TX, TY, F<:Function} = zero(FT)
@inline δzᵃᵃᶠ(i, j, k, grid::AG{FT, TX, TY, Flat}, f::F, args...) where {FT, TX, TY, F<:Function} = zero(FT)

#####
##### Support for aquaplanet simulations on conformal cubed sphere grids
#####

@inline δxᶠᶜᶜ(i, j, k, grid::OrthogonalSphericalShellGrid, c) =
    @inbounds ifelse((i == 1) & (j < 1),               c[1, j, k]           - c[j, 1, k],
              ifelse((i == grid.Nx+1) & (j < 1),       c[grid.Nx-j+1, 1, k] - c[grid.Nx, j, k],
              ifelse((i == grid.Nx+1) & (j > grid.Ny), c[j, grid.Ny, k]     - c[grid.Nx, j, k],
              ifelse((i == 1) & (j > grid.Ny),         c[1, j, k]           - c[grid.Ny-j+1, grid.Ny, k],
                                                       c[i, j, k]           - c[i-1, j, k]))))

@inline δxᶠᶜᶠ(i, j, k, grid::OrthogonalSphericalShellGrid, c) = δxᶠᶜᶜ(i, j, k, grid, c)

@inline δyᶜᶠᶜ(i, j, k, grid::OrthogonalSphericalShellGrid, c) =
    @inbounds ifelse((i < 1) & (j == 1),               c[i, 1, k]           - c[1, i, k],
              ifelse((i > grid.Nx) & (j == 1),         c[i, 1, k]           - c[grid.Nx, grid.Ny+1-i, k],
              ifelse((i > grid.Nx) & (j == grid.Ny+1), c[grid.Nx, i, k]     - c[i, grid.Ny, k],
              ifelse((i < 1) & (j == grid.Ny+1),       c[1, grid.Ny-i+1, k] - c[i, grid.Ny, k],
                                                       c[i, j, k]           - c[i, j-1, k]))))

@inline δyᶜᶠᶠ(i, j, k, grid::OrthogonalSphericalShellGrid, c) = δyᶜᶠᶜ(i, j, k, grid, c)

@inline δxᶠᶜᶜ(i, j, k, grid::OrthogonalSphericalShellGrid, f::F, args...) where F<:Function =
    @inbounds ifelse((i == 1) & (j < 1),               f(1, j, k, grid, args...)           - f(j, 1, k, grid, args...),
              ifelse((i == grid.Nx+1) & (j < 1),       f(grid.Nx-j+1, 1, k, grid, args...) - f(grid.Nx, j, k, grid, args...),
              ifelse((i == grid.Nx+1) & (j > grid.Ny), f(j, grid.Ny, k, grid, args...)     - f(grid.Nx, j, k, grid, args...),
              ifelse((i == 1) & (j > grid.Ny),         f(1, j, k, grid, args...)           - f(grid.Nx-j+1, grid.Ny, k, grid, args...),
                                                       f(i, j, k, grid, args...)           - f(i-1, j, k, grid, args...)))))

@inline δxᶠᶜᶠ(i, j, k, grid::OrthogonalSphericalShellGrid, f::F, args...) where F<:Function =
    δxᶠᶜᶜ(i, j, k, grid, f, args...)

@inline δyᶜᶠᶜ(i, j, k, grid::OrthogonalSphericalShellGrid, f::F, args...) where F<:Function =
    @inbounds ifelse((i < 1) & (j == 1),               f(i, 1, k, grid, args...)           - f(1, i, k, grid, args...),
              ifelse((i > grid.Nx) & (j == 1),         f(i, 1, k, grid, args...)           - f(grid.Nx, grid.Ny+1-i, k, grid, args...),
              ifelse((i > grid.Nx) & (j == grid.Ny+1), f(grid.Nx, i, k, grid, args...)     - f(i, grid.Ny, k, grid, args...),
              ifelse((i < 1) & (j == grid.Ny+1),       f(1, grid.Ny-i+1, k, grid, args...) - f(i, grid.Ny, k, grid, args...),
                                                       f(i, j, k, grid, args...)           - f(i, j-1, k, grid, args...)))))

@inline δyᶜᶠᶠ(i, j, k, grid::OrthogonalSphericalShellGrid, f::F, args...) where F<:Function =
    δyᶜᶠᶜ(i, j, k, grid, f, args...)

#####
##### 3D differences
#####

for ℓx in (:ᶜ, :ᶠ), ℓy in (:ᶜ, :ᶠ), ℓz in (:ᶜ, :ᶠ)
    δx = Symbol(:δx, ℓx, ℓy, ℓz)
    δy = Symbol(:δy, ℓx, ℓy, ℓz)
    δz = Symbol(:δz, ℓx, ℓy, ℓz)

    δxᵃ = Symbol(:δx, ℓx, :ᵃ, :ᵃ)
    δyᵃ = Symbol(:δy, :ᵃ, ℓy, :ᵃ)
    δzᵃ = Symbol(:δz, :ᵃ, :ᵃ, ℓz)

    @eval begin
        @inline $δx(i, j, k, grid, f::Function, args...) = $δxᵃ(i, j, k, grid, f, args...)
        @inline $δy(i, j, k, grid, f::Function, args...) = $δyᵃ(i, j, k, grid, f, args...)
        @inline $δz(i, j, k, grid, f::Function, args...) = $δzᵃ(i, j, k, grid, f, args...)

        @inline $δx(i, j, k, grid, c) = $δxᵃ(i, j, k, grid, c)
        @inline $δy(i, j, k, grid, c) = $δyᵃ(i, j, k, grid, c)
        @inline $δz(i, j, k, grid, c) = $δzᵃ(i, j, k, grid, c)
    end
end
