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
##### 3D differences
#####

for ℓx in (:ᶜ, :ᶠ), ℓy in (:ᶜ, :ᶠ), ℓz in (:ᶜ, :ᶠ)
    δx = Symbol(:δx, ℓx, ℓy, ℓz)
    δy = Symbol(:δy, ℓx, ℓy, ℓz)
    δz = Symbol(:δz, ℓx, ℓy, ℓz)

    δxᵃᵃ = Symbol(:δx, ℓx, :ᵃ, :ᵃ)
    δyᵃᵃ = Symbol(:δy, :ᵃ, ℓy, :ᵃ)
    δzᵃᵃ = Symbol(:δz, :ᵃ, :ᵃ, ℓz)

    δx_xy = Symbol(:δx, ℓx, ℓy, :ᵃ)
    δx_xz = Symbol(:δx, ℓx, :ᵃ, ℓz)
    δx_yz = Symbol(:δx, :ᵃ, ℓy, ℓz)

    δy_xy = Symbol(:δy, ℓx, ℓy, :ᵃ)
    δy_xz = Symbol(:δy, ℓx, :ᵃ, ℓz)
    δy_yz = Symbol(:δy, :ᵃ, ℓy, ℓz)

    δz_xy = Symbol(:δz, ℓx, ℓy, :ᵃ)
    δz_xz = Symbol(:δz, ℓx, :ᵃ, ℓz)
    δz_yz = Symbol(:δz, :ᵃ, ℓy, ℓz)


    @eval begin
        @inline $δx(i, j, k, grid, f::Function, args...) = $δxᵃᵃ(i, j, k, grid, f, args...)
        @inline $δy(i, j, k, grid, f::Function, args...) = $δyᵃᵃ(i, j, k, grid, f, args...)
        @inline $δz(i, j, k, grid, f::Function, args...) = $δzᵃᵃ(i, j, k, grid, f, args...)

        @inline $δx(i, j, k, grid, c) = $δxᵃᵃ(i, j, k, grid, c)
        @inline $δy(i, j, k, grid, c) = $δyᵃᵃ(i, j, k, grid, c)
        @inline $δz(i, j, k, grid, c) = $δzᵃᵃ(i, j, k, grid, c)

        @inline $δx_xy(i, j, k, grid, c) = $δxᵃᵃ(i, j, k, grid, c)
        @inline $δx_xz(i, j, k, grid, c) = $δxᵃᵃ(i, j, k, grid, c)
        @inline $δx_yz(i, j, k, grid, c) = $δxᵃᵃ(i, j, k, grid, c)

        @inline $δy_xy(i, j, k, grid, c) = $δyᵃᵃ(i, j, k, grid, c)
        @inline $δy_xz(i, j, k, grid, c) = $δyᵃᵃ(i, j, k, grid, c)
        @inline $δy_yz(i, j, k, grid, c) = $δyᵃᵃ(i, j, k, grid, c)

        @inline $δz_xy(i, j, k, grid, c) = $δzᵃᵃ(i, j, k, grid, c)
        @inline $δz_xz(i, j, k, grid, c) = $δzᵃᵃ(i, j, k, grid, c)
        @inline $δz_yz(i, j, k, grid, c) = $δzᵃᵃ(i, j, k, grid, c)

        @inline $δx_xy(i, j, k, grid, f::Function, args...) = $δxᵃᵃ(i, j, k, grid, f, args...)
        @inline $δx_xz(i, j, k, grid, f::Function, args...) = $δxᵃᵃ(i, j, k, grid, f, args...)
        @inline $δx_yz(i, j, k, grid, f::Function, args...) = $δxᵃᵃ(i, j, k, grid, f, args...)

        @inline $δy_xy(i, j, k, grid, f::Function, args...) = $δyᵃᵃ(i, j, k, grid, f, args...)
        @inline $δy_xz(i, j, k, grid, f::Function, args...) = $δyᵃᵃ(i, j, k, grid, f, args...)
        @inline $δy_yz(i, j, k, grid, f::Function, args...) = $δyᵃᵃ(i, j, k, grid, f, args...)

        @inline $δz_xy(i, j, k, grid, f::Function, args...) = $δzᵃᵃ(i, j, k, grid, f, args...)
        @inline $δz_xz(i, j, k, grid, f::Function, args...) = $δzᵃᵃ(i, j, k, grid, f, args...)
        @inline $δz_yz(i, j, k, grid, f::Function, args...) = $δzᵃᵃ(i, j, k, grid, f, args...)
    end
end

