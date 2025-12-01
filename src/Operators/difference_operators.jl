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
##### 3D and 2D differences
#####

# Export one dimensional difference operators
export δxᶠᵃᵃ, δxᶜᵃᵃ
export δyᵃᶠᵃ, δyᵃᶜᵃ
export δzᵃᵃᶠ, δzᵃᵃᶜ

# Define and export 2D and 3D differences
for ℓ1 in (:ᶜ, :ᶠ), ℓ2 in (:ᶜ, :ᶠ, :ᵃ), ℓ3 in (:ᶜ, :ᶠ, :ᵃ)
    if !(ℓ2 == ℓ3 == :ᵃ) # 1D differences are defined above!
        δx = Symbol(:δx, ℓ1, ℓ2, ℓ3)
        δy = Symbol(:δy, ℓ2, ℓ1, ℓ3)
        δz = Symbol(:δz, ℓ2, ℓ3, ℓ1)

        δxᵃ = Symbol(:δx, ℓ1, :ᵃ, :ᵃ)
        δyᵃ = Symbol(:δy, :ᵃ, ℓ1, :ᵃ)
        δzᵃ = Symbol(:δz, :ᵃ, :ᵃ, ℓ1)

        @eval begin
            @inline $δx(i, j, k, grid, f::Function, args...) = $δxᵃ(i, j, k, grid, f, args...)
            @inline $δy(i, j, k, grid, f::Function, args...) = $δyᵃ(i, j, k, grid, f, args...)
            @inline $δz(i, j, k, grid, f::Function, args...) = $δzᵃ(i, j, k, grid, f, args...)

            @inline $δx(i, j, k, grid, c) = $δxᵃ(i, j, k, grid, c)
            @inline $δy(i, j, k, grid, c) = $δyᵃ(i, j, k, grid, c)
            @inline $δz(i, j, k, grid, c) = $δzᵃ(i, j, k, grid, c)

            export $δx, $δy, $δz
        end
    end
end
