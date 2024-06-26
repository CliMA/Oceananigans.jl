using Oceananigans.Grids: Flat

#####
##### Base interpolation operators without grid
#####

@inline ℑxᶜᵃᵃ(i, j, k, u) = @inbounds (u[i,   j, k] + u[i+1, j, k]) / 2
@inline ℑxᶠᵃᵃ(i, j, k, c) = @inbounds (c[i-1, j, k] + c[i,   j, k]) / 2

@inline ℑyᵃᶜᵃ(i, j, k, v) = @inbounds (v[i, j,   k] + v[i,  j+1, k]) / 2
@inline ℑyᵃᶠᵃ(i, j, k, c) = @inbounds (c[i, j-1, k] + c[i,  j,   k]) / 2

@inline ℑzᵃᵃᶜ(i, j, k, w) = @inbounds (w[i, j,   k] + w[i, j, k+1]) / 2
@inline ℑzᵃᵃᶠ(i, j, k, c) = @inbounds (c[i, j, k-1] + c[i, j,   k]) / 2

#####
##### Base interpolation operators
#####

@inline ℑxᶜᵃᵃ(i, j, k, grid::AG{FT}, u) where FT = @inbounds FT(0.5) * (u[i,   j, k] + u[i+1, j, k])
@inline ℑxᶠᵃᵃ(i, j, k, grid::AG{FT}, c) where FT = @inbounds FT(0.5) * (c[i-1, j, k] + c[i,   j, k])

@inline ℑyᵃᶜᵃ(i, j, k, grid::AG{FT}, v) where FT = @inbounds FT(0.5) * (v[i, j,   k] + v[i,  j+1, k])
@inline ℑyᵃᶠᵃ(i, j, k, grid::AG{FT}, c) where FT = @inbounds FT(0.5) * (c[i, j-1, k] + c[i,  j,   k])

@inline ℑzᵃᵃᶜ(i, j, k, grid::AG{FT}, w) where FT = @inbounds FT(0.5) * (w[i, j,   k] + w[i, j, k+1])
@inline ℑzᵃᵃᶠ(i, j, k, grid::AG{FT}, c) where FT = @inbounds FT(0.5) * (c[i, j, k-1] + c[i, j,   k])

#####
##### Interpolation operators acting on functions
#####

@inline ℑxᶜᵃᵃ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i,   j, k, grid, args...) + f(i+1, j, k, grid, args...))
@inline ℑxᶠᵃᵃ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i-1, j, k, grid, args...) + f(i,   j, k, grid, args...))

@inline ℑyᵃᶜᵃ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i, j,   k, grid, args...) + f(i, j+1, k, grid, args...))
@inline ℑyᵃᶠᵃ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i, j-1, k, grid, args...) + f(i, j,   k, grid, args...))

@inline ℑzᵃᵃᶜ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i, j, k,   grid, args...) + f(i, j, k+1, grid, args...))
@inline ℑzᵃᵃᶠ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i, j, k-1, grid, args...) + f(i, j, k,   grid, args...))

#####
##### Convenience operators for "interpolating constants"
#####

@inline ℑxᶠᵃᵃ(i, j, k, grid::AG, f::Number, args...) = f
@inline ℑxᶜᵃᵃ(i, j, k, grid::AG, f::Number, args...) = f
@inline ℑyᵃᶠᵃ(i, j, k, grid::AG, f::Number, args...) = f
@inline ℑyᵃᶜᵃ(i, j, k, grid::AG, f::Number, args...) = f
@inline ℑzᵃᵃᶠ(i, j, k, grid::AG, f::Number, args...) = f
@inline ℑzᵃᵃᶜ(i, j, k, grid::AG, f::Number, args...) = f

#####
##### Double interpolation
#####

@inline ℑxyᶜᶜᵃ(i, j, k, grid, f, args...) = ℑyᵃᶜᵃ(i, j, k, grid, ℑxᶜᵃᵃ, f, args...)
@inline ℑxyᶠᶜᵃ(i, j, k, grid, f, args...) = ℑyᵃᶜᵃ(i, j, k, grid, ℑxᶠᵃᵃ, f, args...)
@inline ℑxyᶠᶠᵃ(i, j, k, grid, f, args...) = ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶠᵃᵃ, f, args...)
@inline ℑxyᶜᶠᵃ(i, j, k, grid, f, args...) = ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, f, args...)
@inline ℑxzᶜᵃᶜ(i, j, k, grid, f, args...) = ℑzᵃᵃᶜ(i, j, k, grid, ℑxᶜᵃᵃ, f, args...)
@inline ℑxzᶠᵃᶜ(i, j, k, grid, f, args...) = ℑzᵃᵃᶜ(i, j, k, grid, ℑxᶠᵃᵃ, f, args...)
@inline ℑxzᶠᵃᶠ(i, j, k, grid, f, args...) = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶠᵃᵃ, f, args...)
@inline ℑxzᶜᵃᶠ(i, j, k, grid, f, args...) = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶜᵃᵃ, f, args...)
@inline ℑyzᵃᶜᶜ(i, j, k, grid, f, args...) = ℑzᵃᵃᶜ(i, j, k, grid, ℑyᵃᶜᵃ, f, args...)
@inline ℑyzᵃᶠᶜ(i, j, k, grid, f, args...) = ℑzᵃᵃᶜ(i, j, k, grid, ℑyᵃᶠᵃ, f, args...)
@inline ℑyzᵃᶠᶠ(i, j, k, grid, f, args...) = ℑzᵃᵃᶠ(i, j, k, grid, ℑyᵃᶠᵃ, f, args...)
@inline ℑyzᵃᶜᶠ(i, j, k, grid, f, args...) = ℑzᵃᵃᶠ(i, j, k, grid, ℑyᵃᶜᵃ, f, args...)

#####
##### Triple interpolation
#####

@inline ℑxyzᶜᶜᶜ(i, j, k, grid, f, args...) = ℑxᶜᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, ℑzᵃᵃᶜ, f, args...)
@inline ℑxyzᶠᶠᶠ(i, j, k, grid, f, args...) = ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ, f, args...)

@inline ℑxyzᶜᶜᶠ(i, j, k, grid, f, args...) = ℑxᶜᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, ℑzᵃᵃᶠ, f, args...)
@inline ℑxyzᶜᶠᶜ(i, j, k, grid, f, args...) = ℑxᶜᵃᵃ(i, j, k, grid, ℑyᵃᶠᵃ, ℑzᵃᵃᶜ, f, args...)
@inline ℑxyzᶠᶜᶜ(i, j, k, grid, f, args...) = ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, ℑzᵃᵃᶜ, f, args...)

@inline ℑxyzᶜᶠᶠ(i, j, k, grid, f, args...) = ℑxᶜᵃᵃ(i, j, k, grid, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ, f, args...)
@inline ℑxyzᶠᶜᶠ(i, j, k, grid, f, args...) = ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, ℑzᵃᵃᶠ, f, args...)
@inline ℑxyzᶠᶠᶜ(i, j, k, grid, f, args...) = ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶠᵃ, ℑzᵃᵃᶜ, f, args...)

#####
##### Leftover
#####

@inline ℑxᶠᵃᵃ(i, j, k, grid::AG{FT}, c, args...) where FT = @inbounds FT(0.5) * (c[i-1, j, k] + c[i, j, k])
@inline ℑyᵃᶠᵃ(i, j, k, grid::AG{FT}, c, args...) where FT = @inbounds FT(0.5) * (c[i, j-1, k] + c[i, j, k])
@inline ℑzᵃᵃᶠ(i, j, k, grid::AG{FT}, c, args...) where FT = @inbounds FT(0.5) * (c[i, j, k-1] + c[i, j, k])

#####
##### Support for Flat Earths
#####

using Oceananigans.Grids: XFlatGrid, YFlatGrid, ZFlatGrid

@inline ℑxᶜᵃᵃ(i, j, k, grid::XFlatGrid, u) = @inbounds u[i, j, k]
@inline ℑxᶠᵃᵃ(i, j, k, grid::XFlatGrid, c) = @inbounds c[i, j, k]

@inline ℑyᵃᶜᵃ(i, j, k, grid::YFlatGrid, w) = @inbounds w[i, j, k]
@inline ℑyᵃᶠᵃ(i, j, k, grid::YFlatGrid, c) = @inbounds c[i, j, k]

@inline ℑzᵃᵃᶜ(i, j, k, grid::ZFlatGrid, w) = @inbounds w[i, j, k]
@inline ℑzᵃᵃᶠ(i, j, k, grid::ZFlatGrid, c) = @inbounds c[i, j, k]

@inline ℑxᶜᵃᵃ(i, j, k, grid::XFlatGrid, f::F, args...) where {F<:Function} = f(i, j, k, grid, args...)
@inline ℑxᶠᵃᵃ(i, j, k, grid::XFlatGrid, f::F, args...) where {F<:Function} = f(i, j, k, grid, args...)

@inline ℑyᵃᶜᵃ(i, j, k, grid::YFlatGrid, f::F, args...) where {F<:Function} = f(i, j, k, grid, args...)
@inline ℑyᵃᶠᵃ(i, j, k, grid::YFlatGrid, f::F, args...) where {F<:Function} = f(i, j, k, grid, args...)

@inline ℑzᵃᵃᶜ(i, j, k, grid::ZFlatGrid, f::F, args...) where {F<:Function} = f(i, j, k, grid, args...)
@inline ℑzᵃᵃᶠ(i, j, k, grid::ZFlatGrid, f::F, args...) where {F<:Function} = f(i, j, k, grid, args...)
