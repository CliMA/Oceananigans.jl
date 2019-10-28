using Oceananigans: AbstractGrid

####
#### Convinient aliases
####

const AG  = AbstractGrid
const RCG = RegularCartesianGrid

####
#### Base interpolation operators
####

@inline ℑxᶜᵃᵃ(i, j, k, grid::AG{FT}, u) where FT = @inbounds FT(0.5) * (u[i,   j, k] + u[i+1, j, k])
@inline ℑxᶠᵃᵃ(i, j, k, grid::AG{FT}, c) where FT = @inbounds FT(0.5) * (c[i-1, j, k] + c[i,   j, k])

@inline ℑyᵃᶜᵃ(i, j, k, grid::AG{FT}, v) where FT = @inbounds FT(0.5) * (v[i, j,   k] + v[i,  j+1, k])
@inline ℑyᵃᶠᵃ(i, j, k, grid::AG{FT}, c) where FT = @inbounds FT(0.5) * (c[i, j-1, k] + c[i,  j,   k])

@inline ℑzᵃᵃᶜ(i, j, k, grid::AG{FT}, w) where FT = @inbounds FT(0.5) * (w[i, j,   k] + w[i, j, k+1])
@inline ℑzᵃᵃᶠ(i, j, k, grid::AG{FT}, c) where FT = @inbounds FT(0.5) * (c[i, j, k-1] + c[i, j,   k])

####
#### Interpolation operators acting on functions
####

@inline ℑxᶜᵃᵃ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i,   j, k, grid, args...) + f(i+1, j, k, grid, args...))
@inline ℑxᶠᵃᵃ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i-1, j, k, grid, args...) + f(i,   j, k, grid, args...))

@inline ℑyᵃᶜᵃ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i, j,   k, grid, args...) + f(i, j+1, k, grid, args...))
@inline ℑyᵃᶠᵃ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i, j-1, k, grid, args...) + f(i, j,   k, grid, args...))

@inline ℑzᵃᵃᶜ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i, j, k,   grid, args...) + f(i, j, k+1, grid, args...))
@inline ℑzᵃᵃᶠ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i, j, k-1, grid, args...) + f(i, j, k,   grid, args...))

####
#### "Flux interpolation" operators of the form ℑ(A*f) where A is an area and f is an array.
####

@inline ℑᴶxᶜᵃᵃ(i, j, k, grid, u) = ℑxᶜᵃᵃ(i, j, k, grid, Ax_u, u)
@inline ℑᴶxᶠᵃᵃ(i, j, k, grid, c) = ℑxᶠᵃᵃ(i, j, k, grid, Ax_c, c)
@inline ℑᴶyᵃᶜᵃ(i, j, k, grid, v) = ℑyᵃᶜᵃ(i, j, k, grid, Ay_v, v)
@inline ℑᴶyᵃᶠᵃ(i, j, k, grid, c) = ℑyᵃᶠᵃ(i, j, k, grid, Ay_c, c)
@inline ℑᴶzᵃᵃᶜ(i, j, k, grid, w) = ℑzᵃᵃᶜ(i, j, k, grid, Az_w, w)
@inline ℑᴶzᵃᵃᶠ(i, j, k, grid, c) = ℑzᵃᵃᶠ(i, j, k, grid, Az_c, c)

####
#### Convenience operators for "interpolating constants"
####

@inline ℑxᶠᵃᵃ(i, j, k, grid, f::Number, args...) = f
@inline ℑxᶜᵃᵃ(i, j, k, grid, f::Number, args...) = f
@inline ℑyᵃᶠᵃ(i, j, k, grid, f::Number, args...) = f
@inline ℑyᵃᶜᵃ(i, j, k, grid, f::Number, args...) = f
@inline ℑzᵃᵃᶠ(i, j, k, grid, f::Number, args...) = f
@inline ℑzᵃᵃᶜ(i, j, k, grid, f::Number, args...) = f

####
#### Double interpolation
####

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

####
#### Triple interpolation
####

@inline ℑxyzᶠᶠᶜ(i, j, k, grid, f, args...) = ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶠᵃ, ℑzᵃᵃᶜ, f, args...)
@inline ℑxyzᶜᶜᶠ(i, j, k, grid, f, args...) = ℑxᶜᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, ℑzᵃᵃᶠ, f, args...)

