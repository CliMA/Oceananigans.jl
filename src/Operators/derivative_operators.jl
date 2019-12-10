####
#### Derivative operators
####

@inline ∂xᶜᵃᵃ(i, j, k, grid, u) = δxᶜᵃᵃ(i, j, k, grid, u) / Δx(i, j, k, grid)
@inline ∂xᶠᵃᵃ(i, j, k, grid, c) = δxᶠᵃᵃ(i, j, k, grid, c) / Δx(i, j, k, grid)

@inline ∂yᵃᶜᵃ(i, j, k, grid, v) = δyᵃᶜᵃ(i, j, k, grid, v) / Δy(i, j, k, grid)
@inline ∂yᵃᶠᵃ(i, j, k, grid, c) = δyᵃᶠᵃ(i, j, k, grid, c) / Δy(i, j, k, grid)

@inline ∂zᵃᵃᶜ(i, j, k, grid, w) = δzᵃᵃᶜ(i, j, k, grid, w) / ΔzF(i, j, k, grid)
@inline ∂zᵃᵃᶠ(i, j, k, grid, c) = δzᵃᵃᶠ(i, j, k, grid, c) / ΔzC(i, j, k, grid)

####
#### Derivative operators acting on functions
####

@inline ∂xᶜᵃᵃ(i, j, k, grid, f::F, args...) where F<:Function = (f(i+1, j, k, grid, args...) - f(i,   j, k, grid, args...)) / Δx(i, j, k, grid)
@inline ∂xᶠᵃᵃ(i, j, k, grid, f::F, args...) where F<:Function = (f(i,   j, k, grid, args...) - f(i-1, j, k, grid, args...)) / Δx(i, j, k, grid)

@inline ∂yᵃᶜᵃ(i, j, k, grid, f::F, args...) where F<:Function = (f(i, j+1, k, grid, args...) - f(i, j,   k, grid, args...)) / Δy(i, j, k, grid)
@inline ∂yᵃᶠᵃ(i, j, k, grid, f::F, args...) where F<:Function = (f(i, j,   k, grid, args...) - f(i, j-1, k, grid, args...)) / Δy(i, j, k, grid)

@inline ∂zᵃᵃᶜ(i, j, k, grid, f::F, args...) where F<:Function = (f(i, j, k+1, grid, args...) - f(i, j, k,   grid, args...)) / ΔzF(i, j, k, grid)
@inline ∂zᵃᵃᶠ(i, j, k, grid, f::F, args...) where F<:Function = (f(i, j, k,   grid, args...) - f(i, j, k-1, grid, args...)) / ΔzC(i, j, k, grid)

####
#### Operators of the form A*δ(q) where A is an area and q is some quantity.
####

@inline Ax_∂xᶜᵃᵃ(i, j, k, grid, u) = Axᵃᵃᶠ(i, j, k, grid) * ∂xᶜᵃᵃ(i, j, k, grid, u)
@inline Ax_∂xᶠᵃᵃ(i, j, k, grid, c) = Axᵃᵃᶠ(i, j, k, grid) * ∂xᶠᵃᵃ(i, j, k, grid, c)

@inline Ay_∂yᵃᶜᵃ(i, j, k, grid, v) = Ayᵃᵃᶠ(i, j, k, grid) * ∂yᵃᶜᵃ(i, j, k, grid, v)
@inline Ay_∂yᵃᶠᵃ(i, j, k, grid, c) = Ayᵃᵃᶠ(i, j, k, grid) * ∂yᵃᶠᵃ(i, j, k, grid, c)

@inline Az_∂zᵃᵃᶜ(i, j, k, grid, w) = Azᵃᵃᵃ(i, j, k, grid) * ∂zᵃᵃᶜ(i, j, k, grid, w)
@inline Az_∂zᵃᵃᶠ(i, j, k, grid, c) = Azᵃᵃᵃ(i, j, k, grid) * ∂zᵃᵃᶠ(i, j, k, grid, c)

####
#### Second derivatives
####

@inline ∂²xᶜᵃᵃ(i, j, k, grid, c) = ∂xᶜᵃᵃ(i, j, k, grid, ∂xᶠᵃᵃ, c)
@inline ∂²xᶠᵃᵃ(i, j, k, grid, u) = ∂xᶠᵃᵃ(i, j, k, grid, ∂xᶜᵃᵃ, u)

@inline ∂²yᵃᶜᵃ(i, j, k, grid, c) = ∂yᵃᶜᵃ(i, j, k, grid, ∂yᵃᶠᵃ, c)
@inline ∂²yᵃᶠᵃ(i, j, k, grid, v) = ∂yᵃᶠᵃ(i, j, k, grid, ∂yᵃᶜᵃ, v)

@inline ∂²zᵃᵃᶜ(i, j, k, grid, c) = ∂zᵃᵃᶜ(i, j, k, grid, ∂zᵃᵃᶠ, c)
@inline ∂²zᵃᵃᶠ(i, j, k, grid, w) = ∂zᵃᵃᶠ(i, j, k, grid, ∂zᵃᵃᶜ, w)

@inline ∂²xᶜᵃᵃ(i, j, k, grid, f::F, args...) where F <: Function = ∂xᶜᵃᵃ(i, j, k, grid, ∂xᶠᵃᵃ, f, args...)
@inline ∂²xᶠᵃᵃ(i, j, k, grid, f::F, args...) where F <: Function = ∂xᶠᵃᵃ(i, j, k, grid, ∂xᶜᵃᵃ, f, args...)

@inline ∂²yᵃᶜᵃ(i, j, k, grid, f::F, args...) where F <: Function = ∂yᵃᶜᵃ(i, j, k, grid, ∂yᵃᶠᵃ, f, args...)
@inline ∂²yᵃᶠᵃ(i, j, k, grid, f::F, args...) where F <: Function = ∂yᵃᶠᵃ(i, j, k, grid, ∂yᵃᶜᵃ, f, args...)

@inline ∂²zᵃᵃᶜ(i, j, k, grid, f::F, args...) where F <: Function = ∂zᵃᵃᶜ(i, j, k, grid, ∂zᵃᵃᶠ, f, args...)
@inline ∂²zᵃᵃᶠ(i, j, k, grid, f::F, args...) where F <: Function = ∂zᵃᵃᶠ(i, j, k, grid, ∂zᵃᵃᶜ, f, args...)

####
#### Fourth derivatives
####

@inline ∂⁴xᶜᵃᵃ(i, j, k, grid, c) = ∂²xᶜᵃᵃ(i, j, k, grid, ∂²xᶜᵃᵃ, c)
@inline ∂⁴xᶠᵃᵃ(i, j, k, grid, u) = ∂²xᶠᵃᵃ(i, j, k, grid, ∂²xᶠᵃᵃ, u)

@inline ∂⁴yᵃᶜᵃ(i, j, k, grid, c) = ∂²yᵃᶜᵃ(i, j, k, grid, ∂²yᵃᶜᵃ, c)
@inline ∂⁴yᵃᶠᵃ(i, j, k, grid, v) = ∂²yᵃᶠᵃ(i, j, k, grid, ∂²yᵃᶠᵃ, v)

@inline ∂⁴zᵃᵃᶜ(i, j, k, grid, c) = ∂²zᵃᵃᶜ(i, j, k, grid, ∂²zᵃᵃᶜ, c)
@inline ∂⁴zᵃᵃᶠ(i, j, k, grid, w) = ∂²zᵃᵃᶠ(i, j, k, grid, ∂²zᵃᵃᶠ, w)
