####
#### Derivative operators
####

@inline ∂xᶜᵃᵃ(i, j, k, grid, u) = δxᶜᵃᵃ(i, j, k, grid, u) / Δx(i, j, k, grid)
@inline ∂xᶠᵃᵃ(i, j, k, grid, c) = δxᶠᵃᵃ(i, j, k, grid, c) / Δx(i, j, k, grid)

@inline ∂yᵃᶜᵃ(i, j, k, grid, v) = δyᵃᶜᵃ(i, j, k, grid, u) / Δy(i, j, k, grid)
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
#### Second derivatives
####

@inline ∂²xᶜᵃᵃ(i, j, k, grid, c) = ∂xᶜᵃᵃ(i, j, k, grid, ∂xᶠᵃᵃ, c)
@inline ∂²xᶠᵃᵃ(i, j, k, grid, u) = ∂xᶠᵃᵃ(i, j, k, grid, ∂xᶜᵃᵃ, u)

@inline ∂²yᵃᶜᵃ(i, j, k, grid, c) = ∂yᵃᶜᵃ(i, j, k, grid, ∂yᵃᶠᵃ, c)
@inline ∂²yᵃᶠᵃ(i, j, k, grid, v) = ∂yᵃᶠᵃ(i, j, k, grid, ∂yᵃᶜᵃ, v)

@inline ∂²zᵃᵃᶜ(i, j, k, grid, c) = ∂zᵃᵃᶜ(i, j, k, grid, ∂zᵃᵃᶠ, c)
@inline ∂²zᵃᵃᶠ(i, j, k, grid, w) = ∂zᵃᵃᶠ(i, j, k, grid, ∂zᵃᵃᶜ, w)
