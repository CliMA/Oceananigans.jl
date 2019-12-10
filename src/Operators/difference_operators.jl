####
#### Base difference operators
####

@inline δxᶜᵃᵃ(i, j, k, grid, u) = @inbounds u[i+1, j, k] - u[i,   j, k]
@inline δxᶠᵃᵃ(i, j, k, grid, c) = @inbounds c[i,   j, k] - c[i-1, j, k]

@inline δyᵃᶜᵃ(i, j, k, grid, v) = @inbounds v[i, j+1, k] - v[i, j,   k]
@inline δyᵃᶠᵃ(i, j, k, grid, c) = @inbounds c[i, j,   k] - c[i, j-1, k]

@inline δzᵃᵃᶜ(i, j, k, grid, w) = @inbounds w[i, j, k+1] - w[i, j,   k]
@inline δzᵃᵃᶠ(i, j, k, grid, c) = @inbounds c[i, j,   k] - c[i, j, k-1]

####
#### Difference operators acting on functions
####

@inline δxᶜᵃᵃ(i, j, k, grid, f::F, args...) where F<:Function = f(i+1, j, k, grid, args...) - f(i,   j, k, grid, args...)
@inline δxᶠᵃᵃ(i, j, k, grid, f::F, args...) where F<:Function = f(i,   j, k, grid, args...) - f(i-1, j, k, grid, args...)

@inline δyᵃᶜᵃ(i, j, k, grid, f::F, args...) where F<:Function = f(i, j+1, k, grid, args...) - f(i, j,   k, grid, args...)
@inline δyᵃᶠᵃ(i, j, k, grid, f::F, args...) where F<:Function = f(i, j,   k, grid, args...) - f(i, j-1, k, grid, args...)

@inline δzᵃᵃᶜ(i, j, k, grid, f::F, args...) where F<:Function = f(i, j, k+1, grid, args...) - f(i, j, k,   grid, args...)
@inline δzᵃᵃᶠ(i, j, k, grid, f::F, args...) where F<:Function = f(i, j, k,   grid, args...) - f(i, j, k-1, grid, args...)

####
#### Operators that calculate A*q where A is an area and q is some quantity.
#### Useful for defining flux difference operators and other flux operators.
####

@inline Ax_ψᵃᵃᶠ(i, j, k, grid, u) = @inbounds Axᵃᵃᶠ(i, j, k, grid) * u[i, j, k]
@inline Ax_ψᵃᵃᶜ(i, j, k, grid, c) = @inbounds Axᵃᵃᶜ(i, j, k, grid) * c[i, j, k]
@inline Ay_ψᵃᵃᶠ(i, j, k, grid, v) = @inbounds Ayᵃᵃᶠ(i, j, k, grid) * v[i, j, k]
@inline Ay_ψᵃᵃᶜ(i, j, k, grid, c) = @inbounds Ayᵃᵃᶜ(i, j, k, grid) * c[i, j, k]
@inline Az_ψᵃᵃᵃ(i, j, k, grid, c) = @inbounds Azᵃᵃᵃ(i, j, k, grid) * c[i, j, k]

####
#### "Flux difference" operators of the form δ(A*f) where A is an area and f is an array.
####

@inline δᴶxᶜᵃᶜ(i, j, k, grid, c) = δxᶜᵃᵃ(i, j, k, grid, Ax_ψᵃᵃᶜ, c)
@inline δᴶxᶜᵃᶠ(i, j, k, grid, c) = δxᶜᵃᵃ(i, j, k, grid, Ax_ψᵃᵃᶠ, c)
@inline δᴶxᶠᵃᶜ(i, j, k, grid, c) = δxᶠᵃᵃ(i, j, k, grid, Ax_ψᵃᵃᶜ, c)
@inline δᴶxᶠᵃᶠ(i, j, k, grid, c) = δxᶠᵃᵃ(i, j, k, grid, Ax_ψᵃᵃᶠ, c)
@inline δᴶyᵃᶜᶜ(i, j, k, grid, c) = δyᵃᶜᵃ(i, j, k, grid, Ay_ψᵃᵃᶜ, c)
@inline δᴶyᵃᶜᶠ(i, j, k, grid, c) = δyᵃᶜᵃ(i, j, k, grid, Ay_ψᵃᵃᶠ, c)
@inline δᴶyᵃᶠᶜ(i, j, k, grid, c) = δyᵃᶠᵃ(i, j, k, grid, Ay_ψᵃᵃᶜ, c)
@inline δᴶyᵃᶠᶠ(i, j, k, grid, c) = δyᵃᶠᵃ(i, j, k, grid, Ay_ψᵃᵃᶠ, c)
@inline δᴶzᵃᵃᶜ(i, j, k, grid, c) = δzᵃᵃᶜ(i, j, k, grid, Az_ψᵃᵃᵃ, c)
@inline δᴶzᵃᵃᶠ(i, j, k, grid, c) = δzᵃᵃᶠ(i, j, k, grid, Az_ψᵃᵃᵃ, c)
