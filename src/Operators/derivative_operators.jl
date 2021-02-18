#####
##### Rectilinear derivative operators
#####

@inline ∂xᶜᵃᵃ(i, j, k, grid, u, args...) = δxᶜᵃᵃ(i, j, k, grid, u, args...) / Δx(i, j, k, grid)
@inline ∂xᶠᵃᵃ(i, j, k, grid, c, args...) = δxᶠᵃᵃ(i, j, k, grid, c, args...) / Δx(i, j, k, grid)

@inline ∂yᵃᶜᵃ(i, j, k, grid, v, args...) = δyᵃᶜᵃ(i, j, k, grid, v, args...) / Δy(i, j, k, grid)
@inline ∂yᵃᶠᵃ(i, j, k, grid, c, args...) = δyᵃᶠᵃ(i, j, k, grid, c, args...) / Δy(i, j, k, grid)

@inline ∂zᵃᵃᶜ(i, j, k, grid, w, args...) = δzᵃᵃᶜ(i, j, k, grid, w, args...) / ΔzF(i, j, k, grid)
@inline ∂zᵃᵃᶠ(i, j, k, grid, c, args...) = δzᵃᵃᶠ(i, j, k, grid, c, args...) / ΔzC(i, j, k, grid)

#####
##### Operators of the form A*δ(q) where A is an area and q is some quantity.
#####

@inline Ax_∂xᶜᵃᵃ(i, j, k, grid, u) = Axᵃᵃᶠ(i, j, k, grid) * ∂xᶜᵃᵃ(i, j, k, grid, u)
@inline Ax_∂xᶠᵃᵃ(i, j, k, grid, c) = Axᵃᵃᶠ(i, j, k, grid) * ∂xᶠᵃᵃ(i, j, k, grid, c)

@inline Ay_∂yᵃᶜᵃ(i, j, k, grid, v) = Ayᵃᵃᶠ(i, j, k, grid) * ∂yᵃᶜᵃ(i, j, k, grid, v)
@inline Ay_∂yᵃᶠᵃ(i, j, k, grid, c) = Ayᵃᵃᶠ(i, j, k, grid) * ∂yᵃᶠᵃ(i, j, k, grid, c)

@inline Az_∂zᵃᵃᶜ(i, j, k, grid, w) = Azᵃᵃᵃ(i, j, k, grid) * ∂zᵃᵃᶜ(i, j, k, grid, w)
@inline Az_∂zᵃᵃᶠ(i, j, k, grid, c) = Azᵃᵃᵃ(i, j, k, grid) * ∂zᵃᵃᶠ(i, j, k, grid, c)

#####
##### Second derivatives
#####

@inline ∂²xᶜᵃᵃ(i, j, k, grid, c, args...) = ∂xᶜᵃᵃ(i, j, k, grid, ∂xᶠᵃᵃ, c, args...)
@inline ∂²xᶠᵃᵃ(i, j, k, grid, u, args...) = ∂xᶠᵃᵃ(i, j, k, grid, ∂xᶜᵃᵃ, u, args...)

@inline ∂²yᵃᶜᵃ(i, j, k, grid, c, args...) = ∂yᵃᶜᵃ(i, j, k, grid, ∂yᵃᶠᵃ, c, args...)
@inline ∂²yᵃᶠᵃ(i, j, k, grid, v, args...) = ∂yᵃᶠᵃ(i, j, k, grid, ∂yᵃᶜᵃ, v, args...)

@inline ∂²zᵃᵃᶜ(i, j, k, grid, c, args...) = ∂zᵃᵃᶜ(i, j, k, grid, ∂zᵃᵃᶠ, c, args...)
@inline ∂²zᵃᵃᶠ(i, j, k, grid, w, args...) = ∂zᵃᵃᶠ(i, j, k, grid, ∂zᵃᵃᶜ, w, args...)

#####
##### Fourth derivatives
#####

@inline ∂⁴xᶜᵃᵃ(i, j, k, grid, c, args...) = ∂²xᶜᵃᵃ(i, j, k, grid, ∂²xᶜᵃᵃ, c, args...)
@inline ∂⁴xᶠᵃᵃ(i, j, k, grid, u, args...) = ∂²xᶠᵃᵃ(i, j, k, grid, ∂²xᶠᵃᵃ, u, args...)

@inline ∂⁴yᵃᶜᵃ(i, j, k, grid, c, args...) = ∂²yᵃᶜᵃ(i, j, k, grid, ∂²yᵃᶜᵃ, c, args...)
@inline ∂⁴yᵃᶠᵃ(i, j, k, grid, v, args...) = ∂²yᵃᶠᵃ(i, j, k, grid, ∂²yᵃᶠᵃ, v, args...)

@inline ∂⁴zᵃᵃᶜ(i, j, k, grid, c, args...) = ∂²zᵃᵃᶜ(i, j, k, grid, ∂²zᵃᵃᶜ, c, args...)
@inline ∂⁴zᵃᵃᶠ(i, j, k, grid, w, args...) = ∂²zᵃᵃᶠ(i, j, k, grid, ∂²zᵃᵃᶠ, w, args...)

#####
##### Horizontally curvilinear derivative operators
#####

@inline ∂xᶜᶜᵃ(i, j, k, grid, u, args...) = δxᶜᵃᵃ(i, j, k, grid, u, args...) / Δxᶜᶜᵃ(i, j, k, grid)
@inline ∂xᶜᶠᵃ(i, j, k, grid, ζ, args...) = δxᶜᵃᵃ(i, j, k, grid, ζ, args...) / Δxᶜᶠᵃ(i, j, k, grid)

@inline ∂xᶠᶠᵃ(i, j, k, grid, v, args...) = δxᶠᵃᵃ(i, j, k, grid, v, args...) / Δxᶠᶠᵃ(i, j, k, grid)
@inline ∂xᶠᶜᵃ(i, j, k, grid, c, args...) = δxᶠᵃᵃ(i, j, k, grid, c, args...) / Δxᶠᶜᵃ(i, j, k, grid)

@inline ∂yᶜᶜᵃ(i, j, k, grid, v, args...) = δyᵃᶜᵃ(i, j, k, grid, v, args...) / Δyᶜᶜᵃ(i, j, k, grid)
@inline ∂yᶠᶜᵃ(i, j, k, grid, ζ, args...) = δyᵃᶜᵃ(i, j, k, grid, ζ, args...) / Δyᶠᶜᵃ(i, j, k, grid)

@inline ∂yᶠᶠᵃ(i, j, k, grid, u, args...) = δyᵃᶠᵃ(i, j, k, grid, u, args...) / Δyᶠᶠᵃ(i, j, k, grid)
@inline ∂yᶜᶠᵃ(i, j, k, grid, c, args...) = δyᵃᶠᵃ(i, j, k, grid, c, args...) / Δyᶜᶠᵃ(i, j, k, grid)
