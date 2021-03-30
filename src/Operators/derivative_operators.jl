#####
##### Rectilinear derivative operators
#####

@inline ∂xᶜᵃᵃ(i, j, k, grid::ARG, u) = δxᶜᵃᵃ(i, j, k, grid, u) / Δx(i, j, k, grid)
@inline ∂xᶠᵃᵃ(i, j, k, grid::ARG, c) = δxᶠᵃᵃ(i, j, k, grid, c) / Δx(i, j, k, grid)

@inline ∂yᵃᶜᵃ(i, j, k, grid::ARG, v) = δyᵃᶜᵃ(i, j, k, grid, v) / Δy(i, j, k, grid)
@inline ∂yᵃᶠᵃ(i, j, k, grid::ARG, c) = δyᵃᶠᵃ(i, j, k, grid, c) / Δy(i, j, k, grid)

@inline ∂zᵃᵃᶜ(i, j, k, grid::Union{ARG, AHCG}, w) = δzᵃᵃᶜ(i, j, k, grid, w) / Δzᵃᵃᶜ(i, j, k, grid)
@inline ∂zᵃᵃᶠ(i, j, k, grid::Union{ARG, AHCG}, c) = δzᵃᵃᶠ(i, j, k, grid, c) / Δzᵃᵃᶠ(i, j, k, grid)


@inline ∂xᶜᵃᵃ(i, j, k, grid::ARG, f::F, args...) where F<:Function = δxᶜᵃᵃ(i, j, k, grid, f, args...) / Δx(i, j, k, grid)
@inline ∂xᶠᵃᵃ(i, j, k, grid::ARG, f::F, args...) where F<:Function = δxᶠᵃᵃ(i, j, k, grid, f, args...) / Δx(i, j, k, grid)
### @inline ∂xᶠᵃᵃ(i, j, k, grid::Union{ARG, AHCG}, f::F, args...) where F<:Function = δxᶠᵃᵃ(i, j, k, grid, f, args...) / Δxᶠᵃᵃ(i, j, k, grid)
@inline ∂xᶠᵃᵃ(i, j, k, grid::Union{ARG, AHCG}, c) = δxᶠᵃᵃ(i, j, k, grid, c) / Δxᶠᵃᵃ(i, j, k, grid)

@inline ∂yᵃᶜᵃ(i, j, k, grid::ARG, f::F, args...) where F<:Function = δyᵃᶜᵃ(i, j, k, grid, f, args...) / Δy(i, j, k, grid)
@inline ∂yᵃᶠᵃ(i, j, k, grid::ARG, f::F, args...) where F<:Function = δyᵃᶠᵃ(i, j, k, grid, f, args...) / Δy(i, j, k, grid)
### @inline ∂yᵃᶠᵃ(i, j, k, grid::Union{ARG, AHCG}, f::F, args...) where F<:Function = δyᵃᶠᵃ(i, j, k, grid, f, args...) / Δyᵃᶠᵃ(i, j, k, grid)
@inline ∂yᵃᶠᵃ(i, j, k, grid::Union{ARG, AHCG}, c) = δyᵃᶠᵃ(i, j, k, grid, c) / Δyᵃᶠᵃ(i, j, k, grid)

@inline ∂zᵃᵃᶜ(i, j, k, grid::Union{ARG, AHCG}, f::F, args...) where F<:Function = δzᵃᵃᶜ(i, j, k, grid, f, args...) / Δzᵃᵃᶜ(i, j, k, grid)
@inline ∂zᵃᵃᶠ(i, j, k, grid::Union{ARG, AHCG}, f::F, args...) where F<:Function = δzᵃᵃᶠ(i, j, k, grid, f, args...) / Δzᵃᵃᶠ(i, j, k, grid)

#####
##### Operators of the form A*δ(q) where A is an area and q is some quantity.
#####

@inline Ax_∂xᶜᵃᵃ(i, j, k, grid::ARG, u) = Axᵃᵃᶠ(i, j, k, grid) * ∂xᶜᵃᵃ(i, j, k, grid, u)
@inline Ax_∂xᶠᵃᵃ(i, j, k, grid::ARG, c) = Axᵃᵃᶠ(i, j, k, grid) * ∂xᶠᵃᵃ(i, j, k, grid, c)

@inline Ay_∂yᵃᶜᵃ(i, j, k, grid::ARG, v) = Ayᵃᵃᶠ(i, j, k, grid) * ∂yᵃᶜᵃ(i, j, k, grid, v)
@inline Ay_∂yᵃᶠᵃ(i, j, k, grid::ARG, c) = Ayᵃᵃᶠ(i, j, k, grid) * ∂yᵃᶠᵃ(i, j, k, grid, c)

@inline Az_∂zᵃᵃᶜ(i, j, k, grid::ARG, w) = Azᵃᵃᵃ(i, j, k, grid) * ∂zᵃᵃᶜ(i, j, k, grid, w)
@inline Az_∂zᵃᵃᶠ(i, j, k, grid::ARG, c) = Azᵃᵃᵃ(i, j, k, grid) * ∂zᵃᵃᶠ(i, j, k, grid, c)

#####
##### Second derivatives
#####

@inline ∂²xᶜᵃᵃ(i, j, k, grid::ARG, c) = ∂xᶜᵃᵃ(i, j, k, grid, ∂xᶠᵃᵃ, c)
@inline ∂²xᶠᵃᵃ(i, j, k, grid::ARG, u) = ∂xᶠᵃᵃ(i, j, k, grid, ∂xᶜᵃᵃ, u)

@inline ∂²yᵃᶜᵃ(i, j, k, grid::ARG, c) = ∂yᵃᶜᵃ(i, j, k, grid, ∂yᵃᶠᵃ, c)
@inline ∂²yᵃᶠᵃ(i, j, k, grid::ARG, v) = ∂yᵃᶠᵃ(i, j, k, grid, ∂yᵃᶜᵃ, v)

@inline ∂²zᵃᵃᶜ(i, j, k, grid::Union{ARG, AHCG}, c) = ∂zᵃᵃᶜ(i, j, k, grid, ∂zᵃᵃᶠ, c)
@inline ∂²zᵃᵃᶠ(i, j, k, grid::Union{ARG, AHCG}, w) = ∂zᵃᵃᶠ(i, j, k, grid, ∂zᵃᵃᶜ, w)


@inline ∂²xᶜᵃᵃ(i, j, k, grid::ARG, f::F, args...) where F<:Function = ∂xᶜᵃᵃ(i, j, k, grid, ∂xᶠᵃᵃ, f, args...)
@inline ∂²xᶠᵃᵃ(i, j, k, grid::ARG, f::F, args...) where F<:Function = ∂xᶠᵃᵃ(i, j, k, grid, ∂xᶜᵃᵃ, f, args...)

@inline ∂²yᵃᶜᵃ(i, j, k, grid::ARG, f::F, args...) where F<:Function = ∂yᵃᶜᵃ(i, j, k, grid, ∂yᵃᶠᵃ, f, args...)
@inline ∂²yᵃᶠᵃ(i, j, k, grid::ARG, f::F, args...) where F<:Function = ∂yᵃᶠᵃ(i, j, k, grid, ∂yᵃᶜᵃ, f, args...)

@inline ∂²zᵃᵃᶜ(i, j, k, grid::Union{ARG, AHCG}, f::F, args...) where F<:Function = ∂zᵃᵃᶜ(i, j, k, grid, ∂zᵃᵃᶠ, f, args...)
@inline ∂²zᵃᵃᶠ(i, j, k, grid::Union{ARG, AHCG}, f::F, args...) where F<:Function = ∂zᵃᵃᶠ(i, j, k, grid, ∂zᵃᵃᶜ, f, args...)

#####
##### Fourth derivatives
#####

@inline ∂⁴xᶜᵃᵃ(i, j, k, grid::ARG, c, args...) = ∂²xᶜᵃᵃ(i, j, k, grid, ∂²xᶜᵃᵃ, c, args...)
@inline ∂⁴xᶠᵃᵃ(i, j, k, grid::ARG, u, args...) = ∂²xᶠᵃᵃ(i, j, k, grid, ∂²xᶠᵃᵃ, u, args...)

@inline ∂⁴yᵃᶜᵃ(i, j, k, grid::ARG, c, args...) = ∂²yᵃᶜᵃ(i, j, k, grid, ∂²yᵃᶜᵃ, c, args...)
@inline ∂⁴yᵃᶠᵃ(i, j, k, grid::ARG, v, args...) = ∂²yᵃᶠᵃ(i, j, k, grid, ∂²yᵃᶠᵃ, v, args...)

@inline ∂⁴zᵃᵃᶜ(i, j, k, grid::Union{ARG, AHCG}, c, args...) = ∂²zᵃᵃᶜ(i, j, k, grid, ∂²zᵃᵃᶜ, c, args...)
@inline ∂⁴zᵃᵃᶠ(i, j, k, grid::Union{ARG, AHCG}, w, args...) = ∂²zᵃᵃᶠ(i, j, k, grid, ∂²zᵃᵃᶠ, w, args...)

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
