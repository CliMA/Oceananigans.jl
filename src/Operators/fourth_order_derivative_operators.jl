#####
##### Fourth-order-accurate first derivatives in flux form
#####
##### `fourth_order_δxᶠᵃᵃ` differences a field centered in x to x-faces,
#####
#####     fourth_order_δxᶠᵃᵃ(i, j, k, grid, φ) = (15 (φᵢ - φᵢ₋₁) - (φᵢ₊₁ - φᵢ₋₂)) / 12 ,
#####
##### and `fourth_order_δxᶜᵃᵃ` differences a field on x-faces to centers. Divided by Δx, this is the
##### fourth-order-accurate first derivative *in flux form*: the flux difference (Fᵢ₊₁ - Fᵢ) / Δx of
##### F = fourth_order_δx φ / Δx is the fourth-order five-point Laplacian
#####
#####     (- φᵢ₋₂ + 16 φᵢ₋₁ - 30 φᵢ + 16 φᵢ₊₁ - φᵢ₊₂) / 12Δx² .
#####
##### (The fourth-order-accurate *pointwise* staggered derivative (27 (φᵢ - φᵢ₋₁) - (φᵢ₊₁ - φᵢ₋₂)) / 24Δx
##### gives a Laplacian that is only second-order accurate when differenced by the second-order flux
##### divergence.)
#####
##### The location-named derivatives, e.g. `fourth_order_∂xᵣᶠᶜᶜ`, use the four-point stencil only where it is
##### valid: in regularly spaced, non-`Flat` directions (see `fourth_order_x_stencil`), and where none of the
##### four nodes of the stencil is an `inactive_node` (halos of `Bounded` directions and the interior of immersed
##### boundaries). Elsewhere they return the second-order derivative `∂xᵣ`, `∂yᵣ`, or `∂z` at the same location.
##### The names follow the second-order fallback: on grids where the fourth-order stencil is used, ∂xᵣ = ∂x.
#####

using Oceananigans.Grids: inactive_node, topology, Flat

#####
##### Differences
#####

@inline fourth_order_δxᶠᵃᵃ(i, j, k, grid, φ) = @inbounds (15 * (φ[i, j, k] - φ[i-1, j, k]) - (φ[i+1, j, k] - φ[i-2, j, k])) / 12
@inline fourth_order_δxᶜᵃᵃ(i, j, k, grid, φ) = @inbounds (15 * (φ[i+1, j, k] - φ[i, j, k]) - (φ[i+2, j, k] - φ[i-1, j, k])) / 12
@inline fourth_order_δyᵃᶠᵃ(i, j, k, grid, φ) = @inbounds (15 * (φ[i, j, k] - φ[i, j-1, k]) - (φ[i, j+1, k] - φ[i, j-2, k])) / 12
@inline fourth_order_δyᵃᶜᵃ(i, j, k, grid, φ) = @inbounds (15 * (φ[i, j+1, k] - φ[i, j, k]) - (φ[i, j+2, k] - φ[i, j-1, k])) / 12
@inline fourth_order_δzᵃᵃᶠ(i, j, k, grid, φ) = @inbounds (15 * (φ[i, j, k] - φ[i, j, k-1]) - (φ[i, j, k+1] - φ[i, j, k-2])) / 12
@inline fourth_order_δzᵃᵃᶜ(i, j, k, grid, φ) = @inbounds (15 * (φ[i, j, k+1] - φ[i, j, k]) - (φ[i, j, k+2] - φ[i, j, k-1])) / 12

@inline fourth_order_δxᶠᵃᵃ(i, j, k, grid, f::Function, args...) = (15 * (f(i, j, k, grid, args...) - f(i-1, j, k, grid, args...)) - (f(i+1, j, k, grid, args...) - f(i-2, j, k, grid, args...))) / 12
@inline fourth_order_δxᶜᵃᵃ(i, j, k, grid, f::Function, args...) = (15 * (f(i+1, j, k, grid, args...) - f(i, j, k, grid, args...)) - (f(i+2, j, k, grid, args...) - f(i-1, j, k, grid, args...))) / 12
@inline fourth_order_δyᵃᶠᵃ(i, j, k, grid, f::Function, args...) = (15 * (f(i, j, k, grid, args...) - f(i, j-1, k, grid, args...)) - (f(i, j+1, k, grid, args...) - f(i, j-2, k, grid, args...))) / 12
@inline fourth_order_δyᵃᶜᵃ(i, j, k, grid, f::Function, args...) = (15 * (f(i, j+1, k, grid, args...) - f(i, j, k, grid, args...)) - (f(i, j+2, k, grid, args...) - f(i, j-1, k, grid, args...))) / 12
@inline fourth_order_δzᵃᵃᶠ(i, j, k, grid, f::Function, args...) = (15 * (f(i, j, k, grid, args...) - f(i, j, k-1, grid, args...)) - (f(i, j, k+1, grid, args...) - f(i, j, k-2, grid, args...))) / 12
@inline fourth_order_δzᵃᵃᶜ(i, j, k, grid, f::Function, args...) = (15 * (f(i, j, k+1, grid, args...) - f(i, j, k, grid, args...)) - (f(i, j, k+2, grid, args...) - f(i, j, k-1, grid, args...))) / 12

export fourth_order_δxᶠᵃᵃ, fourth_order_δxᶜᵃᵃ, fourth_order_δyᵃᶠᵃ, fourth_order_δyᵃᶜᵃ, fourth_order_δzᵃᵃᶠ, fourth_order_δzᵃᵃᶜ

#####
##### Whether the four-point stencil may be used in a direction: `Val(true)` for regularly spaced,
##### non-`Flat` directions of a `RectilinearGrid`. Extended by `ImmersedBoundaries`.
#####

"""
    fourth_order_x_stencil(grid)

Return `Val(true)` if the fourth-order-accurate first derivatives (e.g. [`fourth_order_∂xᵣᶠᶜᶜ`](@ref))
use their four-point stencil in the x-direction of `grid`, and `Val(false)` if they always fall back
to second order. Fourth-order stencils require regular spacing.
"""
@inline fourth_order_x_stencil(grid) = Val(false)
@inline fourth_order_y_stencil(grid) = Val(false)
@inline fourth_order_z_stencil(grid) = Val(false)

@inline fourth_order_x_stencil(grid::RGX) = nonflat_stencil(topology(grid, 1))
@inline fourth_order_y_stencil(grid::RGY) = nonflat_stencil(topology(grid, 2))
@inline fourth_order_z_stencil(grid::RGZ) = nonflat_stencil(topology(grid, 3))

@inline nonflat_stencil(topology) = Val(true)
@inline nonflat_stencil(::Type{Flat}) = Val(false)

# Whether none of the four nodes i₁, ..., i₁ + 3 at the location (ℓx, ℓy, ℓz) is inactive.
# `inactive_node` is identically `false` in `Periodic` directions of non-immersed grids.
@inline active_x_stencil(i₁, j, k, grid, ℓx, ℓy, ℓz) =
    !(inactive_node(i₁,   j, k, grid, ℓx, ℓy, ℓz) | inactive_node(i₁+1, j, k, grid, ℓx, ℓy, ℓz) |
      inactive_node(i₁+2, j, k, grid, ℓx, ℓy, ℓz) | inactive_node(i₁+3, j, k, grid, ℓx, ℓy, ℓz))

@inline active_y_stencil(i, j₁, k, grid, ℓx, ℓy, ℓz) =
    !(inactive_node(i, j₁,   k, grid, ℓx, ℓy, ℓz) | inactive_node(i, j₁+1, k, grid, ℓx, ℓy, ℓz) |
      inactive_node(i, j₁+2, k, grid, ℓx, ℓy, ℓz) | inactive_node(i, j₁+3, k, grid, ℓx, ℓy, ℓz))

@inline active_z_stencil(i, j, k₁, grid, ℓx, ℓy, ℓz) =
    !(inactive_node(i, j, k₁,   grid, ℓx, ℓy, ℓz) | inactive_node(i, j, k₁+1, grid, ℓx, ℓy, ℓz) |
      inactive_node(i, j, k₁+2, grid, ℓx, ℓy, ℓz) | inactive_node(i, j, k₁+3, grid, ℓx, ℓy, ℓz))

#####
##### Location-named derivatives with second-order fallback
#####

location_instance(ℓ) = ℓ == :ᶜ ? :c : :f
flip_symbol(ℓ) = ℓ == :ᶜ ? :ᶠ : :ᶜ

for dir in (:x, :y, :z), ℓ1 in (:ᶜ, :ᶠ), ℓ2 in (:ᶜ, :ᶠ), ℓ3 in (:ᶜ, :ᶠ)

    # Location of the derivative: ℓ1 in the differentiated direction, ℓ2 and ℓ3 in the others
    loc = insert_symbol(dir, ℓ1, ℓ2, ℓ3)

    # Location of the differenced field
    ψloc = insert_symbol(dir, flip_symbol(ℓ1), ℓ2, ℓ3)
    ℓx, ℓy, ℓz = location_instance.(ψloc)

    ξ = dir == :z ? :∂z : Symbol(:∂, dir, :ᵣ)
    second_order = Symbol(ξ, loc...)
    fourth_order = Symbol(:fourth_order_, ξ, loc...)

    difference = dir == :x ? Symbol(:fourth_order_δx, ℓ1, :ᵃ, :ᵃ) :
                 dir == :y ? Symbol(:fourth_order_δy, :ᵃ, ℓ1, :ᵃ) :
                             Symbol(:fourth_order_δz, :ᵃ, :ᵃ, ℓ1)

    rcp_spacing    = Symbol(:Δ, dir, :⁻¹, loc...)
    stencil        = Symbol(:fourth_order_, dir, :_stencil)
    active_stencil = Symbol(:active_, dir, :_stencil)

    # First index of the stencil: i-2:i+1 for derivatives on faces, i-1:i+2 for derivatives at centers
    offset = ℓ1 == :ᶠ ? -2 : -1
    first_node = dir == :x ? :(i + $offset, j, k) : dir == :y ? :(i, j + $offset, k) : :(i, j, k + $offset)

    docstring = """
        $fourth_order(i, j, k, grid, φ)

    Fourth-order-accurate first derivative of `φ` in the $dir-direction, in flux form
    (see [`$difference`](@ref)), at the location `$(join(loc))`. Falls back to `$second_order` in
    directions that are not regularly spaced, and where the four-point stencil includes an inactive node.
    """

    @eval begin
        @doc $docstring
        @inline $fourth_order(i, j, k, grid, φ) = $fourth_order(i, j, k, grid, $stencil(grid), φ)
        @inline $fourth_order(i, j, k, grid, ::Val{false}, φ) = $second_order(i, j, k, grid, φ)

        @inline function $fourth_order(i, j, k, grid, ::Val{true}, φ)
            ∂φ⁽⁴⁾ = $difference(i, j, k, grid, φ) * $rcp_spacing(i, j, k, grid)
            ∂φ⁽²⁾ = $second_order(i, j, k, grid, φ)
            return ifelse($active_stencil($(first_node.args...), grid, $ℓx, $ℓy, $ℓz), ∂φ⁽⁴⁾, ∂φ⁽²⁾)
        end

        export $fourth_order
    end
end
