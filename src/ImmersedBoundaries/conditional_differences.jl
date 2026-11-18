using Oceananigans.Operators: Operators, δxTᶠᵃᵃ, δyTᵃᶠᵃ, Δx⁻¹ᶠᶜᶠ, Δy⁻¹ᶜᶠᶠ, ℑxyᶠᶠᵃ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ

# Conditional differences that are "immersed boundary aware".
# Here we return `zero(ibg)` rather than `δx` (for example) when _one_ of the
# nodes involved in the difference is `immersed_inactive_node`.
#
# These functions are used to generate all conditioned difference operators via metaprogramming.
# There are 24 difference operators in all:
# 1 operator for each of 8 locations (ᶠᶜᶜ, ᶠᶜᶠ, ...) × 3 directions (δx, δy, δz)).
#
# The operators depend on the location they end up at. For example, conditional_δx_f is used
# to construct `δxᶠᶜᶜ`, `δxᶠᶜᶠ`, `δxᶠᶠᶜ`, `δxᶠᶠᶠ`, all of which difference values `Center`ed in `x`
# at `Face`.

@inline conditional_δx_f(ℓy, ℓz, i, j, k, ibg::IBG, δx, args...) = ifelse(immersed_inactive_node(i,   j, k, ibg, c, ℓy, ℓz) |
                                                                          immersed_inactive_node(i-1, j, k, ibg, c, ℓy, ℓz),
                                                                          zero(ibg),
                                                                          δx(i, j, k, ibg, args...))

@inline conditional_δx_c(ℓy, ℓz, i, j, k, ibg::IBG, δx, args...) = ifelse(immersed_inactive_node(i,   j, k, ibg, f, ℓy, ℓz) |
                                                                          immersed_inactive_node(i+1, j, k, ibg, f, ℓy, ℓz),
                                                                          zero(ibg),
                                                                          δx(i, j, k, ibg, args...))

@inline conditional_δy_f(ℓx, ℓz, i, j, k, ibg::IBG, δy, args...) = ifelse(immersed_inactive_node(i, j,   k, ibg, ℓx, c, ℓz) |
                                                                          immersed_inactive_node(i, j-1, k, ibg, ℓx, c, ℓz),
                                                                          zero(ibg),
                                                                          δy(i, j, k, ibg, args...))

@inline conditional_δy_c(ℓx, ℓz, i, j, k, ibg::IBG, δy, args...) = ifelse(immersed_inactive_node(i, j,   k, ibg, ℓx, f, ℓz) |
                                                                          immersed_inactive_node(i, j+1, k, ibg, ℓx, f, ℓz),
                                                                          zero(ibg),
                                                                          δy(i, j, k, ibg, args...))

@inline conditional_δz_f(ℓx, ℓy, i, j, k, ibg::IBG, δz, args...) = ifelse(immersed_inactive_node(i, j, k,   ibg, ℓx, ℓy, c) |
                                                                          immersed_inactive_node(i, j, k-1, ibg, ℓx, ℓy, c),
                                                                          zero(ibg),
                                                                          δz(i, j, k, ibg, args...))

@inline conditional_δz_c(ℓx, ℓy, i, j, k, ibg::IBG, δz, args...) = ifelse(immersed_inactive_node(i, j, k,   ibg, ℓx, ℓy, f) |
                                                                          immersed_inactive_node(i, j, k+1, ibg, ℓx, ℓy, f),
                                                                          zero(ibg),
                                                                          δz(i, j, k, ibg, args...))

@inline translate_loc(a) = a == :ᶠ ? :f : :c

for (d, ξ) in enumerate((:x, :y, :z))
    for ℓx in (:ᶠ, :ᶜ), ℓy in (:ᶠ, :ᶜ), ℓz in (:ᶠ, :ᶜ)

        δξ             = Symbol(:δ, ξ, ℓx, ℓy, ℓz)
        loc            = translate_loc.((ℓx, ℓy, ℓz))
        conditional_δξ = Symbol(:conditional_δ, ξ, :_, loc[d])

        if ξ == :x
            δξᵃ = Symbol(:δ, ξ, ℓx, :ᵃ, :ᵃ)
        elseif ξ == :y
            δξᵃ = Symbol(:δ, ξ, :ᵃ, ℓy, :ᵃ)
        else
            δξᵃ = Symbol(:δ, ξ, :ᵃ, :ᵃ, ℓz)
        end

        # `other_locs` contains locations in the two "other" directions not being differenced
        other_locs = if d == 1
            (loc[2], loc[3])
        elseif d == 2
            (loc[1], loc[3])
        else
            (loc[1], loc[2])
        end

        @eval begin
            @inline Operators.$δξ(i, j, k, ibg::IBG, args...)              = $conditional_δξ($(other_locs[1]), $(other_locs[2]), i, j, k, ibg, $δξᵃ, args...)
            @inline Operators.$δξ(i, j, k, ibg::IBG, ϕ::Function, args...) = $conditional_δξ($(other_locs[1]), $(other_locs[2]), i, j, k, ibg, $δξᵃ, ϕ, args...)
       end
    end
end

# Topology-aware immersed boundary derivatives: tracers ensure no-flux on `inactive_node`s, so the derivative
# vanishes when either node is inactive. Otherwise it is the topology-aware derivative evaluated on the immersed
# grid itself, so `f` receives the immersed grid rather than its underlying grid. The topology-aware differences
# `δxTᶠᵃᵃ`, `δxTᶜᵃᵃ` (and their y-counterparts) need no immersed-boundary treatment and dispatch on the
# immersed grid's topology directly.
@inline conditional_∂xTᶠᶜᶠ(i, j, k, ibg::IBG, args...) = ifelse(inactive_node(i, j, k, ibg, c, c, f) | inactive_node(i-1, j, k, ibg, c, c, f), zero(ibg), δxTᶠᵃᵃ(i, j, k, ibg, args...) * Δx⁻¹ᶠᶜᶠ(i, j, k, ibg))
@inline conditional_∂yTᶜᶠᶠ(i, j, k, ibg::IBG, args...) = ifelse(inactive_node(i, j, k, ibg, c, c, f) | inactive_node(i, j-1, k, ibg, c, c, f), zero(ibg), δyTᵃᶠᵃ(i, j, k, ibg, args...) * Δy⁻¹ᶜᶠᶠ(i, j, k, ibg))

# Constant-r versions for free surface derivatives
@inline conditional_∂xᵣTᶠᶜᶠ(i, j, k, ibg::IBG, args...) = ifelse(inactive_node(i, j, k, ibg, c, c, f) | inactive_node(i-1, j, k, ibg, c, c, f), zero(ibg), δxTᶠᵃᵃ(i, j, k, ibg, args...) * Δx⁻¹ᶠᶜᶠ(i, j, k, ibg))
@inline conditional_∂yᵣTᶜᶠᶠ(i, j, k, ibg::IBG, args...) = ifelse(inactive_node(i, j, k, ibg, c, c, f) | inactive_node(i, j-1, k, ibg, c, c, f), zero(ibg), δyTᵃᶠᵃ(i, j, k, ibg, args...) * Δy⁻¹ᶜᶠᶠ(i, j, k, ibg))

@inline Operators.∂xTᶠᶜᶠ(i, j, k, ibg::IBG, f, args...) = conditional_∂xTᶠᶜᶠ(i, j, k, ibg, f, args...)
@inline Operators.∂yTᶜᶠᶠ(i, j, k, ibg::IBG, f, args...) = conditional_∂yTᶜᶠᶠ(i, j, k, ibg, f, args...)

@inline Operators.∂xTᶠᶜᶠ(i, j, k, ibg::IBG, w::AbstractArray) = conditional_∂xTᶠᶜᶠ(i, j, k, ibg, w)
@inline Operators.∂yTᶜᶠᶠ(i, j, k, ibg::IBG, w::AbstractArray) = conditional_∂yTᶜᶠᶠ(i, j, k, ibg, w)

# Constant-r versions for free surface derivatives on immersed boundaries
@inline Operators.∂xᵣTᶠᶜᶠ(i, j, k, ibg::IBG, f, args...) = conditional_∂xᵣTᶠᶜᶠ(i, j, k, ibg, f, args...)
@inline Operators.∂yᵣTᶜᶠᶠ(i, j, k, ibg::IBG, f, args...) = conditional_∂yᵣTᶜᶠᶠ(i, j, k, ibg, f, args...)

@inline Operators.∂xᵣTᶠᶜᶠ(i, j, k, ibg::IBG, η::AbstractArray) = conditional_∂xᵣTᶠᶜᶠ(i, j, k, ibg, η)
@inline Operators.∂yᵣTᶜᶠᶠ(i, j, k, ibg::IBG, η::AbstractArray) = conditional_∂yᵣTᶜᶠᶠ(i, j, k, ibg, η)
