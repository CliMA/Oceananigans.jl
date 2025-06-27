import Oceananigans.Operators:
    δxᶠᶜᶜ, δxᶠᶜᶠ, δxᶠᶠᶜ, δxᶠᶠᶠ,
    δxᶜᶜᶜ, δxᶜᶜᶠ, δxᶜᶠᶜ, δxᶜᶠᶠ,
    δyᶜᶠᶜ, δyᶜᶠᶠ, δyᶠᶠᶜ, δyᶠᶠᶠ,
    δyᶜᶜᶜ, δyᶜᶜᶠ, δyᶠᶜᶜ, δyᶠᶜᶠ,
    δzᶜᶜᶠ, δzᶜᶠᶠ, δzᶠᶜᶠ, δzᶠᶠᶠ,
    δzᶜᶜᶜ, δzᶜᶠᶜ, δzᶠᶜᶜ, δzᶠᶠᶜ

import Oceananigans.Operators:
    δxTᶜᵃᵃ, δyTᵃᶜᵃ,
    ∂xTᶠᶜᶠ, ∂yTᶜᶠᶠ

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
                                                                          δx(i, j, k, ibg.underlying_grid, args...))

@inline conditional_δx_c(ℓy, ℓz, i, j, k, ibg::IBG, δx, args...) = ifelse(immersed_inactive_node(i,   j, k, ibg, f, ℓy, ℓz) |
                                                                          immersed_inactive_node(i+1, j, k, ibg, f, ℓy, ℓz),
                                                                          zero(ibg),
                                                                          δx(i, j, k, ibg.underlying_grid, args...))

@inline conditional_δy_f(ℓx, ℓz, i, j, k, ibg::IBG, δy, args...) = ifelse(immersed_inactive_node(i, j,   k, ibg, ℓx, c, ℓz) |
                                                                          immersed_inactive_node(i, j-1, k, ibg, ℓx, c, ℓz),
                                                                          zero(ibg),
                                                                          δy(i, j, k, ibg.underlying_grid, args...))

@inline conditional_δy_c(ℓx, ℓz, i, j, k, ibg::IBG, δy, args...) = ifelse(immersed_inactive_node(i, j,   k, ibg, ℓx, f, ℓz) |
                                                                          immersed_inactive_node(i, j+1, k, ibg, ℓx, f, ℓz),
                                                                          zero(ibg),
                                                                          δy(i, j, k, ibg.underlying_grid, args...))

@inline conditional_δz_f(ℓx, ℓy, i, j, k, ibg::IBG, δz, args...) = ifelse(immersed_inactive_node(i, j, k,   ibg, ℓx, ℓy, c) |
                                                                          immersed_inactive_node(i, j, k-1, ibg, ℓx, ℓy, c),
                                                                          zero(ibg),
                                                                          δz(i, j, k, ibg.underlying_grid, args...))

@inline conditional_δz_c(ℓx, ℓy, i, j, k, ibg::IBG, δz, args...) = ifelse(immersed_inactive_node(i, j, k,   ibg, ℓx, ℓy, f) |
                                                                          immersed_inactive_node(i, j, k+1, ibg, ℓx, ℓy, f),
                                                                          zero(ibg),
                                                                          δz(i, j, k, ibg.underlying_grid, args...))

@inline translate_loc(a) = a == :ᶠ ? :f : :c

for (d, ξ) in enumerate((:x, :y, :z))
    for ℓx in (:ᶠ, :ᶜ), ℓy in (:ᶠ, :ᶜ), ℓz in (:ᶠ, :ᶜ)

        δξ             = Symbol(:δ, ξ, ℓx, ℓy, ℓz)
        loc            = translate_loc.((ℓx, ℓy, ℓz))
        conditional_δξ = Symbol(:conditional_δ, ξ, :_, loc[d])

        # `other_locs` contains locations in the two "other" directions not being differenced
        other_locs = []
        for l in 1:3
            if l != d
                push!(other_locs, loc[l])
            end
        end

        @eval begin
            @inline $δξ(i, j, k, ibg::IBG, args...)              = $conditional_δξ($(other_locs[1]), $(other_locs[2]), i, j, k, ibg, $δξ, args...)
            @inline $δξ(i, j, k, ibg::IBG, f::Function, args...) = $conditional_δξ($(other_locs[1]), $(other_locs[2]), i, j, k, ibg, $δξ, f::Function, args...)
       end
    end
end

# Topology-aware immersed boundary operators
#   * Velocities are `0` on `peripheral_node`s and
#   * Tracers should ensure no-flux on `inactive_node`s
@inline conditional_uᶠᶜᶜ(i, j, k, grid, f, args...) = f(i, j, k, grid, args...) * !peripheral_node(i, j, k, grid, f, c, c)
@inline conditional_vᶜᶠᶜ(i, j, k, grid, f, args...) = f(i, j, k, grid, args...) * !peripheral_node(i, j, k, grid, c, f, c)

@inline conditional_uᶠᶜᶜ(i, j, k, grid, u::AbstractArray) = @inbounds u[i, j, k] * !peripheral_node(i, j, k, grid, f, c, c)
@inline conditional_vᶜᶠᶜ(i, j, k, grid, v::AbstractArray) = @inbounds v[i, j, k] * !peripheral_node(i, j, k, grid, c, f, c)

@inline conditional_∂xTᶠᶜᶠ(i, j, k, ibg::IBG, args...) = ifelse(inactive_node(i, j, k, ibg, c, c, f) | inactive_node(i-1, j, k, ibg, c, c, f), zero(ibg), ∂xTᶠᶜᶠ(i, j, k, ibg.underlying_grid, args...))
@inline conditional_∂yTᶜᶠᶠ(i, j, k, ibg::IBG, args...) = ifelse(inactive_node(i, j, k, ibg, c, c, f) | inactive_node(i, j-1, k, ibg, c, c, f), zero(ibg), ∂yTᶜᶠᶠ(i, j, k, ibg.underlying_grid, args...))

@inline δxTᶜᵃᵃ(i, j, k, ibg::IBG, f, args...) = δxTᶜᵃᵃ(i, j, k, ibg.underlying_grid, conditional_uᶠᶜᶜ, f, args...)
@inline δyTᵃᶜᵃ(i, j, k, ibg::IBG, f, args...) = δyTᵃᶜᵃ(i, j, k, ibg.underlying_grid, conditional_vᶜᶠᶜ, f, args...)

@inline ∂xTᶠᶜᶠ(i, j, k, ibg::IBG, f, args...) = conditional_∂xTᶠᶜᶠ(i, j, k, ibg, f, args...)
@inline ∂yTᶜᶠᶠ(i, j, k, ibg::IBG, f, args...) = conditional_∂yTᶜᶠᶠ(i, j, k, ibg, f, args...)

@inline ∂xTᶠᶜᶠ(i, j, k, ibg::IBG, w::AbstractArray) = conditional_∂xTᶠᶜᶠ(i, j, k, ibg, f, args...)
@inline ∂yTᶜᶠᶠ(i, j, k, ibg::IBG, w::AbstractArray) = conditional_∂yTᶜᶠᶠ(i, j, k, ibg, f, args...)
