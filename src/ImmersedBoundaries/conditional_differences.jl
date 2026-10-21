using Oceananigans.Operators: Operators, δxTᶜᵃᵃ, δyTᵃᶜᵃ, ∂xᵣTᶠᶜᶠ, ∂yᵣTᶜᶠᶠ

# Topology-aware immersed boundary operators: velocities are `0` on `peripheral_node`s
@inline conditional_uᶠᶜᶜ(i, j, k, grid, ϕ, args...) = ϕ(i, j, k, grid, args...) * !peripheral_node(i, j, k, grid, f, c, c)
@inline conditional_vᶜᶠᶜ(i, j, k, grid, ϕ, args...) = ϕ(i, j, k, grid, args...) * !peripheral_node(i, j, k, grid, c, f, c)

@inline conditional_uᶠᶜᶜ(i, j, k, grid, u::AbstractArray) = @inbounds u[i, j, k] * !peripheral_node(i, j, k, grid, f, c, c)
@inline conditional_vᶜᶠᶜ(i, j, k, grid, v::AbstractArray) = @inbounds v[i, j, k] * !peripheral_node(i, j, k, grid, c, f, c)

@inline Operators.δxTᶜᵃᵃ(i, j, k, ibg::IBG, f, args...) = δxTᶜᵃᵃ(i, j, k, ibg.underlying_grid, conditional_uᶠᶜᶜ, f, args...)
@inline Operators.δyTᵃᶜᵃ(i, j, k, ibg::IBG, f, args...) = δyTᵃᶜᵃ(i, j, k, ibg.underlying_grid, conditional_vᶜᶠᶜ, f, args...)

# The immersed test selects the whole derivative: Δ⁻¹ is infinite at the singular points of a grid (the tripolar fold poles)
@inline Operators.∂xᵣTᶠᶜᶠ(i, j, k, ibg::IBG, ϕ, args...) = ifelse(immersed_inactive_node(i, j, k, ibg, c, c, f) | immersed_inactive_node(i-1, j, k, ibg, c, c, f),
                                                                  zero(ibg), ∂xᵣTᶠᶜᶠ(i, j, k, ibg.underlying_grid, ϕ, args...))

@inline Operators.∂yᵣTᶜᶠᶠ(i, j, k, ibg::IBG, ϕ, args...) = ifelse(immersed_inactive_node(i, j, k, ibg, c, c, f) | immersed_inactive_node(i, j-1, k, ibg, c, c, f),
                                                                  zero(ibg), ∂yᵣTᶜᶠᶠ(i, j, k, ibg.underlying_grid, ϕ, args...))
