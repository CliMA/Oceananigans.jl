using Oceananigans.ImmersedBoundaries: BottomAndTopIBG
using Oceananigans.Operators: ∂xᶜᶜᶜ, ∂yᶜᶜᶜ, ∂xᶠᶠᶜ, ∂yᶠᶠᶜ
using Oceananigans.Grids: peripheral_node

# Difference before weighting, so uniform flow is stress-free across partial-cell steps

# Free slip: no stress into a peripheral neighbor
@inline function viscous_flux_ux(i, j, k, grid::BottomAndTopIBG, clo::AHD, K, clk, fields, b)
    τ = - νhᶜᶜᶜ(i, j, k, grid, clo, K, clk, fields) * ∂xᶜᶜᶜ(i, j, k, grid, fields.u)
    peripheral = peripheral_node(i, j, k, grid, f, c, c) | peripheral_node(i + 1, j, k, grid, f, c, c)
    return ifelse(peripheral, zero(grid), τ)
end

@inline function viscous_flux_vy(i, j, k, grid::BottomAndTopIBG, clo::AHD, K, clk, fields, b)
    τ = - νhᶜᶜᶜ(i, j, k, grid, clo, K, clk, fields) * ∂yᶜᶜᶜ(i, j, k, grid, fields.v)
    peripheral = peripheral_node(i, j, k, grid, c, f, c) | peripheral_node(i, j + 1, k, grid, c, f, c)
    return ifelse(peripheral, zero(grid), τ)
end

@inline viscous_flux_uy(i, j, k, grid::BottomAndTopIBG, clo::AHD, K, clk, fields, b) =
    - νhᶠᶠᶜ(i, j, k, grid, clo, K, clk, fields) * ∂yᶠᶠᶜ(i, j, k, grid, fields.u)

@inline viscous_flux_vx(i, j, k, grid::BottomAndTopIBG, clo::AHD, K, clk, fields, b) =
    - νhᶠᶠᶜ(i, j, k, grid, clo, K, clk, fields) * ∂xᶠᶠᶜ(i, j, k, grid, fields.v)
