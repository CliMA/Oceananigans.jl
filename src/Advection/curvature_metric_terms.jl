using Oceananigans.Grids: AbstractGrid, RectilinearGrid
using Oceananigans.Operators

#####
##### Hydrostatic curvature metric terms
#####
##### These correct the flux-form momentum advection operator for the rotation
##### of basis vectors on curvilinear grids. They arise from the Christoffel
##### symbols and are NOT part of the flux divergence ∇·(v⊗v).
#####
##### The metric-ratio approach is used: tan(φ)/a ≈ -δy(Δx)/Az, which
##### generalises to any orthogonal curvilinear grid (not just LatitudeLongitudeGrid).
#####

# --- Hydrostatic u-metric at (f, c, c) ---

@inline function U_dot_∇u_hydrostatic_metric(i, j, k, grid, advection, U, V)
    V̂₂ = ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, V[2]) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    Û₂ = ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, U[2]) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    û₁ = @inbounds U[1][i, j, k]

    return + V̂₂ * û₁ * δyᵃᶜᵃ(i, j, k, grid, Δxᶠᶠᶜ) * Az⁻¹ᶠᶜᶜ(i, j, k, grid) -
             V̂₂ * Û₂ * δxᶠᵃᵃ(i, j, k, grid, Δyᶜᶜᶜ) * Az⁻¹ᶠᶜᶜ(i, j, k, grid)
end

# --- Hydrostatic v-metric at (c, f, c) ---

@inline function U_dot_∇v_hydrostatic_metric(i, j, k, grid, advection, U, V)
    V̂₁ = ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, V[1]) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    Û₁ = ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, U[1]) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    v̂₂ = @inbounds U[2][i, j, k]

    return + V̂₁ * v̂₂ * δxᶜᵃᵃ(i, j, k, grid, Δyᶠᶠᶜ) * Az⁻¹ᶜᶠᶜ(i, j, k, grid) -
             V̂₁ * Û₁ * δyᵃᶠᵃ(i, j, k, grid, Δxᶜᶜᶜ) * Az⁻¹ᶜᶠᶜ(i, j, k, grid)
end

#####
##### Non-hydrostatic curvature metric terms (w-coupling)
#####
##### These arise when the thin-atmosphere approximation is dropped.
##### MITgcm equations 2.105–2.107.
#####

# --- Non-hydrostatic u-metric at (f, c, c): -u w / R ---

@inline function _nonhydrostatic_metric_u(i, j, k, grid, U, V)
    ŵ = ℑxᶠᵃᵃ(i, j, k, grid, ℑzᵃᵃᶜ, V[3])
    û₁ = @inbounds U[1][i, j, k]
    return -û₁ * ŵ / grid.radius
end

# --- Non-hydrostatic v-metric at (c, f, c): -v w / R ---

@inline function _nonhydrostatic_metric_v(i, j, k, grid, U, V)
    ŵ = ℑyᵃᶠᵃ(i, j, k, grid, ℑzᵃᵃᶜ, V[3])
    v̂₂ = @inbounds U[2][i, j, k]
    return -v̂₂ * ŵ / grid.radius
end

# --- w-metric at (c, c, f): +(u² + v²) / R ---

@inline function U_dot_∇w_metric(i, j, k, grid, advection, U, V)
    û₁ = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶜᵃᵃ, U[1])
    v̂₂ = ℑzᵃᵃᶠ(i, j, k, grid, ℑyᵃᶜᵃ, U[2])
    Û₁ = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶜᵃᵃ, V[1])
    V̂₂ = ℑzᵃᵃᶠ(i, j, k, grid, ℑyᵃᶜᵃ, V[2])
    return (û₁ * Û₁ + v̂₂ * V̂₂) / grid.radius
end

#####
##### Full (non-hydrostatic) metric = hydrostatic + w-coupling
#####

@inline function U_dot_∇u_metric(i, j, k, grid, advection, U, V)
    return U_dot_∇u_hydrostatic_metric(i, j, k, grid, advection, U, V) +
           _nonhydrostatic_metric_u(i, j, k, grid, U, V)
end

@inline function U_dot_∇v_metric(i, j, k, grid, advection, U, V)
    return U_dot_∇v_hydrostatic_metric(i, j, k, grid, advection, U, V) +
           _nonhydrostatic_metric_v(i, j, k, grid, U, V)
end

#####
##### Zero dispatches
#####

# RectilinearGrid: no curvature
@inline U_dot_∇u_hydrostatic_metric(i, j, k, grid::RectilinearGrid, advection, U, V) = zero(grid)
@inline U_dot_∇v_hydrostatic_metric(i, j, k, grid::RectilinearGrid, advection, U, V) = zero(grid)
@inline U_dot_∇u_metric(i, j, k, grid::RectilinearGrid, advection, U, V) = zero(grid)
@inline U_dot_∇v_metric(i, j, k, grid::RectilinearGrid, advection, U, V) = zero(grid)
@inline U_dot_∇w_metric(i, j, k, grid::RectilinearGrid, advection, U, V) = zero(grid)

# No advection: no metric corrections
@inline U_dot_∇u_hydrostatic_metric(i, j, k, grid, ::Nothing, U, V) = zero(grid)
@inline U_dot_∇v_hydrostatic_metric(i, j, k, grid, ::Nothing, U, V) = zero(grid)
@inline U_dot_∇u_metric(i, j, k, grid, ::Nothing, U, V) = zero(grid)
@inline U_dot_∇v_metric(i, j, k, grid, ::Nothing, U, V) = zero(grid)
@inline U_dot_∇w_metric(i, j, k, grid, ::Nothing, U, V) = zero(grid)

# Ambiguity resolution: RectilinearGrid + Nothing
@inline U_dot_∇u_hydrostatic_metric(i, j, k, grid::RectilinearGrid, ::Nothing, U, V) = zero(grid)
@inline U_dot_∇v_hydrostatic_metric(i, j, k, grid::RectilinearGrid, ::Nothing, U, V) = zero(grid)
@inline U_dot_∇u_metric(i, j, k, grid::RectilinearGrid, ::Nothing, U, V) = zero(grid)
@inline U_dot_∇v_metric(i, j, k, grid::RectilinearGrid, ::Nothing, U, V) = zero(grid)
@inline U_dot_∇w_metric(i, j, k, grid::RectilinearGrid, ::Nothing, U, V) = zero(grid)

# VectorInvariant already accounts for horizontal curvature via vorticity/Bernoulli decomposition,
# so the hydrostatic metric functions return zero. The full (non-hydrostatic) functions still
# contribute the w-coupling terms.
@inline U_dot_∇u_hydrostatic_metric(i, j, k, grid, ::VectorInvariant, U, V) = zero(grid)
@inline U_dot_∇v_hydrostatic_metric(i, j, k, grid, ::VectorInvariant, U, V) = zero(grid)

@inline function U_dot_∇u_metric(i, j, k, grid, ::VectorInvariant, U, V)
    return _nonhydrostatic_metric_u(i, j, k, grid, U, V)
end

@inline function U_dot_∇v_metric(i, j, k, grid, ::VectorInvariant, U, V)
    return _nonhydrostatic_metric_v(i, j, k, grid, U, V)
end

# VectorInvariant + RectilinearGrid ambiguity
@inline U_dot_∇u_hydrostatic_metric(i, j, k, grid::RectilinearGrid, ::VectorInvariant, U, V) = zero(grid)
@inline U_dot_∇v_hydrostatic_metric(i, j, k, grid::RectilinearGrid, ::VectorInvariant, U, V) = zero(grid)
@inline U_dot_∇u_metric(i, j, k, grid::RectilinearGrid, ::VectorInvariant, U, V) = zero(grid)
@inline U_dot_∇v_metric(i, j, k, grid::RectilinearGrid, ::VectorInvariant, U, V) = zero(grid)
@inline U_dot_∇w_metric(i, j, k, grid::RectilinearGrid, ::VectorInvariant, U, V) = zero(grid)
