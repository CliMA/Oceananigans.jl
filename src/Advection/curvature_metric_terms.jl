using Oceananigans.Grids: RectilinearGrid
using Oceananigans.Operators

#####
##### Curvature metric terms for flux-form momentum advection
#####
##### These correct for the rotation of basis vectors on curvilinear grids.
##### They arise from the Christoffel symbols and are NOT part of the flux
##### divergence ∇·(v⊗v).
#####
##### Argument convention matches div_𝐯u(i, j, k, grid, advection, U, V):
#####   U = advector (transport / mass-flux)
#####   V = advectee (velocity)
#####
##### The metric-ratio approach is used for the hydrostatic terms:
#####   tan(φ)/a ≈ −δy(Δx)/Az
##### which generalises to any orthogonal curvilinear grid.
#####

# --- Hydrostatic u-metric at (f, c, c) ---

@inline function U_dot_∇u_hydrostatic_metric(i, j, k, grid, advection, U, V)
    Û₂ = ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, U[2]) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    V̂₂ = ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, V[2]) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    v̂₁ = @inbounds V[1][i, j, k]

    return + Û₂ * v̂₁ * δyᵃᶜᵃ(i, j, k, grid, Δxᶠᶠᶜ) * Az⁻¹ᶠᶜᶜ(i, j, k, grid) -
             Û₂ * V̂₂ * δxᶠᵃᵃ(i, j, k, grid, Δyᶜᶜᶜ) * Az⁻¹ᶠᶜᶜ(i, j, k, grid)
end

# --- Hydrostatic v-metric at (c, f, c) ---

@inline function U_dot_∇v_hydrostatic_metric(i, j, k, grid, advection, U, V)
    Û₁ = ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, U[1]) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    V̂₁ = ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, V[1]) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    v̂₂ = @inbounds V[2][i, j, k]

    return + Û₁ * v̂₂ * δxᶜᵃᵃ(i, j, k, grid, Δyᶠᶠᶜ) * Az⁻¹ᶜᶠᶜ(i, j, k, grid) -
             Û₁ * V̂₁ * δyᵃᶠᵃ(i, j, k, grid, Δxᶜᶜᶜ) * Az⁻¹ᶜᶠᶜ(i, j, k, grid)
end

#####
##### Non-hydrostatic curvature metric terms (w-coupling)
#####
##### These arise when the thin-atmosphere approximation is dropped.
##### Energy-conserving volume-weighted discretization (MITgcm eqs 2.105–2.107):
#####   V_u G_u = − ī[ ū^i w̄^k V_c / a ]       (2.105)
#####   V_v G_v = − j̄[ v̄^j w̄^k V_c / a ]       (2.106)
#####   V_w G_w = + k̄[ (ū^i² + v̄^j²) V_c / a ]  (2.107)
#####

# --- Volume-weighted products at (c, c, c) for interpolation back to velocity points ---

@inline function _uw_Vᶜᶜᶜ(i, j, k, grid, U, V)
    ū = ℑxᶜᵃᵃ(i, j, k, grid, V[1])
    w̄ = ℑzᵃᵃᶜ(i, j, k, grid, U[3])
    return ū * w̄ * Vᶜᶜᶜ(i, j, k, grid)
end

@inline function _vw_Vᶜᶜᶜ(i, j, k, grid, U, V)
    v̄ = ℑyᵃᶜᵃ(i, j, k, grid, V[2])
    w̄ = ℑzᵃᵃᶜ(i, j, k, grid, U[3])
    return v̄ * w̄ * Vᶜᶜᶜ(i, j, k, grid)
end

@inline function _u²v²_Vᶜᶜᶜ(i, j, k, grid, U, V)
    ū = ℑxᶜᵃᵃ(i, j, k, grid, V[1])
    v̄ = ℑyᵃᶜᵃ(i, j, k, grid, V[2])
    Ū = ℑxᶜᵃᵃ(i, j, k, grid, U[1])
    V̄ = ℑyᵃᶜᵃ(i, j, k, grid, U[2])
    return (ū * Ū + v̄ * V̄) * Vᶜᶜᶜ(i, j, k, grid)
end

# --- Non-hydrostatic u-metric at (f, c, c): eq 2.105 ---
# G_u = −(1/a V_u) ī[ ū w̄ V_c ]
# Returns −G_u (positive) since the tendency subtracts U_dot_∇u_metric.

@inline function _nonhydrostatic_metric_u(i, j, k, grid, U, V)
    return V⁻¹ᶠᶜᶜ(i, j, k, grid) / grid.radius * ℑxᶠᵃᵃ(i, j, k, grid, _uw_Vᶜᶜᶜ, U, V)
end

# --- Non-hydrostatic v-metric at (c, f, c): eq 2.106 ---

@inline function _nonhydrostatic_metric_v(i, j, k, grid, U, V)
    return V⁻¹ᶜᶠᶜ(i, j, k, grid) / grid.radius * ℑyᵃᶠᵃ(i, j, k, grid, _vw_Vᶜᶜᶜ, U, V)
end

# --- w-metric at (c, c, f): eq 2.107 ---
# G_w = +(1/a V_w) k̄[ (ū² + v̄²) V_c ]
# Returns −G_w (negative) since the tendency subtracts U_dot_∇w_metric.

@inline function U_dot_∇w_metric(i, j, k, grid, advection, U, V)
    return -V⁻¹ᶜᶜᶠ(i, j, k, grid) / grid.radius * ℑzᵃᵃᶠ(i, j, k, grid, _u²v²_Vᶜᶜᶜ, U, V)
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
