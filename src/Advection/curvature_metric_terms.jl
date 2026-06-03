using Oceananigans.Grids: AbstractHorizontallyCurvilinearGrid, SphericalShellGrid
using Oceananigans.Operators: Δxᶜᶜᶜ, Δxᶠᶠᶜ, Δyᶜᶜᶜ, Δyᶠᶠᶜ,
                              V⁻¹ᶠᶜᶜ, V⁻¹ᶜᶠᶜ,
                              δxᶠᵃᵃ, δxᶜᵃᵃ, δyᵃᶜᵃ, δyᵃᶠᵃ,
                              covariant_rotational_advection_uᶠᶜᶜ,
                              covariant_rotational_advection_vᶜᶠᶜ,
                              covariant_bernoulli_head_uᶠᶜᶜ,
                              covariant_bernoulli_head_vᶜᶠᶜ

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

# Grids with horizontal curvature: LatitudeLongitudeGrid, OrthogonalSphericalShellGrid,
# and ImmersedBoundaryGrid wrapping either of those.
const HCG = AbstractHorizontallyCurvilinearGrid
const HCGOrIBG = Union{HCG, ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:HCG}}

#####
##### Default fallbacks: no curvature → zero metric.
##### These cover RectilinearGrid, ImmersedBoundaryGrid wrapping RectilinearGrid,
##### Nothing advection, VectorInvariant advection (which already includes the
##### horizontal metric in its vorticity / Bernoulli decomposition), and any
##### combination thereof.
#####

@inline U_dot_∇u_hydrostatic_metric(i, j, k, grid, advection, U, V) = zero(grid)
@inline U_dot_∇v_hydrostatic_metric(i, j, k, grid, advection, U, V) = zero(grid)

@inline U_dot_∇u_metric(i, j, k, grid, advection, U, V) = zero(grid)
@inline U_dot_∇v_metric(i, j, k, grid, advection, U, V) = zero(grid)
@inline U_dot_∇w_metric(i, j, k, grid, advection, U, V) = zero(grid)

@inline function horizontal_div_𝐯u(i, j, k, grid, advection, U, u)
    u_component = u_velocity(U)
    v_component = v_velocity(U)
    return V⁻¹ᶠᶜᶜ(i, j, k, grid) *
           (δxᶠᵃᵃ(i, j, k, grid, _advective_momentum_flux_Uu, advection, u_component, u) +
            δyᵃᶜᵃ(i, j, k, grid, _advective_momentum_flux_Vu, advection, v_component, u))
end

@inline function horizontal_div_𝐯u(i, j, k, grid::SphericalShellGrid, advection, U, u)
    return V⁻¹ᶠᶜᶜ(i, j, k, grid) *
           (δxᶠᵃᵃ(i, j, k, grid, _advective_momentum_flux_Uu, advection, U, u) +
            δyᵃᶜᵃ(i, j, k, grid, _advective_momentum_flux_Vu, advection, U, u))
end

@inline function horizontal_div_𝐯v(i, j, k, grid, advection, U, v)
    u_component = u_velocity(U)
    v_component = v_velocity(U)
    return V⁻¹ᶜᶠᶜ(i, j, k, grid) *
           (δxᶜᵃᵃ(i, j, k, grid, _advective_momentum_flux_Uv, advection, u_component, v) +
            δyᵃᶠᵃ(i, j, k, grid, _advective_momentum_flux_Vv, advection, v_component, v))
end

@inline function horizontal_div_𝐯v(i, j, k, grid::SphericalShellGrid, advection, U, v)
    return V⁻¹ᶜᶠᶜ(i, j, k, grid) *
           (δxᶜᵃᵃ(i, j, k, grid, _advective_momentum_flux_Uv, advection, U, v) +
            δyᵃᶠᵃ(i, j, k, grid, _advective_momentum_flux_Vv, advection, U, v))
end

#####
##### Hydrostatic curvature metric terms — active on horizontally-curvilinear grids.
#####

# u-metric at (f, c, c)
@inline function U_dot_∇u_hydrostatic_metric(i, j, k, grid::HCGOrIBG, advection, U, V)
    u_component = u_velocity(V)
    Uv = v_velocity(U)
    Vv = v_velocity(V)
    Û₂ = ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, Uv) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    V̂₂ = ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, Vv) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    v̂₁ = @inbounds u_component[i, j, k]

    return + Û₂ * v̂₁ * δyᵃᶜᵃ(i, j, k, grid, Δxᶠᶠᶜ) * Az⁻¹ᶠᶜᶜ(i, j, k, grid) -
             Û₂ * V̂₂ * δxᶠᵃᵃ(i, j, k, grid, Δyᶜᶜᶜ) * Az⁻¹ᶠᶜᶜ(i, j, k, grid)
end

# v-metric at (c, f, c)
@inline function U_dot_∇v_hydrostatic_metric(i, j, k, grid::HCGOrIBG, advection, U, V)
    v_component = v_velocity(V)
    Uu = u_velocity(U)
    Vu = u_velocity(V)
    Û₁ = ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, Uu) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    V̂₁ = ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, Vu) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    v̂₂ = @inbounds v_component[i, j, k]

    return + Û₁ * v̂₂ * δxᶜᵃᵃ(i, j, k, grid, Δyᶠᶠᶜ) * Az⁻¹ᶜᶠᶜ(i, j, k, grid) -
             Û₁ * V̂₁ * δyᵃᶠᵃ(i, j, k, grid, Δxᶜᶜᶜ) * Az⁻¹ᶜᶠᶜ(i, j, k, grid)
end

@inline function U_dot_∇u_hydrostatic_metric(i, j, k, grid::SphericalShellGrid, advection::Centered, U, V)
    u_component = u_velocity(V)
    v_component = v_velocity(V)
    return covariant_rotational_advection_uᶠᶜᶜ(i, j, k, grid, u_component, v_component) +
           covariant_bernoulli_head_uᶠᶜᶜ(i, j, k, grid, u_component, v_component) -
           horizontal_div_𝐯u(i, j, k, grid, advection, U, u_component)
end

@inline function U_dot_∇v_hydrostatic_metric(i, j, k, grid::SphericalShellGrid, advection::Centered, U, V)
    u_component = u_velocity(V)
    v_component = v_velocity(V)
    return covariant_rotational_advection_vᶜᶠᶜ(i, j, k, grid, u_component, v_component) +
           covariant_bernoulli_head_vᶜᶠᶜ(i, j, k, grid, u_component, v_component) -
           horizontal_div_𝐯v(i, j, k, grid, advection, U, v_component)
end

@inline function U_dot_∇u_hydrostatic_metric(i, j, k,
                                             grid::SphericalShellGrid,
                                             advection::FluxFormAdvection,
                                             U, V)
    u_component = u_velocity(V)
    v_component = v_velocity(V)
    return covariant_rotational_advection_uᶠᶜᶜ(i, j, k, grid, u_component, v_component) +
           covariant_bernoulli_head_uᶠᶜᶜ(i, j, k, grid, u_component, v_component) -
           horizontal_div_𝐯u(i, j, k, grid, advection, U, u_component)
end

@inline function U_dot_∇v_hydrostatic_metric(i, j, k,
                                             grid::SphericalShellGrid,
                                             advection::FluxFormAdvection,
                                             U, V)
    u_component = u_velocity(V)
    v_component = v_velocity(V)
    return covariant_rotational_advection_vᶜᶠᶜ(i, j, k, grid, u_component, v_component) +
           covariant_bernoulli_head_vᶜᶠᶜ(i, j, k, grid, u_component, v_component) -
           horizontal_div_𝐯v(i, j, k, grid, advection, U, v_component)
end

#####
##### Non-hydrostatic curvature metric terms (w-coupling) — active on horizontally-curvilinear grids.
#####
##### These arise when the thin-atmosphere approximation is dropped.
##### Energy-conserving volume-weighted discretization (MITgcm eqs 2.105–2.107):
#####   V_u G_u = − ī[ ū^i w̄^k V_c / a ]       (2.105)
#####   V_v G_v = − j̄[ v̄^j w̄^k V_c / a ]       (2.106)
#####   V_w G_w = + k̄[ (ū^i² + v̄^j²) V_c / a ]  (2.107)
#####

# Volume-weighted products at (c, c, c) for interpolation back to velocity points

@inline function _uw_Vᶜᶜᶜ(i, j, k, grid, U, V)
    u_component = u_velocity(V)
    w_component = w_velocity(U)
    ū = ℑxᶜᵃᵃ(i, j, k, grid, u_component)
    w̄ = ℑzᵃᵃᶜ(i, j, k, grid, w_component)
    return ū * w̄ * Vᶜᶜᶜ(i, j, k, grid)
end

@inline function _vw_Vᶜᶜᶜ(i, j, k, grid, U, V)
    v_component = v_velocity(V)
    w_component = w_velocity(U)
    v̄ = ℑyᵃᶜᵃ(i, j, k, grid, v_component)
    w̄ = ℑzᵃᵃᶜ(i, j, k, grid, w_component)
    return v̄ * w̄ * Vᶜᶜᶜ(i, j, k, grid)
end

@inline function _u²v²_Vᶜᶜᶜ(i, j, k, grid, U, V)
    u_component = u_velocity(V)
    v_component = v_velocity(V)
    Uu = u_velocity(U)
    Uv = v_velocity(U)
    ū = ℑxᶜᵃᵃ(i, j, k, grid, u_component)
    v̄ = ℑyᵃᶜᵃ(i, j, k, grid, v_component)
    Ū = ℑxᶜᵃᵃ(i, j, k, grid, Uu)
    V̄ = ℑyᵃᶜᵃ(i, j, k, grid, Uv)
    return (ū * Ū + v̄ * V̄) * Vᶜᶜᶜ(i, j, k, grid)
end

# u-metric (nonhydrostatic w-coupling part) at (f, c, c): eq 2.105
# G_u = −(1/a V_u) ī[ ū w̄ V_c ]
# Returns −G_u (positive) since the tendency subtracts U_dot_∇u_nonhydrostatic_metric.

@inline function U_dot_∇u_nonhydrostatic_metric(i, j, k, grid::HCGOrIBG, U, V)
    return V⁻¹ᶠᶜᶜ(i, j, k, grid) / grid.radius * ℑxᶠᵃᵃ(i, j, k, grid, _uw_Vᶜᶜᶜ, U, V)
end

# v-metric (nonhydrostatic w-coupling part) at (c, f, c): eq 2.106

@inline function U_dot_∇v_nonhydrostatic_metric(i, j, k, grid::HCGOrIBG, U, V)
    return V⁻¹ᶜᶠᶜ(i, j, k, grid) / grid.radius * ℑyᵃᶠᵃ(i, j, k, grid, _vw_Vᶜᶜᶜ, U, V)
end

# w-metric at (c, c, f): eq 2.107
# G_w = +(1/a V_w) k̄[ (ū² + v̄²) V_c ]
# Returns −G_w (negative) since the tendency subtracts U_dot_∇w_metric.

@inline function U_dot_∇w_metric(i, j, k, grid::HCGOrIBG, advection, U, V)
    return -V⁻¹ᶜᶜᶠ(i, j, k, grid) / grid.radius * ℑzᵃᵃᶠ(i, j, k, grid, _u²v²_Vᶜᶜᶜ, U, V)
end

#####
##### Full (non-hydrostatic) metric on horizontally-curvilinear grids = hydrostatic + w-coupling
#####

@inline function U_dot_∇u_metric(i, j, k, grid::HCGOrIBG, advection, U, V)
    return U_dot_∇u_hydrostatic_metric(i, j, k, grid, advection, U, V) +
           U_dot_∇u_nonhydrostatic_metric(i, j, k, grid, U, V)
end

@inline function U_dot_∇v_metric(i, j, k, grid::HCGOrIBG, advection, U, V)
    return U_dot_∇v_hydrostatic_metric(i, j, k, grid, advection, U, V) +
           U_dot_∇v_nonhydrostatic_metric(i, j, k, grid, U, V)
end

#####
##### VectorInvariant on curvilinear grids: vorticity / Bernoulli decomposition already
##### accounts for horizontal curvature, so the hydrostatic metric is zero. The
##### nonhydrostatic w-coupling terms still apply.
#####

@inline U_dot_∇u_hydrostatic_metric(i, j, k, grid::HCGOrIBG, ::VectorInvariant, U, V) = zero(grid)
@inline U_dot_∇v_hydrostatic_metric(i, j, k, grid::HCGOrIBG, ::VectorInvariant, U, V) = zero(grid)

@inline function U_dot_∇u_metric(i, j, k, grid::HCGOrIBG, ::VectorInvariant, U, V)
    return U_dot_∇u_nonhydrostatic_metric(i, j, k, grid, U, V)
end

@inline function U_dot_∇v_metric(i, j, k, grid::HCGOrIBG, ::VectorInvariant, U, V)
    return U_dot_∇v_nonhydrostatic_metric(i, j, k, grid, U, V)
end

#####
##### Nothing advection on curvilinear grids: no advection ⇒ no metric correction.
##### These exist purely for ambiguity resolution against the generic Nothing fallback above.
#####

@inline U_dot_∇u_hydrostatic_metric(i, j, k, grid::HCGOrIBG, ::Nothing, U, V) = zero(grid)
@inline U_dot_∇v_hydrostatic_metric(i, j, k, grid::HCGOrIBG, ::Nothing, U, V) = zero(grid)
@inline U_dot_∇u_metric(i, j, k, grid::HCGOrIBG, ::Nothing, U, V) = zero(grid)
@inline U_dot_∇v_metric(i, j, k, grid::HCGOrIBG, ::Nothing, U, V) = zero(grid)
@inline U_dot_∇w_metric(i, j, k, grid::HCGOrIBG, ::Nothing, U, V) = zero(grid)
