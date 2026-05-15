using Oceananigans.Operators: Δz, Az, volume, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using Oceananigans.Grids: peripheral_node, Center, Face

const AIVA = AdaptiveImplicitVerticalAdvection

#####
##### Implicit vertical velocity: wⁱ = w - wᵉ = w * (1 - 1/f(α, cfl))
#####
##### When α ≤ cfl: wⁱ = 0 (fully explicit)
##### When α > cfl: wⁱ = w * (1 - cfl/α)
#####
##### `w` is the W field at (Center, Center, Face). For tracers (CCC) the
##### value at (i, j, k) is correct; for u (FCC) and v (CFC) it must be
##### interpolated horizontally so the local CFL matches the actual
##### face velocity advecting the field.
#####

@inline w_at_face_loc(i, j, k, grid, w, ::Center, ::Center) = @inbounds w[i, j, k]
@inline w_at_face_loc(i, j, k, grid, w, ::Face,   ::Center) = ℑxᶠᵃᵃ(i, j, k, grid, w)
@inline w_at_face_loc(i, j, k, grid, w, ::Center, ::Face)   = ℑyᵃᶠᵃ(i, j, k, grid, w)

@inline function implicit_vertical_velocity(i, j, k, grid, w, Δt, cfl, ℓx, ℓy)
    wᵢ = w_at_face_loc(i, j, k, grid, w, ℓx, ℓy)
    Δzᵢ = Δz(i, j, k, grid, ℓx, ℓy, Face())
    α = abs(wᵢ) * Δt / Δzᵢ
    scale = ifelse(α > cfl, one(α) - cfl / α, zero(α))
    return wᵢ * scale
end

#####
##### Tridiagonal coefficients for implicit first-order upwind advection
##### (for fields at cell Centers in z: tracers and horizontal velocities).
#####
##### The upwind flux at face k+1 (top of cell k):
#####   F_{k+1} = Az_{k+1} * [max(wⁱ_{k+1}, 0) * c_k + min(wⁱ_{k+1}, 0) * c_{k+1}]
#####
##### The implicit system (I - Δt * L) cⁿ⁺¹ = c★ gives:
#####
##### Upper diagonal (coeff of c_{k+1}):  Δt / V_k * Az_{k+1} * min(wⁱ_{k+1}, 0)
##### Lower diagonal (coeff of c_{k-1}): -Δt / V_k * Az_k * max(wⁱ_k, 0)
##### Diagonal: -(upper at k) - (lower at k-1)
#####

# Upper diagonal: coefficient of c_{k+1} in the tridiagonal system
@inline function implicit_advection_upper_diagonal(i, j, k, grid, advection::AIVA, w, Δt, ℓx, ℓy)
    wⁱ = implicit_vertical_velocity(i, j, k+1, grid, w, Δt, advection.cfl, ℓx, ℓy)
    Azᵢ = Az(i, j, k+1, grid, ℓx, ℓy, Face())
    V⁻¹ = 1 / volume(i, j, k, grid, ℓx, ℓy, Center())
    return Δt * V⁻¹ * Azᵢ * min(wⁱ, zero(wⁱ)) * !peripheral_node(i, j, k+1, grid, ℓx, ℓy, Face())
end

# Lower diagonal: coefficient of c_{k-1} in the tridiagonal system
# Uses k′ = k-1 indexing convention (LinearAlgebra.Tridiagonal convention, matching ivd_lower_diagonal)
@inline function implicit_advection_lower_diagonal(i, j, k′, grid, advection::AIVA, w, Δt, ℓx, ℓy)
    k = k′ + 1
    wⁱ = implicit_vertical_velocity(i, j, k, grid, w, Δt, advection.cfl, ℓx, ℓy)
    Azᵢ = Az(i, j, k, grid, ℓx, ℓy, Face())
    V⁻¹ = 1 / volume(i, j, k, grid, ℓx, ℓy, Center())
    return -Δt * V⁻¹ * Azᵢ * max(wⁱ, zero(wⁱ)) * !peripheral_node(i, j, k′, grid, ℓx, ℓy, Center())
end

# Diagonal: ensures the row sums to the correct value
@inline function implicit_advection_diagonal(i, j, k, grid, advection::AIVA, w, Δt, ℓx, ℓy)
    return - implicit_advection_upper_diagonal(i, j, k, grid, advection, w, Δt, ℓx, ℓy) -
             implicit_advection_lower_diagonal(i, j, k-1, grid, advection, w, Δt, ℓx, ℓy)
end
