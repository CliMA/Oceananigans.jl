using Oceananigans.Operators: Δz, Az, volume, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using Oceananigans.Grids: peripheral_node, Center, Face

const AIVA = AdaptiveImplicitVerticalAdvection

#####
##### Implicit vertical velocity: wⁱ = w - wᵉ = w * (1 - 1/f(α, cfl))
#####
##### When α ≤ cfl: wⁱ = 0 (fully explicit)
##### When α > cfl: wⁱ = w * (1 - cfl/α)
#####
##### `w` is the W field at (Center, Center, Face). For tracers (CCC) the value at (i, j, k) is correct;
##### for u (FCC) and v (CFC) it must be reconstructed horizontally so the local CFL matches the actual
##### face velocity advecting the field.
#####

@inline function implicit_vertical_velocityᶜᶜᶠ(i, j, k, grid, scheme, td, W)
    Δt = td.Δt[]
    Δz = Δzᶜᶜᶠ(i, j, k, grid)
    w  = @inbounds W[i, j, k]
    α  = abs(w) * Δt / Δz
    return w * (1 - ifelse(α > td.cfl, td.cfl / α, one(α)))
end

@inline function implicit_velocity_scaleᶠᶜᶠ(i, j, k, grid, scheme, td, W)
    Δt = td.Δt[]
    Δz = Δzᶠᶜᶠ(i, j, k, grid)
    w  = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, W)
    α  = abs(w) * Δt / Δz
    return w * (1 - ifelse(α > td.cfl, td.cfl / α, one(α)))
end

@inline function implicit_velocity_scaleᶜᶠᶠ(i, j, k, grid, scheme, W)
    Δt = td.Δt[]
    Δz = Δzᶜᶠᶠ(i, j, k, grid)
    w  = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, W)
    α  = abs(w) * Δt / Δz
    return w * (1 - ifelse(α > td.cfl, td.cfl / α, one(α)))
end

#####
##### Tridiagonal coefficients for implicit first-order upwind advection (for fields at cell Centers in z).
#####
##### The upwind flux at face k+1 (top of cell k):
#####   F_{k+1} = Az_{k+1} * [max(wⁱ_{k+1}, 0) * c_k + min(wⁱ_{k+1}, 0) * c_{k+1}]
#####
##### The implicit system (I - Δt * L) cⁿ⁺¹ = c★ gives:
#####
##### Upper diagonal (coeff of c_{k+1}):   Δt / V_k * Az_{k+1} * min(wⁱ_{k+1}, 0)
##### Lower diagonal (coeff of c_{k-1}): - Δt / V_k * Az_k * max(wⁱ_k, 0)
##### Diagonal: - (upper at k) - (lower at k-1)
#####

@inline implicit_vertical_velocity(::Center, ::Center, args...) = implicit_vertical_velocityᶜᶜᶠ(args...)
@inline implicit_vertical_velocity(::Face,   ::Center, args...) = implicit_vertical_velocityᶠᶜᶠ(args...)
@inline implicit_vertical_velocity(::Center, ::Face,   args...) = implicit_vertical_velocityᶜᶠᶠ(args...)

# Upper diagonal: coefficient of c_{k+1} in the tridiagonal system
@inline function implicit_advection_upper_diagonal(i, j, k, grid, advection::AIVA, w, Δt, ℓx, ℓy)
    wⁱ  = implicit_vertical_velocity(ℓx, ℓy, i, j, k+1, grid, advection, advection.time_discretization, w)
    Azᵢ = Az(i, j, k+1, grid, ℓx, ℓy, Face())
    V⁻¹ = 1 / volume(i, j, k, grid, ℓx, ℓy, Center())
    return Δt * V⁻¹ * Azᵢ * min(wⁱ, zero(wⁱ)) * !peripheral_node(i, j, k+1, grid, ℓx, ℓy, Face())
end

# Lower diagonal: coefficient of c_{k-1} in the tridiagonal system
# Uses k′ = k-1 indexing convention (LinearAlgebra.Tridiagonal convention, matching ivd_lower_diagonal)
@inline function implicit_advection_lower_diagonal(i, j, k′, grid, advection::AIVA, w, Δt, ℓx, ℓy)
    k   = k′ + 1
    wⁱ  = implicit_vertical_velocity(ℓx, ℓy, i, j, k, grid, advection, advection.time_discretization, w)
    Azᵢ = Az(i, j, k, grid, ℓx, ℓy, Face())
    V⁻¹ = 1 / volume(i, j, k, grid, ℓx, ℓy, Center())
    return - Δt * V⁻¹ * Azᵢ * max(wⁱ, zero(wⁱ)) * !peripheral_node(i, j, k′, grid, ℓx, ℓy, Center())
end

@inline function implicit_advection_diagonal(i, j, k, grid, advection::AIVA, w, Δt, ℓx, ℓy)
    wⁱ⁺ = implicit_vertical_velocity(ℓx, ℓy, i, j, k+1, grid, advection, advection.time_discretization, w)
    wⁱ⁻ = implicit_vertical_velocity(ℓx, ℓy, i, j, k,   grid, advection, advection.time_discretization, w)

    Az⁺ = Az(i, j, k+1, grid, ℓx, ℓy, Face())
    Az⁻ = Az(i, j, k,   grid, ℓx, ℓy, Face())

    active⁺ = !peripheral_node(i, j, k+1, grid, ℓx, ℓy, Face())
    active⁻ = !peripheral_node(i, j, k,   grid, ℓx, ℓy, Face())

    V⁻¹ = 1 / volume(i, j, k, grid, ℓx, ℓy, Center())

    return Δt * V⁻¹ * (Az⁺ * max(wⁱ⁺, zero(wⁱ⁺)) * active⁺ -
                       Az⁻ * min(wⁱ⁻, zero(wⁱ⁻)) * active⁻)
end
