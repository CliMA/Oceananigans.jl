using Oceananigans.Grids: peripheral_node, Center, Face
using Oceananigans.Operators: Az, volume, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ

@inline vertical_scheme(advection) = advection
@inline vertical_scheme(advection::VectorInvariant) = advection.vertical_advection_scheme

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
    Δt = _unwrap_for_gpu(td.Δt)
    Δz = Δzᶜᶜᶠ(i, j, k, grid)
    w  = @inbounds W[i, j, k]
    α  = abs(w) * Δt / Δz
    return w * (1 - ifelse(α > td.cfl, td.cfl / α, one(α)))
end

@inline function implicit_vertical_velocityᶠᶜᶠ(i, j, k, grid, scheme, td, W)
    Δt = _unwrap_for_gpu(td.Δt)
    Δz = Δzᶠᶜᶠ(i, j, k, grid)
    w  = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, W)
    α  = abs(w) * Δt / Δz
    return w * (1 - ifelse(α > td.cfl, td.cfl / α, one(α)))
end

@inline function implicit_vertical_velocityᶜᶠᶠ(i, j, k, grid, scheme, td, W)
    Δt = _unwrap_for_gpu(td.Δt)
    Δz = Δzᶜᶠᶠ(i, j, k, grid)
    w  = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, W)
    α  = abs(w) * Δt / Δz
    return w * (1 - ifelse(α > td.cfl, td.cfl / α, one(α)))
end

#####
##### Optional density weighting for mass-flux (anelastic / compressible) models.
#####
##### Boussinesq models advect the tracer `c` with a volume-conserving flux, so the default
##### `density === nothing` reproduces the volume-conserving coefficients exactly. Mass-flux models
##### evolve `q = ρ c` with the flux `Az ρ w · upwind(c)`, `c = q / ρ`: pass the (reference or
##### prognostic) density `ρ` and the coefficients are weighted by the density interpolated to the
##### advecting face and divided by the density at the reconstructed cell centre. `ρ` is evaluated at
##### the advected field's location, so this is intended for tracers (Center, Center, Center).
#####

# Density at the tracer cell centre (ᶜᶜᶜ) and the vertical interface (ᶜᶜᶠ). `nothing` ⇒ unit weight,
# which recovers the volume-conserving (Boussinesq) coefficients.
@inline densityᶜᶜᶜ(i, j, k, grid, ρ) = @inbounds ρ[i, j, k]
@inline densityᶜᶜᶠ(i, j, k, grid, ρ) = ℑzᵃᵃᶠ(i, j, k, grid, ρ)
@inline densityᶜᶜᶜ(i, j, k, grid, ::Nothing) = one(grid)
@inline densityᶜᶜᶠ(i, j, k, grid, ::Nothing) = one(grid)

#####
##### Tridiagonal coefficients for implicit first-order upwind advection (for fields at cell Centers in z).
#####
##### The upwind flux at face k+1 (top of cell k), weighted by the face density ϖᶠ:
#####   F_{k+1} = Az_{k+1} ϖᶠ_{k+1} * [max(wⁱ_{k+1}, 0) * c_k + min(wⁱ_{k+1}, 0) * c_{k+1}],   c = q / ρ
#####
##### The implicit system (I - Δt * L) qⁿ⁺¹ = q★ gives (with ϖᶜ the cell density of the reconstructed value):
#####
##### Upper diagonal (coeff of q_{k+1}):   Δt / V_k * Az_{k+1} ϖᶠ_{k+1} / ϖᶜ_{k+1} * min(wⁱ_{k+1}, 0)
##### Lower diagonal (coeff of q_{k-1}): - Δt / V_k * Az_k     ϖᶠ_k     / ϖᶜ_{k-1} * max(wⁱ_k, 0)
#####
##### With `density === nothing`, ϖᶠ = ϖᶜ = 1 and these reduce to the volume-conserving coefficients.
#####

@inline implicit_vertical_velocity(::Center, ::Center, args...) = implicit_vertical_velocityᶜᶜᶠ(args...)
@inline implicit_vertical_velocity(::Face,   ::Center, args...) = implicit_vertical_velocityᶠᶜᶠ(args...)
@inline implicit_vertical_velocity(::Center, ::Face,   args...) = implicit_vertical_velocityᶜᶠᶠ(args...)

# Upper diagonal: coefficient of q_{k+1} in the tridiagonal system
@inline function implicit_advection_upper_diagonal(i, j, k, grid, advection::AIVA, w, Δt, ℓx, ℓy, density=nothing)
    scheme = vertical_scheme(advection)
    td     = TimeSteppers.time_discretization(scheme)
    wⁱ  = implicit_vertical_velocity(ℓx, ℓy, i, j, k+1, grid, scheme, td, w)
    Azᵢ = Az(i, j, k+1, grid, ℓx, ℓy, Face())
    ϖᶠ  = densityᶜᶜᶠ(i, j, k+1, grid, density)
    ϖᶜ  = densityᶜᶜᶜ(i, j, k+1, grid, density)
    V⁻¹ = 1 / volume(i, j, k, grid, ℓx, ℓy, Center())
    return Δt * V⁻¹ * Azᵢ * ϖᶠ / ϖᶜ * min(wⁱ, zero(wⁱ)) * !peripheral_node(i, j, k+1, grid, ℓx, ℓy, Face())
end

# Lower diagonal: coefficient of q_{k-1} in the tridiagonal system
# Uses k′ = k-1 indexing convention (LinearAlgebra.Tridiagonal convention, matching ivd_lower_diagonal)
@inline function implicit_advection_lower_diagonal(i, j, k′, grid, advection::AIVA, w, Δt, ℓx, ℓy, density=nothing)
    scheme = vertical_scheme(advection)
    td     = TimeSteppers.time_discretization(scheme)
    k   = k′ + 1
    wⁱ  = implicit_vertical_velocity(ℓx, ℓy, i, j, k, grid, scheme, td, w)
    Azᵢ = Az(i, j, k, grid, ℓx, ℓy, Face())
    ϖᶠ  = densityᶜᶜᶠ(i, j, k, grid, density)
    ϖᶜ  = densityᶜᶜᶜ(i, j, k-1, grid, density)
    V⁻¹ = 1 / volume(i, j, k, grid, ℓx, ℓy, Center())
    return - Δt * V⁻¹ * Azᵢ * ϖᶠ / ϖᶜ * max(wⁱ, zero(wⁱ)) * !peripheral_node(i, j, k′, grid, ℓx, ℓy, Center())
end

@inline function implicit_advection_diagonal(i, j, k, grid, advection::AIVA, w, Δt, ℓx, ℓy, density=nothing)
    scheme = vertical_scheme(advection)
    td     = TimeSteppers.time_discretization(scheme)
    wⁱ⁺ = implicit_vertical_velocity(ℓx, ℓy, i, j, k+1, grid, scheme, td, w)
    wⁱ⁻ = implicit_vertical_velocity(ℓx, ℓy, i, j, k,   grid, scheme, td, w)

    Az⁺ = Az(i, j, k+1, grid, ℓx, ℓy, Face())
    Az⁻ = Az(i, j, k,   grid, ℓx, ℓy, Face())

    ϖᶠ⁺ = densityᶜᶜᶠ(i, j, k+1, grid, density)
    ϖᶠ⁻ = densityᶜᶜᶠ(i, j, k,   grid, density)
    ϖᶜ  = densityᶜᶜᶜ(i, j, k,   grid, density)   # reconstructed value at cell k

    active⁺ = !peripheral_node(i, j, k+1, grid, ℓx, ℓy, Face())
    active⁻ = !peripheral_node(i, j, k,   grid, ℓx, ℓy, Face())

    V⁻¹ = 1 / volume(i, j, k, grid, ℓx, ℓy, Center())

    return Δt * V⁻¹ / ϖᶜ * (Az⁺ * ϖᶠ⁺ * max(wⁱ⁺, zero(wⁱ⁺)) * active⁺ -
                            Az⁻ * ϖᶠ⁻ * min(wⁱ⁻, zero(wⁱ⁻)) * active⁻)
end
