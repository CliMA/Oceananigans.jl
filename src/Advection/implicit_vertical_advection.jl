using Oceananigans.Grids: peripheral_node, Center, Face
using Oceananigans.Operators: Az, volume, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ

@inline vertical_scheme(advection) = advection
@inline vertical_scheme(advection::VectorInvariant) = advection.vertical_advection_scheme
@inline vertical_scheme(advection::FluxFormAdvection) = advection.z

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

@inline function implicit_vertical_velocityᶜᶜᶜ(i, j, k, grid, scheme, td, W)
    Δt = _unwrap_for_gpu(td.Δt)
    Δz = Δzᶜᶜᶜ(i, j, k, grid)
    w  = _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, W)
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
##### The upwind flux at face k+1 (top of cell k), weighted by the face density ρᶠ:
#####   F_{k+1} = Az_{k+1} ρᶠ_{k+1} * [max(wⁱ_{k+1}, 0) * c_k + min(wⁱ_{k+1}, 0) * c_{k+1}],   c = q / ρ
#####
##### The implicit system (I - Δt * L) qⁿ⁺¹ = q★ gives (with ρᶜ the cell density of the reconstructed value):
#####
##### Upper diagonal (coeff of q_{k+1}):   Δt / V_k * Az_{k+1} ρᶠ_{k+1} / ρᶜ_{k+1} * min(wⁱ_{k+1}, 0)
##### Lower diagonal (coeff of q_{k-1}): - Δt / V_k * Az_k     ρᶠ_k     / ρᶜ_{k-1} * max(wⁱ_k, 0)
#####
##### With `density === nothing`, ρᶠ = ρᶜ = 1 and these reduce to the volume-conserving coefficients.
#####

@inline implicit_vertical_velocity(::Center, ::Center, args...) = implicit_vertical_velocityᶜᶜᶠ(args...)
@inline implicit_vertical_velocity(::Face,   ::Center, args...) = implicit_vertical_velocityᶠᶜᶠ(args...)
@inline implicit_vertical_velocity(::Center, ::Face,   args...) = implicit_vertical_velocityᶜᶠᶠ(args...)

# An advection scheme without an adaptive-implicit vertical discretization contributes nothing.
# `ℓz` is typed so that these cannot absorb a call that omits it: an untyped slot would make a stale
# argument order return zero rather than raise a MethodError.
const AdvectionOrNothing = Union{Nothing, AbstractAdvectionScheme}
const CenterOrFace = Union{Center, Face}

@inline implicit_advection_upper_diagonal(i, j, k, grid, ::AdvectionOrNothing, w, Δt, ℓx, ℓy, ℓz::CenterOrFace, density=nothing) = zero(grid)
@inline implicit_advection_lower_diagonal(i, j, k, grid, ::AdvectionOrNothing, w, Δt, ℓx, ℓy, ℓz::CenterOrFace, density=nothing) = zero(grid)
@inline implicit_advection_diagonal(i, j, k, grid,       ::AdvectionOrNothing, w, Δt, ℓx, ℓy, ℓz::CenterOrFace, density=nothing) = zero(grid)

# Upper diagonal: coefficient of q_{k+1} in the tridiagonal system
@inline function implicit_advection_upper_diagonal(i, j, k, grid, advection::AIVA, w, Δt, ℓx, ℓy, ℓz::Center, density=nothing)
    scheme = vertical_scheme(advection)
    td  = TimeSteppers.time_discretization(scheme)
    wⁱ  = implicit_vertical_velocity(ℓx, ℓy, i, j, k+1, grid, scheme, td, w)
    Azᵢ = Az(i, j, k+1, grid, ℓx, ℓy, Face())
    ρᶠ  = densityᶜᶜᶠ(i, j, k+1, grid, density)
    ρᶜ  = densityᶜᶜᶜ(i, j, k+1, grid, density)
    V⁻¹ = 1 / volume(i, j, k, grid, ℓx, ℓy, Center())
    return Δt * V⁻¹ * Azᵢ * ρᶠ / ρᶜ * min(wⁱ, zero(wⁱ)) * !peripheral_node(i, j, k+1, grid, ℓx, ℓy, Face())
end

# Lower diagonal: coefficient of q_{k-1} in the tridiagonal system
# Uses k′ = k-1 indexing convention (LinearAlgebra.Tridiagonal convention, matching ivd_lower_diagonal)
@inline function implicit_advection_lower_diagonal(i, j, k′, grid, advection::AIVA, w, Δt, ℓx, ℓy, ℓz::Center, density=nothing)
    scheme = vertical_scheme(advection)
    td  = TimeSteppers.time_discretization(scheme)
    k   = k′ + 1
    wⁱ  = implicit_vertical_velocity(ℓx, ℓy, i, j, k, grid, scheme, td, w)
    Azᵢ = Az(i, j, k, grid, ℓx, ℓy, Face())
    ρᶠ  = densityᶜᶜᶠ(i, j, k, grid, density)
    ρᶜ  = densityᶜᶜᶜ(i, j, k-1, grid, density)
    V⁻¹ = 1 / volume(i, j, k, grid, ℓx, ℓy, Center())
    return - Δt * V⁻¹ * Azᵢ * ρᶠ / ρᶜ * max(wⁱ, zero(wⁱ)) * !peripheral_node(i, j, k′, grid, ℓx, ℓy, Center())
end

@inline function implicit_advection_diagonal(i, j, k, grid, advection::AIVA, w, Δt, ℓx, ℓy, ℓz::Center, density=nothing)
    scheme = vertical_scheme(advection)
    td     = TimeSteppers.time_discretization(scheme)
    wⁱ⁺ = implicit_vertical_velocity(ℓx, ℓy, i, j, k+1, grid, scheme, td, w)
    wⁱ⁻ = implicit_vertical_velocity(ℓx, ℓy, i, j, k,   grid, scheme, td, w)

    Az⁺ = Az(i, j, k+1, grid, ℓx, ℓy, Face())
    Az⁻ = Az(i, j, k,   grid, ℓx, ℓy, Face())

    ρᶠ⁺ = densityᶜᶜᶠ(i, j, k+1, grid, density)
    ρᶠ⁻ = densityᶜᶜᶠ(i, j, k,   grid, density)
    ρᶜ  = densityᶜᶜᶜ(i, j, k,   grid, density)   # reconstructed value at cell k

    active⁺ = !peripheral_node(i, j, k+1, grid, ℓx, ℓy, Face())
    active⁻ = !peripheral_node(i, j, k,   grid, ℓx, ℓy, Face())

    V⁻¹ = 1 / volume(i, j, k, grid, ℓx, ℓy, Center())

    return Δt * V⁻¹ / ρᶜ * (Az⁺ * ρᶠ⁺ * max(wⁱ⁺, zero(wⁱ⁺)) * active⁺ -
                            Az⁻ * ρᶠ⁻ * min(wⁱ⁻, zero(wⁱ⁻)) * active⁻)
end

#####
##### Tridiagonal coefficients for fields at cell Faces in z (`w`).
#####
##### Row k is the control volume around face k, bounded by the cell centers k-1 and k where the
##### upwind fluxes live. With `ρᶜ` the cell density and `ρᶠ` the density at the reconstructed face:
#####
#####   Fⁱ_k = Az_k ρᶜ_k [max(wⁱ_k, 0) q_k / ρᶠ_k + min(wⁱ_k, 0) q_{k+1} / ρᶠ_{k+1}],   q = ρ w
#####
##### Boundary faces reduce to identity rows, and the coupling of row Nz to the untouched face
##### Nz+1 vanishes with `w` there.
#####

@inline function implicit_advection_upper_diagonal(i, j, k, grid, advection::AIVA, w, Δt, ℓx, ℓy, ℓz::Face, density=nothing)
    scheme = vertical_scheme(advection)
    td  = TimeSteppers.time_discretization(scheme)
    wⁱ  = implicit_vertical_velocityᶜᶜᶜ(i, j, k, grid, scheme, td, w)
    Azᵢ = Az(i, j, k, grid, ℓx, ℓy, Center())
    ρᶜ  = densityᶜᶜᶜ(i, j, k, grid, density)
    ρᶠ  = densityᶜᶜᶠ(i, j, k+1, grid, density)
    V⁻¹ = 1 / volume(i, j, k, grid, ℓx, ℓy, Face())
    active = !peripheral_node(i, j, k, grid, ℓx, ℓy, Face()) & !peripheral_node(i, j, k, grid, ℓx, ℓy, Center())
    return Δt * V⁻¹ * Azᵢ * ρᶜ / ρᶠ * min(wⁱ, zero(wⁱ)) * active
end

@inline function implicit_advection_lower_diagonal(i, j, k′, grid, advection::AIVA, w, Δt, ℓx, ℓy, ℓz::Face, density=nothing)
    scheme = vertical_scheme(advection)
    td  = TimeSteppers.time_discretization(scheme)
    k   = k′ + 1
    wⁱ  = implicit_vertical_velocityᶜᶜᶜ(i, j, k′, grid, scheme, td, w)
    Azᵢ = Az(i, j, k′, grid, ℓx, ℓy, Center())
    ρᶜ  = densityᶜᶜᶜ(i, j, k′, grid, density)
    ρᶠ  = densityᶜᶜᶠ(i, j, k′, grid, density)
    V⁻¹ = 1 / volume(i, j, k, grid, ℓx, ℓy, Face())
    active = !peripheral_node(i, j, k, grid, ℓx, ℓy, Face()) & !peripheral_node(i, j, k′, grid, ℓx, ℓy, Center())
    return - Δt * V⁻¹ * Azᵢ * ρᶜ / ρᶠ * max(wⁱ, zero(wⁱ)) * active
end

@inline function implicit_advection_diagonal(i, j, k, grid, advection::AIVA, w, Δt, ℓx, ℓy, ℓz::Face, density=nothing)
    scheme = vertical_scheme(advection)
    td     = TimeSteppers.time_discretization(scheme)
    wⁱ⁺ = implicit_vertical_velocityᶜᶜᶜ(i, j, k,   grid, scheme, td, w)
    wⁱ⁻ = implicit_vertical_velocityᶜᶜᶜ(i, j, k-1, grid, scheme, td, w)

    Az⁺ = Az(i, j, k,   grid, ℓx, ℓy, Center())
    Az⁻ = Az(i, j, k-1, grid, ℓx, ℓy, Center())

    ρᶜ⁺ = densityᶜᶜᶜ(i, j, k,   grid, density)
    ρᶜ⁻ = densityᶜᶜᶜ(i, j, k-1, grid, density)
    ρᶠ  = densityᶜᶜᶠ(i, j, k,   grid, density)

    active⁺ = !peripheral_node(i, j, k,   grid, ℓx, ℓy, Center())
    active⁻ = !peripheral_node(i, j, k-1, grid, ℓx, ℓy, Center())
    active  = !peripheral_node(i, j, k,   grid, ℓx, ℓy, Face())

    V⁻¹ = 1 / volume(i, j, k, grid, ℓx, ℓy, Face())

    return Δt * V⁻¹ / ρᶠ * (Az⁺ * ρᶜ⁺ * max(wⁱ⁺, zero(wⁱ⁺)) * active⁺ -
                            Az⁻ * ρᶜ⁻ * min(wⁱ⁻, zero(wⁱ⁻)) * active⁻) * active
end
