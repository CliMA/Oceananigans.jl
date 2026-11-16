using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: XFaceField, YFaceField, ZFaceField
using Oceananigans.Grids: topology
using Oceananigans.Utils: KernelParameters

#####
##### Flux-corrected transport for bounds-preserving WENO
#####
#
# The rescaling of Zhang and Shu (2010) keeps a forward Euler update within `bounds` only while the
# Courant number stays below ω̂₁. Above it, the flux limiter applies flux-corrected transport (Zalesak, 1979)
# with the global bounds: every face carries the first-order upwind flux Jᵘ, which keeps the update within
# `bounds` for outgoing Courant numbers up to one, plus a fraction ϑ ∈ [0, 1] of the antidiffusive flux
# A = Jᴴ - Jᵘ, where Jᴴ is the rescaled high-order flux. Both cells sharing a face see the same flux
# Jᵘ + ϑA, so the limited fluxes conserve the tracer. Where ϑ = 1 the scheme is the bounds-preserving WENO
# scheme, and where ϑ = 0 it is first-order upwind.
#
# Bounds. Write Jᵐ = ρ u A m and Jᴹ = ρ u A M for the fluxes that transport the bounds. With continuity,
# ρ' = ρ - Δt V⁻¹ ∇⋅(ρu), the forward Euler update (ρc)' = ρc - Δt V⁻¹ ∇⋅J of cell i satisfies
#
#     ρ'(c'ᵢ - m) = Qᵢ⁻ - Δt V⁻¹ Σ_faces ±ϑA,    Qᵢ⁻ = ρ(cᵢ - m) - Δt V⁻¹ ∇⋅(Jᵘ - Jᵐ),
#     ρ'(M - c'ᵢ) = Qᵢ⁺ + Δt V⁻¹ Σ_faces ±ϑA,    Qᵢ⁺ = ρ(M - cᵢ) + Δt V⁻¹ ∇⋅(Jᵘ - Jᴹ),
#
# where Q⁻ and Q⁺ are the room the upwind update leaves above m and below M. For divergence-free transport
# ∇⋅Jᵐ = ∇⋅Jᴹ = 0 and Q⁻ = c̃ - m, Q⁺ = M - c̃ with c̃ the upwind provisional solution; both are non-negative
# when the outgoing Courant number of every cell is at most one.
#
# Fractions. Each cell carries the fraction R⁻ = min(1, Q⁻ / P⁻) of its outgoing antidiffusive fluxes, with
# P⁻ = Δt V⁻¹ Σ (outgoing A), its lower bound affords, and the fraction R⁺ = min(1, Q⁺ / P⁺) of its incoming
# antidiffusive fluxes, with P⁺ = Δt V⁻¹ Σ (incoming A), its upper bound affords. A face takes ϑ = min(R⁻ of
# the cell A leaves, R⁺ of the cell A enters), so Σ_out ϑA ≤ R⁻ P⁻ ≤ Q⁻ and Σ_in ϑA ≤ R⁺ P⁺ ≤ Q⁺, and c'ᵢ
# lies in [m, M]. This is a single limiting pass; Q⁻ and Q⁺ ignore the antidiffusive fluxes in the other
# direction, so the limiter may withhold more than strictly necessary, but never too little.
#
# Where the outgoing Courant number exceeds one the upwind update itself leaves the bounds; Q is then
# clipped at zero, the limiter falls back to upwind, and the bounds are not guaranteed.

"""
    FluxLimiter()

The `flux_limiter` of a [`BoundsPreservation`](@ref), which keeps the tracer within the bounds `(cᵐⁱⁿ, cᵐᵃˣ)`
for outgoing Courant numbers up to one with flux-corrected transport (Zalesak, 1979). Each face flux is the
first-order upwind flux plus a fraction ``ϑ`` of the antidiffusive flux, the difference between the
bounds-preserving WENO flux and the upwind flux. Each cell carries two fractions: the fraction
``R⁻ = min(1, Q⁻ / P⁻)`` of its outgoing antidiffusive fluxes ``P⁻`` that the room ``Q⁻`` above `cᵐⁱⁿ`
left by the upwind update affords, and likewise the fraction ``R⁺ = min(1, Q⁺ / P⁺)`` of its incoming
antidiffusive fluxes that fit below `cᵐᵃˣ`. A face takes ``ϑ`` as the smaller of the outgoing fraction of the
cell the antidiffusive flux leaves and the incoming fraction of the cell it enters. Where ``ϑ = 1`` the scheme
is bounds-preserving WENO, and where ``ϑ = 0`` it is first-order upwind.

The fractions and the face fluxes they are built from are computed by [`update_flux_limiter!`](@ref) before
the fluxes are evaluated; the fields are `nothing` until the scheme is materialized on a grid.
"""
struct FluxLimiter{R, F}
    outgoing_fraction :: R  # R⁻, the fraction of each cell's outgoing fluxes its lower bound affords
    incoming_fraction :: R  # R⁺, the fraction of each cell's incoming fluxes its upper bound affords
    fluxes :: F             # the antidiffusive face fluxes (x, y, z) A = Jᴴ - Jᵘ the fractions are built from
end

FluxLimiter() = FluxLimiter(nothing, nothing, nothing)

Adapt.adapt_structure(to, limiter::FluxLimiter) =
    FluxLimiter(Adapt.adapt(to, limiter.outgoing_fraction),
                Adapt.adapt(to, limiter.incoming_fraction),
                Adapt.adapt(to, limiter.fluxes))

Architectures.on_architecture(to, limiter::FluxLimiter) =
    FluxLimiter(on_architecture(to, limiter.outgoing_fraction),
                on_architecture(to, limiter.incoming_fraction),
                on_architecture(to, limiter.fluxes))

const FluxLimitedBounds = BoundsPreservation{<:Any, <:Any, <:Any, <:FluxLimiter}
const FluxLimitedWENO = WENO{<:Any, <:Any, <:Any, <:Any, <:FluxLimitedBounds}

flux_limiter_summary(::FluxLimitedBounds) = ", flux_limiter=FluxLimiter()"

#####
##### Upwind and bound-transporting face fluxes, and the cached antidiffusive fluxes
#####

# The flux ρ u A b that transports the uniform value b through a face, and the first-order upwind flux of c,
# with the same velocity as the high-order flux (in z, scaled to its explicit part).
@inline bound_flux_x(i, j, k, grid::XFlatGrid, advection, ρ, u, b) = zero(grid)
@inline bound_flux_y(i, j, k, grid::YFlatGrid, advection, ρ, v, b) = zero(grid)
@inline bound_flux_z(i, j, k, grid::ZFlatGrid, advection, ρ, w, b) = zero(grid)
@inline upwind_face_flux_x(i, j, k, grid::XFlatGrid, advection, ρ, u, c) = zero(grid)
@inline upwind_face_flux_y(i, j, k, grid::YFlatGrid, advection, ρ, v, c) = zero(grid)
@inline upwind_face_flux_z(i, j, k, grid::ZFlatGrid, advection, ρ, w, c) = zero(grid)

@inline transport_x(i, j, k, grid, advection, ρ, u) = @inbounds ℑxᶠᵃᵃ(i, j, k, grid, ρ) * Axᶠᶜᶜ(i, j, k, grid) * u[i, j, k]
@inline transport_y(i, j, k, grid, advection, ρ, v) = @inbounds ℑyᵃᶠᵃ(i, j, k, grid, ρ) * Ayᶜᶠᶜ(i, j, k, grid) * v[i, j, k]

@inline function transport_z(i, j, k, grid, advection, ρ, w)
    td = advection.time_discretization
    w̃ = @inbounds explicit_velocity_scaleᶜᶜᶠ(i, j, k, grid, advection, td, w) * w[i, j, k]
    return ℑzᵃᵃᶠ(i, j, k, grid, ρ) * Azᶜᶜᶠ(i, j, k, grid) * w̃
end

@inline bound_flux_x(i, j, k, grid, advection, ρ, u, b) = transport_x(i, j, k, grid, advection, ρ, u) * b
@inline bound_flux_y(i, j, k, grid, advection, ρ, v, b) = transport_y(i, j, k, grid, advection, ρ, v) * b
@inline bound_flux_z(i, j, k, grid, advection, ρ, w, b) = transport_z(i, j, k, grid, advection, ρ, w) * b

@inline upwind_face_flux_x(i, j, k, grid, advection, ρ, u, c) =
    @inbounds upwind_biased_product(transport_x(i, j, k, grid, advection, ρ, u), c[i-1, j, k], c[i, j, k])

@inline upwind_face_flux_y(i, j, k, grid, advection, ρ, v, c) =
    @inbounds upwind_biased_product(transport_y(i, j, k, grid, advection, ρ, v), c[i, j-1, k], c[i, j, k])

@inline upwind_face_flux_z(i, j, k, grid, advection, ρ, w, c) =
    @inbounds upwind_biased_product(transport_z(i, j, k, grid, advection, ρ, w), c[i, j, k-1], c[i, j, k])

# The faces the interior cells' divergences read: one more than the cells in every non-Flat direction.
function face_flux_kernel_parameters(grid)
    N = size(grid)
    T = topology(grid)
    faces = ntuple(d -> T[d] === Flat ? N[d] : N[d] + 1, 3)
    return KernelParameters(faces, (0, 0, 0))
end

@kernel function _compute_antidiffusive_face_fluxes!(fluxes, grid, advection, ρ, U, c)
    i, j, k = @index(Global, NTuple)
    @inbounds fluxes.x[i, j, k] = bounded_face_flux_x(i, j, k, grid, advection, ρ, U.u, c) - upwind_face_flux_x(i, j, k, grid, advection, ρ, U.u, c)
    @inbounds fluxes.y[i, j, k] = bounded_face_flux_y(i, j, k, grid, advection, ρ, U.v, c) - upwind_face_flux_y(i, j, k, grid, advection, ρ, U.v, c)
    @inbounds fluxes.z[i, j, k] = bounded_face_flux_z(i, j, k, grid, advection, ρ, U.w, c) - upwind_face_flux_z(i, j, k, grid, advection, ρ, U.w, c)
end

#####
##### Limited face fluxes and their divergence
#####

# The factor of a face: the outgoing fraction of the cell the antidiffusive flux leaves and the incoming fraction
# of the cell it enters, which the flux's sign identifies, whichever is smaller.
@inline function face_factor(E, R⁻, R⁺, i⁻, j⁻, k⁻, i⁺, j⁺, k⁺)
    @inbounds ϑ⁺ = min(R⁻[i⁻, j⁻, k⁻], R⁺[i⁺, j⁺, k⁺]) # flow from the (i⁻, j⁻, k⁻) cell into the (i⁺, j⁺, k⁺) cell
    @inbounds ϑ⁻ = min(R⁻[i⁺, j⁺, k⁺], R⁺[i⁻, j⁻, k⁻])
    return ifelse(E ≥ 0, ϑ⁺, ϑ⁻)
end

@inline function limited_face_flux_x(i, j, k, grid, advection, ρ, u, c)
    limiter = advection.bounds.flux_limiter
    E = @inbounds limiter.fluxes.x[i, j, k]
    ϑ = face_factor(E, limiter.outgoing_fraction, limiter.incoming_fraction, i-1, j, k, i, j, k)
    return upwind_face_flux_x(i, j, k, grid, advection, ρ, u, c) + ϑ * E
end

@inline function limited_face_flux_y(i, j, k, grid, advection, ρ, v, c)
    limiter = advection.bounds.flux_limiter
    E = @inbounds limiter.fluxes.y[i, j, k]
    ϑ = face_factor(E, limiter.outgoing_fraction, limiter.incoming_fraction, i, j-1, k, i, j, k)
    return upwind_face_flux_y(i, j, k, grid, advection, ρ, v, c) + ϑ * E
end

@inline function limited_face_flux_z(i, j, k, grid, advection, ρ, w, c)
    limiter = advection.bounds.flux_limiter
    E = @inbounds limiter.fluxes.z[i, j, k]
    ϑ = face_factor(E, limiter.outgoing_fraction, limiter.incoming_fraction, i, j, k-1, i, j, k)
    return upwind_face_flux_z(i, j, k, grid, advection, ρ, w, c) + ϑ * E
end

# Support for Flat directions, which have no neighbors to index
@inline limited_tracer_flux_divergence_x(i, j, k, grid::XFlatGrid, advection, ρ, u, c) = zero(grid)
@inline limited_tracer_flux_divergence_y(i, j, k, grid::YFlatGrid, advection, ρ, v, c) = zero(grid)
@inline limited_tracer_flux_divergence_z(i, j, k, grid::ZFlatGrid, advection, ρ, w, c) = zero(grid)

@inline limited_tracer_flux_divergence_x(i, j, k, grid, advection, ρ, u, c) =
    limited_face_flux_x(i+1, j, k, grid, advection, ρ, u, c) - limited_face_flux_x(i, j, k, grid, advection, ρ, u, c)

@inline limited_tracer_flux_divergence_y(i, j, k, grid, advection, ρ, v, c) =
    limited_face_flux_y(i, j+1, k, grid, advection, ρ, v, c) - limited_face_flux_y(i, j, k, grid, advection, ρ, v, c)

@inline limited_tracer_flux_divergence_z(i, j, k, grid, advection, ρ, w, c) =
    limited_face_flux_z(i, j, k+1, grid, advection, ρ, w, c) - limited_face_flux_z(i, j, k, grid, advection, ρ, w, c)

@inline div_Uc(i, j, k, grid, advection::FluxLimitedWENO, U, ::ZeroField) = zero(grid)
@inline div_Uc(i, j, k, grid, advection::FluxLimitedWENO, ::ZeroU, c) = zero(grid)
@inline div_Uc(i, j, k, grid, advection::FluxLimitedWENO, ::ZeroU, ::ZeroField) = zero(grid)

# The divergence of the upwind fluxes of `c` plus the cached antidiffusive fluxes `update_flux_limiter!` built
# from `U` and `c`, each scaled by its factor. With every fraction equal to one this is the bounds-preserving divergence.
@inline function div_Uc(i, j, k, grid, advection::FluxLimitedWENO, U, c)
    div_x = limited_tracer_flux_divergence_x(i, j, k, grid, advection, 1, U.u, c)
    div_y = limited_tracer_flux_divergence_y(i, j, k, grid, advection, 1, U.v, c)
    div_z = limited_tracer_flux_divergence_z(i, j, k, grid, advection, 1, U.w, c)

    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (div_x + div_y + div_z)
end

#####
##### Fractions
#####

# Flat directions have no faces to sum over and no neighbors to credit
@inline outgoing_flux_x(i, j, k, grid::XFlatGrid, fluxes) = zero(grid)
@inline outgoing_flux_y(i, j, k, grid::YFlatGrid, fluxes) = zero(grid)
@inline outgoing_flux_z(i, j, k, grid::ZFlatGrid, fluxes) = zero(grid)
@inline incoming_flux_x(i, j, k, grid::XFlatGrid, fluxes) = zero(grid)
@inline incoming_flux_y(i, j, k, grid::YFlatGrid, fluxes) = zero(grid)
@inline incoming_flux_z(i, j, k, grid::ZFlatGrid, fluxes) = zero(grid)

@inline outgoing_flux_x(i, j, k, grid, fluxes) = @inbounds max(0, fluxes.x[i+1, j, k]) - min(0, fluxes.x[i, j, k])
@inline outgoing_flux_y(i, j, k, grid, fluxes) = @inbounds max(0, fluxes.y[i, j+1, k]) - min(0, fluxes.y[i, j, k])
@inline outgoing_flux_z(i, j, k, grid, fluxes) = @inbounds max(0, fluxes.z[i, j, k+1]) - min(0, fluxes.z[i, j, k])
@inline incoming_flux_x(i, j, k, grid, fluxes) = @inbounds max(0, fluxes.x[i, j, k]) - min(0, fluxes.x[i+1, j, k])
@inline incoming_flux_y(i, j, k, grid, fluxes) = @inbounds max(0, fluxes.y[i, j, k]) - min(0, fluxes.y[i, j+1, k])
@inline incoming_flux_z(i, j, k, grid, fluxes) = @inbounds max(0, fluxes.z[i, j, k]) - min(0, fluxes.z[i, j, k+1])

# P⁻ and P⁺: the antidiffusive fluxes leaving and entering cell (i, j, k), per unit volume, over Δt
@inline function outgoing_antidiffusion(i, j, k, grid, fluxes, Δt)
    outgoing = outgoing_flux_x(i, j, k, grid, fluxes) + outgoing_flux_y(i, j, k, grid, fluxes) + outgoing_flux_z(i, j, k, grid, fluxes)
    return Δt * V⁻¹ᶜᶜᶜ(i, j, k, grid) * outgoing
end

@inline function incoming_antidiffusion(i, j, k, grid, fluxes, Δt)
    incoming = incoming_flux_x(i, j, k, grid, fluxes) + incoming_flux_y(i, j, k, grid, fluxes) + incoming_flux_z(i, j, k, grid, fluxes)
    return Δt * V⁻¹ᶜᶜᶜ(i, j, k, grid) * incoming
end

# The density of cell (i, j, k): a field, or the number `1` of `div_Uc`
@inline cell_density(i, j, k, ρ::Number) = ρ
@inline cell_density(i, j, k, ρ) = @inbounds ρ[i, j, k]

# The divergence ∇⋅(Jᵘ - Jᵇ) of the upwind fluxes of c relative to the fluxes transporting the value b
@inline function relative_upwind_divergence(i, j, k, grid, advection, ρ, U, c, b)
    div_x = (upwind_face_flux_x(i+1, j, k, grid, advection, ρ, U.u, c) - bound_flux_x(i+1, j, k, grid, advection, ρ, U.u, b)) -
            (upwind_face_flux_x(i,   j, k, grid, advection, ρ, U.u, c) - bound_flux_x(i,   j, k, grid, advection, ρ, U.u, b))
    div_y = (upwind_face_flux_y(i, j+1, k, grid, advection, ρ, U.v, c) - bound_flux_y(i, j+1, k, grid, advection, ρ, U.v, b)) -
            (upwind_face_flux_y(i, j,   k, grid, advection, ρ, U.v, c) - bound_flux_y(i, j,   k, grid, advection, ρ, U.v, b))
    div_z = (upwind_face_flux_z(i, j, k+1, grid, advection, ρ, U.w, c) - bound_flux_z(i, j, k+1, grid, advection, ρ, U.w, b)) -
            (upwind_face_flux_z(i, j, k,   grid, advection, ρ, U.w, c) - bound_flux_z(i, j, k,   grid, advection, ρ, U.w, b))
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (div_x + div_y + div_z)
end

# Q⁻ and Q⁺: the room the upwind update leaves above cᵐⁱⁿ and below cᵐᵃˣ, in the units of ρ c
@inline function upwind_room(i, j, k, grid, advection, ρ, U, c, Δt)
    bounds = advection.bounds
    m = bounds.minimum_value
    M = bounds.maximum_value
    ρᶜ = cell_density(i, j, k, ρ)
    @inbounds cᵢ = c[i, j, k]
    Q⁻ = ρᶜ * (cᵢ - m) - Δt * relative_upwind_divergence(i, j, k, grid, advection, ρ, U, c, m)
    Q⁺ = ρᶜ * (M - cᵢ) + Δt * relative_upwind_divergence(i, j, k, grid, advection, ρ, U, c, M)
    return max(0, Q⁻), max(0, Q⁺)
end

# min(1, available / transported), guarded against non-finite fluxes
@inline flux_fraction(transported, available) =
    ifelse(isfinite(transported) & (transported > available), available / transported, one(available))

# R⁻ = min(1, Q⁻ / P⁻) and R⁺ = min(1, Q⁺ / P⁺)
@kernel function _compute_flux_fractions!(R⁻, R⁺, grid, advection, fluxes, U, c, ρ, Δt)
    i, j, k = @index(Global, NTuple)
    Q⁻, Q⁺ = upwind_room(i, j, k, grid, advection, ρ, U, c, Δt)
    @inbounds R⁻[i, j, k] = flux_fraction(outgoing_antidiffusion(i, j, k, grid, fluxes, Δt), Q⁻)
    @inbounds R⁺[i, j, k] = flux_fraction(incoming_antidiffusion(i, j, k, grid, fluxes, Δt), Q⁺)
end

"""
$(TYPEDSIGNATURES)

Refresh a flux-limited `scheme` (see [`FluxLimiter`](@ref)) for the coming forward Euler update
`c - Δt * div_Uc(c)` of a tracer: cache the antidiffusive face fluxes of `c`, the bounds-preserving WENO fluxes
minus the first-order upwind fluxes, and compute the fractions of each cell's outgoing and incoming
antidiffusive fluxes the update applies. `U` are the velocities the fluxes
transport with and `ρ` the density weighting the fluxes, which `div_Uc` takes as `1`.

A no-op for every other scheme.
"""
update_flux_limiter!(scheme, grid, c, U, Δt, ρ=1) = nothing

function update_flux_limiter!(scheme::FluxLimitedWENO, grid, c, U, Δt, ρ=1)
    limiter = scheme.bounds.flux_limiter
    fluxes = limiter.fluxes
    R⁻ = limiter.outgoing_fraction
    R⁺ = limiter.incoming_fraction
    parameters = face_flux_kernel_parameters(grid)
    launch!(architecture(grid), grid, parameters, _compute_antidiffusive_face_fluxes!, fluxes, grid, scheme, ρ, U, c)
    launch!(architecture(grid), grid, :xyz, _compute_flux_fractions!, R⁻, R⁺, grid, scheme, fluxes, U, c, ρ, Δt)
    fill_halo_regions!(R⁻)
    fill_halo_regions!(R⁺)
    return nothing
end
