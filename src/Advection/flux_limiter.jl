using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: XFaceField, YFaceField, ZFaceField
using Oceananigans.Grids: topology
using Oceananigans.Utils: KernelParameters

#####
##### Flux limiting for bounds-preserving WENO
#####
#
# The rescaling of Zhang and Shu (2010) keeps a forward Euler update within `bounds` only while the
# Courant number stays below ω̂₁; a larger one can remove more from a cell than it holds above the lower
# bound, or bring more into a cell than it has room for below the upper bound. The flux limiter
# scales the part of each face flux that moves tracer relative to the lower bound by a factor ϑ ∈ [0, 1]
# the coming update can afford: a conservative, zeroth-order flux-corrected transport limiter in the
# two-sided form of Zalesak (1979), which keeps the tracer within `bounds` for any time step, at
# first-order accuracy where it acts.
#
# Fluxes. With face values c̃ ∈ [m, M] (which the rescaling guarantees from a state in [m, M]), the
# advective flux J = ρ u A c̃ through a face splits as J = Jᵐ + E, where Jᵐ = ρ u A m transports the lower
# bound and E = ρ u A (c̃ - m) ≥ 0 in the direction of the flow is the excess over it. The limited flux is
# J̃ = Jᵐ + ϑ E. Both cells sharing a face see the same J̃, so the limited fluxes conserve the tracer.
# For a cell, Σ ±Jᵐ = m Δt (∇⋅ρu) V vanishes for divergence-free transport (or cancels against the change
# of ρ when ρ obeys continuity), so with Oᵢ = Δt V⁻¹ Σ (outgoing E) and Iᵢ = Δt V⁻¹ Σ (incoming E),
#
#     c'ᵢ - m = (cᵢ - m) - Σ_out ϑ E Δt V⁻¹ + Σ_in ϑ E Δt V⁻¹,
#     M - c'ᵢ = (M - cᵢ) + Σ_out ϑ E Δt V⁻¹ - Σ_in ϑ E Δt V⁻¹
#
# for the forward Euler update c' of c.
#
# For divergent transport the floor flux violates the bounds by m Δt ∇⋅u, exactly as the unlimited scheme
# fails to preserve a uniform tracer there.
#
# Fractions. Each cell carries an outgoing fraction R⁻ (its lower bound) and an incoming fraction R⁺ (its
# upper bound), and a face takes ϑ = min(R⁻ of the cell it leaves, R⁺ of the cell it enters), so that
# Σ_out ϑ E ≤ R⁻ᵢ Oᵢ and Σ_in ϑ E ≤ R⁺ᵢ Iᵢ. Writing x = (R⁻, R⁺) and
#
#     G⁻(x)ᵢ = min(1, (cᵢ - m + Σ_in ϑ(x) E Δt V⁻¹) / Oᵢ),
#     G⁺(x)ᵢ = min(1, (M - cᵢ + Σ_out ϑ(x) E Δt V⁻¹) / Iᵢ),
#
# the two displays above give c'ᵢ ∈ [m, M] whenever R⁻ ≤ G⁻(x) and R⁺ ≤ G⁺(x). G = (G⁻, G⁺) is nondecreasing
# in x, and x = 0 satisfies x ≤ G(0) when c ∈ [m, M]. If x ≤ G(x) and some components are replaced by
# G̃(x)ᵢ, where G̃ ≤ G credits a subset of the fluxes (at their current factors ϑ(x)) and a cell's credited
# subset only grows between its successive updates, then the fractions never decrease, so for the replaced
# components G̃(x)ᵢ ≤ G(x)ᵢ ≤ G(x')ᵢ and for the others xᵢ ≤ G(x)ᵢ ≤ G(x')ᵢ: x' ≤ G(x') again. Every such
# process therefore ends with x ≤ G(x), that is, with a tracer in [m, M], for any time step and with no
# Courant condition: the guarantee needs only a state c in [m, M] (so that the rescaled face values are) and
# divergence-free transport. The limiter below is the simplest instance, one pass crediting nothing, so it
# withholds more than the least fixed point of G would where a cell's outgoing fluxes exceed its content
# but its incoming fluxes would replace them; crediting them (Jacobi passes, or a Gauss–Seidel sweep along
# each column) is a possible extension.

"""
    FluxLimiter()

The `flux_limiter` of a [`BoundsPreservation`](@ref), which keeps the tracer within the bounds `(cᵐⁱⁿ, cᵐᵃˣ)`
for any time step by limiting the fluxes between cells (Zalesak, 1979). Each cell carries two fractions: the
fraction ``R⁻ = min(1, available / removed)`` of its outgoing fluxes the coming forward Euler update can
afford, where `removed` is the tracer above `cᵐⁱⁿ` the unlimited outgoing fluxes take out of the cell and
`available` its content above `cᵐⁱⁿ`, and likewise the fraction ``R⁺ = min(1, room / supplied)`` of its
incoming fluxes that fit below `cᵐᵃˣ`. A face flux is scaled by the smaller of the outgoing fraction of the
cell it leaves and the incoming fraction of the cell it enters.

The fractions and the face fluxes they are built from are computed by [`update_flux_limiter!`](@ref) before
the fluxes are evaluated; the fields are `nothing` until the scheme is materialized on a grid.
"""
struct FluxLimiter{R, F}
    outgoing_fraction :: R  # R⁻, the fraction of each cell's outgoing fluxes its lower bound affords
    incoming_fraction :: R  # R⁺, the fraction of each cell's incoming fluxes its upper bound affords
    fluxes :: F             # the bounded face fluxes (x, y, z) of c - cᵐⁱⁿ the fractions are built from
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
##### Cached face fluxes of c - cᵐⁱⁿ
#####

# The flux Jᵐ = ρ u A cᵐⁱⁿ that transports the lower bound through a face, which the limiter leaves alone
@inline floor_flux_x(i, j, k, grid::XFlatGrid, ρ, u, cᵐⁱⁿ) = zero(grid)
@inline floor_flux_y(i, j, k, grid::YFlatGrid, ρ, v, cᵐⁱⁿ) = zero(grid)
@inline floor_flux_z(i, j, k, grid::ZFlatGrid, ρ, w, cᵐⁱⁿ) = zero(grid)

@inline floor_flux_x(i, j, k, grid, ρ, u, cᵐⁱⁿ) = @inbounds ℑxᶠᵃᵃ(i, j, k, grid, ρ) * Axᶠᶜᶜ(i, j, k, grid) * u[i, j, k] * cᵐⁱⁿ
@inline floor_flux_y(i, j, k, grid, ρ, v, cᵐⁱⁿ) = @inbounds ℑyᵃᶠᵃ(i, j, k, grid, ρ) * Ayᶜᶠᶜ(i, j, k, grid) * v[i, j, k] * cᵐⁱⁿ
@inline floor_flux_z(i, j, k, grid, ρ, w, cᵐⁱⁿ) = @inbounds ℑzᵃᵃᶠ(i, j, k, grid, ρ) * Azᶜᶜᶠ(i, j, k, grid) * w[i, j, k] * cᵐⁱⁿ

# The faces the interior cells' divergences read: one more than the cells in every non-Flat direction.
function face_flux_kernel_parameters(grid)
    N = size(grid)
    T = topology(grid)
    faces = ntuple(d -> T[d] === Flat ? N[d] : N[d] + 1, 3)
    return KernelParameters(faces, (0, 0, 0))
end

@kernel function _compute_excess_face_fluxes!(fluxes, grid, advection, ρ, U, c)
    i, j, k = @index(Global, NTuple)
    cᵐⁱⁿ = advection.bounds.minimum_value
    @inbounds fluxes.x[i, j, k] = bounded_face_flux_x(i, j, k, grid, advection, ρ, U.u, c) - floor_flux_x(i, j, k, grid, ρ, U.u, cᵐⁱⁿ)
    @inbounds fluxes.y[i, j, k] = bounded_face_flux_y(i, j, k, grid, advection, ρ, U.v, c) - floor_flux_y(i, j, k, grid, ρ, U.v, cᵐⁱⁿ)
    @inbounds fluxes.z[i, j, k] = bounded_face_flux_z(i, j, k, grid, advection, ρ, U.w, c) - floor_flux_z(i, j, k, grid, ρ, U.w, cᵐⁱⁿ)
end

#####
##### Limited face fluxes and their divergence
#####

# The factor of a face: the outgoing fraction of the cell the excess flux leaves and the incoming fraction
# of the cell it enters, which the flux's sign identifies, whichever is smaller.
@inline function face_factor(E, R⁻, R⁺, i⁻, j⁻, k⁻, i⁺, j⁺, k⁺)
    @inbounds ϑ⁺ = min(R⁻[i⁻, j⁻, k⁻], R⁺[i⁺, j⁺, k⁺]) # flow from the (i⁻, j⁻, k⁻) cell into the (i⁺, j⁺, k⁺) cell
    @inbounds ϑ⁻ = min(R⁻[i⁺, j⁺, k⁺], R⁺[i⁻, j⁻, k⁻])
    return ifelse(E ≥ 0, ϑ⁺, ϑ⁻)
end

@inline function limited_face_flux_x(i, j, k, grid, advection, ρ, u)
    limiter = advection.bounds.flux_limiter
    E = @inbounds limiter.fluxes.x[i, j, k]
    ϑ = face_factor(E, limiter.outgoing_fraction, limiter.incoming_fraction, i-1, j, k, i, j, k)
    return floor_flux_x(i, j, k, grid, ρ, u, advection.bounds.minimum_value) + ϑ * E
end

@inline function limited_face_flux_y(i, j, k, grid, advection, ρ, v)
    limiter = advection.bounds.flux_limiter
    E = @inbounds limiter.fluxes.y[i, j, k]
    ϑ = face_factor(E, limiter.outgoing_fraction, limiter.incoming_fraction, i, j-1, k, i, j, k)
    return floor_flux_y(i, j, k, grid, ρ, v, advection.bounds.minimum_value) + ϑ * E
end

@inline function limited_face_flux_z(i, j, k, grid, advection, ρ, w)
    limiter = advection.bounds.flux_limiter
    E = @inbounds limiter.fluxes.z[i, j, k]
    ϑ = face_factor(E, limiter.outgoing_fraction, limiter.incoming_fraction, i, j, k-1, i, j, k)
    return floor_flux_z(i, j, k, grid, ρ, w, advection.bounds.minimum_value) + ϑ * E
end

# Support for Flat directions, which have no neighbors to index
@inline limited_tracer_flux_divergence_x(i, j, k, grid::XFlatGrid, advection, ρ, u) = zero(grid)
@inline limited_tracer_flux_divergence_y(i, j, k, grid::YFlatGrid, advection, ρ, v) = zero(grid)
@inline limited_tracer_flux_divergence_z(i, j, k, grid::ZFlatGrid, advection, ρ, w) = zero(grid)

@inline limited_tracer_flux_divergence_x(i, j, k, grid, advection, ρ, u) =
    limited_face_flux_x(i+1, j, k, grid, advection, ρ, u) - limited_face_flux_x(i, j, k, grid, advection, ρ, u)

@inline limited_tracer_flux_divergence_y(i, j, k, grid, advection, ρ, v) =
    limited_face_flux_y(i, j+1, k, grid, advection, ρ, v) - limited_face_flux_y(i, j, k, grid, advection, ρ, v)

@inline limited_tracer_flux_divergence_z(i, j, k, grid, advection, ρ, w) =
    limited_face_flux_z(i, j, k+1, grid, advection, ρ, w) - limited_face_flux_z(i, j, k, grid, advection, ρ, w)

@inline div_Uc(i, j, k, grid, advection::FluxLimitedWENO, U, ::ZeroField) = zero(grid)
@inline div_Uc(i, j, k, grid, advection::FluxLimitedWENO, ::ZeroU, c) = zero(grid)
@inline div_Uc(i, j, k, grid, advection::FluxLimitedWENO, ::ZeroU, ::ZeroField) = zero(grid)

# The divergence of the floor fluxes of `U` plus the cached excess fluxes `update_flux_limiter!` built from
# `U` and `c`, each scaled by its factor. With every fraction equal to one this is the bounds-preserving divergence.
@inline function div_Uc(i, j, k, grid, advection::FluxLimitedWENO, U, c)
    div_x = limited_tracer_flux_divergence_x(i, j, k, grid, advection, 1, U.u)
    div_y = limited_tracer_flux_divergence_y(i, j, k, grid, advection, 1, U.v)
    div_z = limited_tracer_flux_divergence_z(i, j, k, grid, advection, 1, U.w)

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

# Tracer above cᵐⁱⁿ the unlimited outgoing fluxes remove from cell (i, j, k), per unit volume, over Δt
@inline function removed_tracer(i, j, k, grid, fluxes, Δt)
    outgoing = outgoing_flux_x(i, j, k, grid, fluxes) + outgoing_flux_y(i, j, k, grid, fluxes) + outgoing_flux_z(i, j, k, grid, fluxes)
    return Δt * V⁻¹ᶜᶜᶜ(i, j, k, grid) * outgoing
end

# Tracer above cᵐⁱⁿ the unlimited incoming fluxes bring into cell (i, j, k), per unit volume, over Δt
@inline function supplied_tracer(i, j, k, grid, fluxes, Δt)
    incoming = incoming_flux_x(i, j, k, grid, fluxes) + incoming_flux_y(i, j, k, grid, fluxes) + incoming_flux_z(i, j, k, grid, fluxes)
    return Δt * V⁻¹ᶜᶜᶜ(i, j, k, grid) * incoming
end

# The density of cell (i, j, k): a field, or the number `1` of `div_Uc`
@inline cell_density(i, j, k, ρ::Number) = ρ
@inline cell_density(i, j, k, ρ) = @inbounds ρ[i, j, k]

# The content of cell (i, j, k) above the lower bound and its room below the upper bound, in the units of ρ c
@inline function content_and_room(i, j, k, c, ρ, bounds)
    ρᶜ = cell_density(i, j, k, ρ)
    @inbounds cᵢ = c[i, j, k]
    content = ρᶜ * max(0, cᵢ - bounds.minimum_value)
    room = ρᶜ * max(0, bounds.maximum_value - cᵢ)
    return content, room
end

# min(1, available / transported), guarded against non-finite fluxes
@inline flux_fraction(transported, available) =
    ifelse(isfinite(transported) & (transported > available), available / transported, one(available))

# R⁻ = min(1, content / removed) and R⁺ = min(1, room / supplied)
@kernel function _compute_flux_fractions!(R⁻, R⁺, grid, bounds, fluxes, c, ρ, Δt)
    i, j, k = @index(Global, NTuple)
    content, room = content_and_room(i, j, k, c, ρ, bounds)
    @inbounds R⁻[i, j, k] = flux_fraction(removed_tracer(i, j, k, grid, fluxes, Δt), content)
    @inbounds R⁺[i, j, k] = flux_fraction(supplied_tracer(i, j, k, grid, fluxes, Δt), room)
end

"""
$(TYPEDSIGNATURES)

Refresh a flux-limited `scheme` (see [`FluxLimiter`](@ref)) for the coming forward Euler update
`c - Δt * div_Uc(c)` of a tracer: cache the bounded face fluxes of `c` above the lower bound and compute the
fractions of each cell's outgoing and incoming fluxes the update applies. `U` are the velocities the fluxes
transport with and `ρ` the density weighting the fluxes, which `div_Uc` takes as `1`.

A no-op for every other scheme.
"""
update_flux_limiter!(scheme, grid, c, U, Δt, ρ=1) = nothing

function update_flux_limiter!(scheme::FluxLimitedWENO, grid, c, U, Δt, ρ=1)
    bounds = scheme.bounds
    limiter = bounds.flux_limiter
    fluxes = limiter.fluxes
    R⁻ = limiter.outgoing_fraction
    R⁺ = limiter.incoming_fraction
    parameters = face_flux_kernel_parameters(grid)
    launch!(architecture(grid), grid, parameters, _compute_excess_face_fluxes!, fluxes, grid, scheme, ρ, U, c)
    launch!(architecture(grid), grid, :xyz, _compute_flux_fractions!, R⁻, R⁺, grid, bounds, fluxes, c, ρ, Δt)
    fill_halo_regions!(R⁻)
    fill_halo_regions!(R⁺)
    return nothing
end
