using Oceananigans.BoundaryConditions: fill_bottom_and_top_halo!
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

The fractions are computed by [`update_flux_limiter!`](@ref) before the fluxes are evaluated; the fields are
`nothing` until the scheme is materialized on a grid. The fractions are computed in the first horizontal halo
layer without communication, so the horizontal halo must be two cells wider than the WENO stencil reaches.

The guarantee holds for each forward Euler update over [`flux_limiter_timestep`](@ref) when
- every cell's outgoing Courant number, summed over its faces, is at most one. With
  `AdaptiveImplicitVerticalAdvection` this applies to the explicit part only, whose vertical Courant number is capped
  at `cfl`; a cell with outflow through both vertical faces needs `cfl ≤ (1 - λʰ) / 2`, with `λʰ` its horizontal
  outgoing Courant number. The implicit upwind remainder preserves the bounds at any Courant number;
- the transport is divergence-free, or `ρ` satisfies discrete continuity with the transport (on grids whose
  cells change thickness, pass the cell thickness or density as `ρ`);
- the time stepper is a convex combination of forward Euler steps (forward Euler, SSP Runge–Kutta, or a
  multirate scheme with non-negative weights), and `flux_limiter_timestep` returns its largest forward Euler step.
  `RungeKutta3TimeStepper` and `SplitRungeKuttaTimeStepper` are not of this kind, so with them only the first
  stage is guaranteed.
"""
struct FluxLimiter{R, A}
    outgoing_fraction :: R     # R⁻, the fraction of each cell's outgoing antidiffusive fluxes its lower bound affords
    incoming_fraction :: R     # R⁺, the fraction of each cell's incoming antidiffusive fluxes its upper bound affords
    antidiffusive_fluxes :: A  # A = Jᴴ - Jᵘ on the faces (x, y, z) of the interior cells and the first horizontal halo layer
end

FluxLimiter() = FluxLimiter(nothing, nothing, nothing)

Adapt.adapt_structure(to, limiter::FluxLimiter) =
    FluxLimiter(Adapt.adapt(to, limiter.outgoing_fraction),
                Adapt.adapt(to, limiter.incoming_fraction),
                Adapt.adapt(to, limiter.antidiffusive_fluxes))

Architectures.on_architecture(to, limiter::FluxLimiter) =
    FluxLimiter(on_architecture(to, limiter.outgoing_fraction),
                on_architecture(to, limiter.incoming_fraction),
                on_architecture(to, limiter.antidiffusive_fluxes))

const FluxLimitedBounds = BoundsPreservation{<:Any, <:Any, <:Any, <:FluxLimiter}
const FluxLimitedWENO = WENO{<:Any, <:Any, <:Any, <:Any, <:FluxLimitedBounds}

flux_limiter_summary(::FluxLimitedBounds) = ", flux_limiter=FluxLimiter()"

#####
##### Upwind, bound-transporting and antidiffusive face fluxes
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

# The antidiffusive flux A = Jᴴ - Jᵘ through a face, computed once per face and cached: the tendency then
# reconstructs nothing, and the fractions read the cache.
@kernel function _compute_antidiffusive_fluxes!(A, grid, advection, ρ, U, c)
    i, j, k = @index(Global, NTuple)
    @inbounds A.x[i, j, k] = bounded_face_flux_x(i, j, k, grid, advection, ρ, U.u, c) - upwind_face_flux_x(i, j, k, grid, advection, ρ, U.u, c)
    @inbounds A.y[i, j, k] = bounded_face_flux_y(i, j, k, grid, advection, ρ, U.v, c) - upwind_face_flux_y(i, j, k, grid, advection, ρ, U.v, c)
    @inbounds A.z[i, j, k] = bounded_face_flux_z(i, j, k, grid, advection, ρ, U.w, c) - upwind_face_flux_z(i, j, k, grid, advection, ρ, U.w, c)
end

# The faces of the interior cells and of the first horizontal halo layer: 0:N+2 horizontally, 1:N+1 vertically
function antidiffusive_flux_kernel_parameters(grid)
    Nx, Ny, Nz = size(grid)
    Tx, Ty, Tz = topology(grid)
    ii = ifelse(Tx == Flat, 1:Nx, 0:Nx+2)
    jj = ifelse(Ty == Flat, 1:Ny, 0:Ny+2)
    kk = ifelse(Tz == Flat, 1:Nz, 1:Nz+1)
    return KernelParameters(ii, jj, kk)
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
    Jᵘ = upwind_face_flux_x(i, j, k, grid, advection, ρ, u, c)
    A = @inbounds limiter.antidiffusive_fluxes.x[i, j, k]
    ϑ = face_factor(A, limiter.outgoing_fraction, limiter.incoming_fraction, i-1, j, k, i, j, k)
    return Jᵘ + ϑ * A
end

@inline function limited_face_flux_y(i, j, k, grid, advection, ρ, v, c)
    limiter = advection.bounds.flux_limiter
    Jᵘ = upwind_face_flux_y(i, j, k, grid, advection, ρ, v, c)
    A = @inbounds limiter.antidiffusive_fluxes.y[i, j, k]
    ϑ = face_factor(A, limiter.outgoing_fraction, limiter.incoming_fraction, i, j-1, k, i, j, k)
    return Jᵘ + ϑ * A
end

@inline function limited_face_flux_z(i, j, k, grid, advection, ρ, w, c)
    limiter = advection.bounds.flux_limiter
    Jᵘ = upwind_face_flux_z(i, j, k, grid, advection, ρ, w, c)
    A = @inbounds limiter.antidiffusive_fluxes.z[i, j, k]
    ϑ = face_factor(A, limiter.outgoing_fraction, limiter.incoming_fraction, i, j, k-1, i, j, k)
    return Jᵘ + ϑ * A
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

# The divergence of the upwind fluxes of `c` plus the antidiffusive fluxes, each scaled by the factor of its face
# from the fractions `update_flux_limiter!` computed. With every fraction equal to one this is the bounds-preserving divergence.
@inline function div_Uc(i, j, k, grid, advection::FluxLimitedWENO, U, c)
    div_x = limited_tracer_flux_divergence_x(i, j, k, grid, advection, 1, U.u, c)
    div_y = limited_tracer_flux_divergence_y(i, j, k, grid, advection, 1, U.v, c)
    div_z = limited_tracer_flux_divergence_z(i, j, k, grid, advection, 1, U.w, c)

    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (div_x + div_y + div_z)
end

#####
##### Fractions
#####

# The antidiffusive fluxes leaving and entering cell (i, j, k) through its two faces in one direction.
# Flat directions have no faces to sum over.
@inline antidiffusion_x(i, j, k, grid::XFlatGrid, A) = zero(grid), zero(grid)
@inline antidiffusion_y(i, j, k, grid::YFlatGrid, A) = zero(grid), zero(grid)
@inline antidiffusion_z(i, j, k, grid::ZFlatGrid, A) = zero(grid), zero(grid)

@inline function antidiffusion_x(i, j, k, grid, A)
    A⁻ = @inbounds A.x[i,   j, k]
    A⁺ = @inbounds A.x[i+1, j, k]
    return max(0, A⁺) - min(0, A⁻), max(0, A⁻) - min(0, A⁺)
end

@inline function antidiffusion_y(i, j, k, grid, A)
    A⁻ = @inbounds A.y[i, j, k]
    A⁺ = @inbounds A.y[i, j+1, k]
    return max(0, A⁺) - min(0, A⁻), max(0, A⁻) - min(0, A⁺)
end

@inline function antidiffusion_z(i, j, k, grid, A)
    A⁻ = @inbounds A.z[i, j, k]
    A⁺ = @inbounds A.z[i, j, k+1]
    return max(0, A⁺) - min(0, A⁻), max(0, A⁻) - min(0, A⁺)
end

# P⁻ and P⁺: the antidiffusive fluxes leaving and entering cell (i, j, k), per unit volume, over Δt
@inline function antidiffusion(i, j, k, grid, A, Δt)
    Oˣ, Iˣ = antidiffusion_x(i, j, k, grid, A)
    Oʸ, Iʸ = antidiffusion_y(i, j, k, grid, A)
    Oᶻ, Iᶻ = antidiffusion_z(i, j, k, grid, A)
    scale = Δt * V⁻¹ᶜᶜᶜ(i, j, k, grid)
    return scale * (Oˣ + Oʸ + Oᶻ), scale * (Iˣ + Iʸ + Iᶻ)
end

# The density of cell (i, j, k): a field, or the number `1` of `div_Uc`
@inline cell_density(i, j, k, ρ::Number) = ρ
@inline cell_density(i, j, k, ρ) = @inbounds ρ[i, j, k]

# The divergences ∇⋅Jᵘ of the upwind fluxes of c and ∇⋅(ρ u A) of the mass fluxes, per unit volume, from
# which the divergence of the fluxes transporting a uniform b follows as b ∇⋅(ρ u A)
@inline function upwind_and_mass_divergences(i, j, k, grid, advection, ρ, U, c)
    Dᵘ = (upwind_face_flux_x(i+1, j, k, grid, advection, ρ, U.u, c) - upwind_face_flux_x(i, j, k, grid, advection, ρ, U.u, c)) +
         (upwind_face_flux_y(i, j+1, k, grid, advection, ρ, U.v, c) - upwind_face_flux_y(i, j, k, grid, advection, ρ, U.v, c)) +
         (upwind_face_flux_z(i, j, k+1, grid, advection, ρ, U.w, c) - upwind_face_flux_z(i, j, k, grid, advection, ρ, U.w, c))

    Dᵐ = (bound_flux_x(i+1, j, k, grid, advection, ρ, U.u, 1) - bound_flux_x(i, j, k, grid, advection, ρ, U.u, 1)) +
         (bound_flux_y(i, j+1, k, grid, advection, ρ, U.v, 1) - bound_flux_y(i, j, k, grid, advection, ρ, U.v, 1)) +
         (bound_flux_z(i, j, k+1, grid, advection, ρ, U.w, 1) - bound_flux_z(i, j, k, grid, advection, ρ, U.w, 1))

    V⁻¹ = V⁻¹ᶜᶜᶜ(i, j, k, grid)
    return V⁻¹ * Dᵘ, V⁻¹ * Dᵐ
end

# Q⁻ and Q⁺: the room the upwind update leaves above cᵐⁱⁿ and below cᵐᵃˣ, in the units of ρ c
@inline function upwind_room(i, j, k, grid, advection, ρ, U, c, Δt)
    bounds = advection.bounds
    m = bounds.minimum_value
    M = bounds.maximum_value
    ρᶜ = cell_density(i, j, k, ρ)
    @inbounds cᵢ = c[i, j, k]
    Dᵘ, Dᵐ = upwind_and_mass_divergences(i, j, k, grid, advection, ρ, U, c)
    Q⁻ = ρᶜ * (cᵢ - m) - Δt * (Dᵘ - m * Dᵐ)
    Q⁺ = ρᶜ * (M - cᵢ) + Δt * (Dᵘ - M * Dᵐ)
    return max(0, Q⁻), max(0, Q⁺)
end

# min(1, available / transported), guarded against non-finite fluxes
@inline flux_fraction(transported, available) =
    ifelse(isfinite(transported) & (transported > available), available / transported, one(available))

# With the rescaled upwind face values cached by the θ kernel, the antidiffusive flux of a face costs a read and
# a few flops, so the fractions kernel computes A on the faces of its cell itself and stores it: each cell writes
# its lower faces, and the last cell of the range in each direction also its upper face. The stored A is the one the
# separate kernel would store (the same expression), and the tendency and any other user of `antidiffusive_fluxes`
# read it as before; the fused kernel saves one launch and the reads of A.
@inline antidiffusive_flux_x(i, j, k, grid, advection, ρ, U, c) = bounded_face_flux_x(i, j, k, grid, advection, ρ, U.u, c) - upwind_face_flux_x(i, j, k, grid, advection, ρ, U.u, c)
@inline antidiffusive_flux_y(i, j, k, grid, advection, ρ, U, c) = bounded_face_flux_y(i, j, k, grid, advection, ρ, U.v, c) - upwind_face_flux_y(i, j, k, grid, advection, ρ, U.v, c)
@inline antidiffusive_flux_z(i, j, k, grid, advection, ρ, U, c) = bounded_face_flux_z(i, j, k, grid, advection, ρ, U.w, c) - upwind_face_flux_z(i, j, k, grid, advection, ρ, U.w, c)

@inline local_antidiffusion_x(i, j, k, grid::XFlatGrid, args...) = zero(grid), zero(grid), zero(grid), zero(grid)
@inline local_antidiffusion_y(i, j, k, grid::YFlatGrid, args...) = zero(grid), zero(grid), zero(grid), zero(grid)
@inline local_antidiffusion_z(i, j, k, grid::ZFlatGrid, args...) = zero(grid), zero(grid), zero(grid), zero(grid)

# The antidiffusive fluxes through the lower and upper faces of cell (i, j, k), and the parts that leave and enter it
@inline function local_antidiffusion_x(i, j, k, grid, advection, ρ, U, c)
    A⁻ = antidiffusive_flux_x(i,   j, k, grid, advection, ρ, U, c)
    A⁺ = antidiffusive_flux_x(i+1, j, k, grid, advection, ρ, U, c)
    return A⁻, A⁺, max(0, A⁺) - min(0, A⁻), max(0, A⁻) - min(0, A⁺)
end

@inline function local_antidiffusion_y(i, j, k, grid, advection, ρ, U, c)
    A⁻ = antidiffusive_flux_y(i, j,   k, grid, advection, ρ, U, c)
    A⁺ = antidiffusive_flux_y(i, j+1, k, grid, advection, ρ, U, c)
    return A⁻, A⁺, max(0, A⁺) - min(0, A⁻), max(0, A⁻) - min(0, A⁺)
end

@inline function local_antidiffusion_z(i, j, k, grid, advection, ρ, U, c)
    A⁻ = antidiffusive_flux_z(i, j, k,   grid, advection, ρ, U, c)
    A⁺ = antidiffusive_flux_z(i, j, k+1, grid, advection, ρ, U, c)
    return A⁻, A⁺, max(0, A⁺) - min(0, A⁻), max(0, A⁻) - min(0, A⁺)
end

@inline store_antidiffusion_x!(A, i, j, k, grid::XFlatGrid, A⁻, A⁺, last) = nothing
@inline store_antidiffusion_y!(A, i, j, k, grid::YFlatGrid, A⁻, A⁺, last) = nothing
@inline store_antidiffusion_z!(A, i, j, k, grid::ZFlatGrid, A⁻, A⁺, last) = nothing

@inline function store_antidiffusion_x!(A, i, j, k, grid, A⁻, A⁺, last)
    @inbounds A[i, j, k] = A⁻
    @inbounds last && (A[i+1, j, k] = A⁺)
    return nothing
end

@inline function store_antidiffusion_y!(A, i, j, k, grid, A⁻, A⁺, last)
    @inbounds A[i, j, k] = A⁻
    @inbounds last && (A[i, j+1, k] = A⁺)
    return nothing
end

@inline function store_antidiffusion_z!(A, i, j, k, grid, A⁻, A⁺, last)
    @inbounds A[i, j, k] = A⁻
    @inbounds last && (A[i, j, k+1] = A⁺)
    return nothing
end

@kernel function _compute_antidiffusive_fluxes_and_fractions!(R⁻, R⁺, A, grid, advection, U, c, ρ, Δt, iᴸ, jᴸ, kᴸ)
    i, j, k = @index(Global, NTuple)
    Q⁻, Q⁺ = upwind_room(i, j, k, grid, advection, ρ, U, c, Δt)
    A⁻ˣ, A⁺ˣ, Oˣ, Iˣ = local_antidiffusion_x(i, j, k, grid, advection, ρ, U, c)
    A⁻ʸ, A⁺ʸ, Oʸ, Iʸ = local_antidiffusion_y(i, j, k, grid, advection, ρ, U, c)
    A⁻ᶻ, A⁺ᶻ, Oᶻ, Iᶻ = local_antidiffusion_z(i, j, k, grid, advection, ρ, U, c)
    store_antidiffusion_x!(A.x, i, j, k, grid, A⁻ˣ, A⁺ˣ, i == iᴸ)
    store_antidiffusion_y!(A.y, i, j, k, grid, A⁻ʸ, A⁺ʸ, j == jᴸ)
    store_antidiffusion_z!(A.z, i, j, k, grid, A⁻ᶻ, A⁺ᶻ, k == kᴸ)
    scale = Δt * V⁻¹ᶜᶜᶜ(i, j, k, grid)
    P⁻ = scale * (Oˣ + Oʸ + Oᶻ)
    P⁺ = scale * (Iˣ + Iʸ + Iᶻ)
    @inbounds R⁻[i, j, k] = flux_fraction(P⁻, Q⁻)
    @inbounds R⁺[i, j, k] = flux_fraction(P⁺, Q⁺)
end

# R⁻ = min(1, Q⁻ / P⁻) and R⁺ = min(1, Q⁺ / P⁺)
@kernel function _compute_flux_fractions!(R⁻, R⁺, grid, advection, A, U, c, ρ, Δt)
    i, j, k = @index(Global, NTuple)
    Q⁻, Q⁺ = upwind_room(i, j, k, grid, advection, ρ, U, c, Δt)
    P⁻, P⁺ = antidiffusion(i, j, k, grid, A, Δt)
    @inbounds R⁻[i, j, k] = flux_fraction(P⁻, Q⁻)
    @inbounds R⁺[i, j, k] = flux_fraction(P⁺, Q⁺)
end

# The fractions are computed in the first horizontal halo layer too, from the halos of `c`, `U` and θ (which
# the rescaling then computes in two halo layers), so that the faces of every interior cell find the fractions
# of both their cells without a halo fill: only the bottom and top halos, which are never communicated, are filled.
flux_fraction_kernel_parameters(grid) = KernelParameters(flux_fraction_ranges(grid)...)

function flux_fraction_ranges(grid)
    Nx, Ny, Nz = size(grid)
    Tx, Ty, Tz = topology(grid)
    return (ifelse(Tx == Flat, 1:Nx, 0:Nx+1), ifelse(Ty == Flat, 1:Ny, 0:Ny+1), 1:Nz)
end

"""
$(TYPEDSIGNATURES)

Refresh a flux-limited `scheme` (see [`FluxLimiter`](@ref)) for the coming forward Euler update
`c - Δt * div_Uc(c)` of a tracer: compute the fractions of each cell's outgoing and incoming antidiffusive
fluxes (the bounds-preserving WENO fluxes minus the first-order upwind fluxes, cached on the faces) the update
applies. The
rescaling factor θ must be up to date. No horizontal halo is filled or communicated. `U` are the velocities
the fluxes transport with and `ρ` the density weighting the fluxes, which `div_Uc` takes as `1`.

A no-op for every other scheme.
"""
update_flux_limiter!(scheme, grid, c, U, Δt, ρ=1) = nothing

function update_flux_limiter!(scheme::FluxLimitedWENO, grid, c, U, Δt, ρ=1)
    limiter = scheme.bounds.flux_limiter
    R⁻ = limiter.outgoing_fraction
    R⁺ = limiter.incoming_fraction
    A = limiter.antidiffusive_fluxes
    compute_antidiffusive_fluxes_and_fractions!(R⁻, R⁺, A, grid, scheme, scheme.bounds.upwind_face_values, U, c, ρ, Δt)
    loc = (Center(), Center(), Center())
    fill_bottom_and_top_halo!(R⁻.data, R⁻.boundary_conditions, R⁻.indices, loc, grid)
    fill_bottom_and_top_halo!(R⁺.data, R⁺.boundary_conditions, R⁺.indices, loc, grid)
    return nothing
end

# Without cached face values (`rescaling=false`) the antidiffusive fluxes are reconstructed once per face in their own
# kernel, which the fractions then read; with them, one fused kernel computes both.
function compute_antidiffusive_fluxes_and_fractions!(R⁻, R⁺, A, grid, scheme, ::Nothing, U, c, ρ, Δt)
    launch!(architecture(grid), grid, antidiffusive_flux_kernel_parameters(grid), _compute_antidiffusive_fluxes!, A, grid, scheme, ρ, U, c)
    launch!(architecture(grid), grid, flux_fraction_kernel_parameters(grid), _compute_flux_fractions!, R⁻, R⁺, grid, scheme, A, U, c, ρ, Δt)
    return nothing
end

function compute_antidiffusive_fluxes_and_fractions!(R⁻, R⁺, A, grid, scheme, face_values, U, c, ρ, Δt)
    parameters = flux_fraction_kernel_parameters(grid)
    iᴸ, jᴸ, kᴸ = map(last, flux_fraction_ranges(grid))
    launch!(architecture(grid), grid, parameters, _compute_antidiffusive_fluxes_and_fractions!, R⁻, R⁺, A, grid, scheme, U, c, ρ, Δt, iᴸ, jᴸ, kᴸ)
    return nothing
end

# The fractions in the first horizontal halo layer need θ in the second, whose reconstructions reach one cell further
@inline Grids.required_halo_size_x(::FluxLimitedWENO{N}) where N = N + 2
@inline Grids.required_halo_size_y(::FluxLimitedWENO{N}) where N = N + 2

rescaling_kernel_parameters(grid, scheme::FluxLimitedWENO) = doubly_extended_interior_kernel_parameters(grid)

function doubly_extended_interior_kernel_parameters(grid)
    Nx, Ny, Nz = size(grid)
    Tx, Ty, Tz = topology(grid)
    ii = ifelse(Tx == Flat, 1:Nx, -1:Nx+2)
    jj = ifelse(Ty == Flat, 1:Ny, -1:Ny+2)
    return KernelParameters(ii, jj, 1:Nz)
end
