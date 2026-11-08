using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: XFaceField, YFaceField, ZFaceField
using Oceananigans.Grids: topology
using Oceananigans.Utils: KernelParameters

#####
##### Flux limiting for bounds-preserving WENO
#####
#
# The rescaling of Zhang and Shu (2010) keeps a forward Euler update within `bounds` only while the
# Courant number stays below ω̂₁; a larger one can remove more from a cell than it holds. The flux limiter scales each cell's outgoing fluxes by a fraction `r ∈ [0, 1]` of them the coming
# update can afford: a conservative, zeroth-order flux-corrected transport limiter that keeps the
# tracer non-negative for any time step, at first-order accuracy where it acts.
#
# Safety. With Oᵢ = Δt V⁻¹ Σ (outgoing fluxes of cell i) and Iᵢ(r) = Δt V⁻¹ Σ (incoming fluxes, each
# scaled by the fraction of the cell it leaves), the forward Euler update is c'ᵢ = cᵢ - rᵢ Oᵢ + Iᵢ(r), so
#
#     c'ᵢ ≥ 0  ⟺  rᵢ ≤ G(r)ᵢ := min(1, (cᵢ + Iᵢ(r)) / Oᵢ).
#
# G is nondecreasing in r, and r = 0 satisfies r ≤ G(r). If r ≤ G(r) and some components are replaced
# by G̃(r)ᵢ, where G̃ ≤ G credits a subset of the incoming fluxes (at their current fractions) and the
# subset only grows between successive updates of a cell, then the fractions never decrease, so
# G̃(r)ᵢ ≤ G(r)ᵢ ≤ G(r')ᵢ for the replaced components and rᵢ ≤ G(r)ᵢ ≤ G(r')ᵢ for the others: r' ≤ G(r')
# again. Every such process therefore ends with r ≤ G(r), that is, with a non-negative tracer. Both
# limiters below are instances: `FluxLimiter()` credits nothing, `FluxLimiter(credit=ColumnSweep())`
# credits the vertical neighbors in Gauss–Seidel order along each column.

"""
    ColumnSweep()

The incoming-flux `credit` of a [`FluxLimiter`](@ref) that sweeps each column downward then upward,
crediting a cell's incoming vertical fluxes at the fractions of its vertical neighbors as they stand
(Gauss–Seidel), and nothing horizontally. Two sweeps give the exact fractions of any vertical Courant
number, which the single pass of `FluxLimiter()` caps at one.
"""
struct ColumnSweep end

"""
    FluxLimiter(; credit=nothing)

The `flux_limiter` of a [`BoundsPreservation`](@ref), which keeps the tracer non-negative by limiting each
cell's outgoing fluxes to the fraction ``r = min(1, available / removed)`` of them the coming forward Euler
update can afford. `removed` is the tracer the unlimited outgoing fluxes take out of the cell, `available`
the cell's content plus the incoming fluxes `credit` admits: nothing by default (one pass
over the grid), or [`ColumnSweep`](@ref) for the vertical ones, which a vertical Courant number above one
needs.

Each face flux is scaled by the fraction of the cell it leaves, so the limited fluxes stay conservative.
The fractions and the face fluxes they are built from are computed by [`update_flux_limiter!`](@ref) before
the fluxes are evaluated; `fraction` and `fluxes` are `nothing` until the scheme is materialized on a grid.
"""
struct FluxLimiter{C, R, F}
    credit :: C    # how incoming fluxes are credited: `nothing` or `ColumnSweep()`
    fraction :: R  # r, the fraction of each cell's outgoing fluxes the coming update applies
    fluxes :: F    # the bounded face fluxes (x, y, z) the fractions are built from and the divergence reads
end

FluxLimiter(; credit=nothing) = FluxLimiter(credit, nothing, nothing)

Adapt.adapt_structure(to, limiter::FluxLimiter) =
    FluxLimiter(Adapt.adapt(to, limiter.credit), Adapt.adapt(to, limiter.fraction), Adapt.adapt(to, limiter.fluxes))

Architectures.on_architecture(to, limiter::FluxLimiter) =
    FluxLimiter(on_architecture(to, limiter.credit), on_architecture(to, limiter.fraction), on_architecture(to, limiter.fluxes))

const FluxLimitedBounds = BoundsPreservation{<:Any, <:Any, <:FluxLimiter}
const FluxLimitedWENO = WENO{<:Any, <:Any, <:Any, <:Any, <:FluxLimitedBounds}

Base.summary(::FluxLimiter{Nothing}) = "FluxLimiter()"
Base.summary(limiter::FluxLimiter) = string("FluxLimiter(credit=", nameof(typeof(limiter.credit)), "())")

Base.show(io::IO, bounds::FluxLimitedBounds) =
    print(io, "(", bounds.minimum_value, ", ", bounds.maximum_value, "), flux_limiter=", summary(bounds.flux_limiter))

#####
##### Cached bounded face fluxes
#####

# The faces the interior cells' divergences read: one more than the cells in every non-Flat direction.
function face_flux_kernel_parameters(grid)
    N = size(grid)
    T = topology(grid)
    faces = ntuple(d -> T[d] === Flat ? N[d] : N[d] + 1, 3)
    return KernelParameters(faces, (0, 0, 0))
end

@kernel function _compute_bounded_face_fluxes!(fluxes, grid, advection, ρ, U, c)
    i, j, k = @index(Global, NTuple)
    @inbounds fluxes.x[i, j, k] = bounded_face_flux_x(i, j, k, grid, advection, ρ, U.u, c)
    @inbounds fluxes.y[i, j, k] = bounded_face_flux_y(i, j, k, grid, advection, ρ, U.v, c)
    @inbounds fluxes.z[i, j, k] = bounded_face_flux_z(i, j, k, grid, advection, ρ, U.w, c)
end

#####
##### Limited face fluxes and their divergence
#####

# Each cached face flux is scaled by the fraction of its upwind cell, which the flux's sign identifies.
@inline function limited_face_flux_x(i, j, k, grid, advection)
    F = advection.bounds.flux_limiter.fluxes.x
    r = advection.bounds.flux_limiter.fraction
    return @inbounds F[i, j, k] * ifelse(F[i, j, k] ≥ 0, r[i-1, j, k], r[i, j, k])
end

@inline function limited_face_flux_y(i, j, k, grid, advection)
    F = advection.bounds.flux_limiter.fluxes.y
    r = advection.bounds.flux_limiter.fraction
    return @inbounds F[i, j, k] * ifelse(F[i, j, k] ≥ 0, r[i, j-1, k], r[i, j, k])
end

@inline function limited_face_flux_z(i, j, k, grid, advection)
    F = advection.bounds.flux_limiter.fluxes.z
    r = advection.bounds.flux_limiter.fraction
    return @inbounds F[i, j, k] * ifelse(F[i, j, k] ≥ 0, r[i, j, k-1], r[i, j, k])
end

# Support for Flat directions, which have no neighbors to index
@inline limited_tracer_flux_divergence_x(i, j, k, grid::XFlatGrid, advection) = zero(grid)
@inline limited_tracer_flux_divergence_y(i, j, k, grid::YFlatGrid, advection) = zero(grid)
@inline limited_tracer_flux_divergence_z(i, j, k, grid::ZFlatGrid, advection) = zero(grid)

@inline limited_tracer_flux_divergence_x(i, j, k, grid, advection) =
    limited_face_flux_x(i+1, j, k, grid, advection) - limited_face_flux_x(i, j, k, grid, advection)

@inline limited_tracer_flux_divergence_y(i, j, k, grid, advection) =
    limited_face_flux_y(i, j+1, k, grid, advection) - limited_face_flux_y(i, j, k, grid, advection)

@inline limited_tracer_flux_divergence_z(i, j, k, grid, advection) =
    limited_face_flux_z(i, j, k+1, grid, advection) - limited_face_flux_z(i, j, k, grid, advection)

@inline div_Uc(i, j, k, grid, advection::FluxLimitedWENO, U, ::ZeroField) = zero(grid)
@inline div_Uc(i, j, k, grid, advection::FluxLimitedWENO, ::ZeroU, c) = zero(grid)
@inline div_Uc(i, j, k, grid, advection::FluxLimitedWENO, ::ZeroU, ::ZeroField) = zero(grid)

# The divergence of the cached fluxes `update_flux_limiter!` built from `U` and `c`, each scaled by the
# fraction of its upwind cell. With every fraction equal to one this is the bounds-preserving divergence.
@inline function div_Uc(i, j, k, grid, advection::FluxLimitedWENO, U, c)
    div_x = limited_tracer_flux_divergence_x(i, j, k, grid, advection)
    div_y = limited_tracer_flux_divergence_y(i, j, k, grid, advection)
    div_z = limited_tracer_flux_divergence_z(i, j, k, grid, advection)

    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (div_x + div_y + div_z)
end

#####
##### Fractions
#####

# Flat directions have no faces to sum over and no neighbors to credit
@inline outgoing_flux_x(i, j, k, grid::XFlatGrid, fluxes) = zero(grid)
@inline outgoing_flux_y(i, j, k, grid::YFlatGrid, fluxes) = zero(grid)
@inline outgoing_flux_z(i, j, k, grid::ZFlatGrid, fluxes) = zero(grid)

@inline outgoing_flux_x(i, j, k, grid, fluxes) = @inbounds max(0, fluxes.x[i+1, j, k]) - min(0, fluxes.x[i, j, k])
@inline outgoing_flux_y(i, j, k, grid, fluxes) = @inbounds max(0, fluxes.y[i, j+1, k]) - min(0, fluxes.y[i, j, k])
@inline outgoing_flux_z(i, j, k, grid, fluxes) = @inbounds max(0, fluxes.z[i, j, k+1]) - min(0, fluxes.z[i, j, k])

# Tracer the unlimited outgoing fluxes remove from cell (i, j, k), per unit volume, over Δt.
@inline function removed_tracer(i, j, k, grid, fluxes, Δt)
    outgoing = outgoing_flux_x(i, j, k, grid, fluxes) +
               outgoing_flux_y(i, j, k, grid, fluxes) +
               outgoing_flux_z(i, j, k, grid, fluxes)
    return Δt * V⁻¹ᶜᶜᶜ(i, j, k, grid) * outgoing
end

# Tracer the incoming vertical fluxes bring into cell (i, j, k), per unit volume, over Δt, each scaled
# by the fraction `r⁻` or `r⁺` of the cell below or above it leaves.
@inline incoming_vertical_tracer(i, j, k, grid::ZFlatGrid, fluxes, Δt, r⁻, r⁺) = zero(grid)

@inline function incoming_vertical_tracer(i, j, k, grid, fluxes, Δt, r⁻, r⁺)
    incoming = @inbounds max(0, fluxes.z[i, j, k]) * r⁻ - min(0, fluxes.z[i, j, k+1]) * r⁺
    return Δt * V⁻¹ᶜᶜᶜ(i, j, k, grid) * incoming
end

# r = min(1, available / removed), guarded against non-finite fluxes
@inline function flux_fraction(removed, content, credited)
    available = content + ifelse(isfinite(credited), credited, zero(credited))
    return ifelse(isfinite(removed) & (removed > available), available / removed, one(available))
end

# One pass with no incoming credit: r = min(1, c / removed)
@kernel function _compute_flux_fraction!(r, grid, fluxes, c, Δt)
    i, j, k = @index(Global, NTuple)
    removed = removed_tracer(i, j, k, grid, fluxes, Δt)
    content = @inbounds max(0, c[i, j, k])
    @inbounds r[i, j, k] = flux_fraction(removed, content, zero(removed))
end

# A downward sweep crediting the cell above, which it has visited, then an upward sweep crediting both
# vertical neighbors. Each cell's credited set grows between its two updates and a neighbor beyond the
# column's ends (not yet visited in the downward sweep, the stale halo otherwise) is credited at zero,
# which the safety argument above allows.
@kernel function _sweep_column_flux_fractions!(r, grid, fluxes, c, Δt)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    for k in Nz:-1:1
        removed = removed_tracer(i, j, k, grid, fluxes, Δt)
        content = @inbounds max(0, c[i, j, k])
        r⁺ = @inbounds ifelse(k < Nz, r[i, j, k+1], zero(content))
        credited = incoming_vertical_tracer(i, j, k, grid, fluxes, Δt, zero(content), r⁺)
        @inbounds r[i, j, k] = flux_fraction(removed, content, credited)
    end

    for k in 1:Nz
        removed = removed_tracer(i, j, k, grid, fluxes, Δt)
        content = @inbounds max(0, c[i, j, k])
        r⁻ = @inbounds ifelse(k > 1,  r[i, j, k-1], zero(content))
        r⁺ = @inbounds ifelse(k < Nz, r[i, j, k+1], zero(content))
        credited = incoming_vertical_tracer(i, j, k, grid, fluxes, Δt, r⁻, r⁺)
        @inbounds r[i, j, k] = flux_fraction(removed, content, credited)
    end
end

compute_flux_fractions!(::Nothing, r, grid, fluxes, c, Δt) =
    launch!(architecture(grid), grid, :xyz, _compute_flux_fraction!, r, grid, fluxes, c, Δt)

compute_flux_fractions!(::ColumnSweep, r, grid, fluxes, c, Δt) =
    launch!(architecture(grid), grid, :xy, _sweep_column_flux_fractions!, r, grid, fluxes, c, Δt)

"""
$(TYPEDSIGNATURES)

Refresh a flux-limited `scheme` (see [`FluxLimiter`](@ref)) for the coming forward Euler update
`c - Δt * div_Uc(c)` of a tracer: cache the bounded face fluxes of `c` and compute the fraction of each
cell's outgoing fluxes the update applies. `U` are the velocities the fluxes transport with and `ρ` the
density weighting the fluxes, which `div_Uc` takes as `1`.

A no-op for every other scheme.
"""
update_flux_limiter!(scheme, grid, c, U, Δt, ρ=1) = nothing

function update_flux_limiter!(scheme::FluxLimitedWENO, grid, c, U, Δt, ρ=1)
    limiter = scheme.bounds.flux_limiter
    fluxes = limiter.fluxes
    r = limiter.fraction
    parameters = face_flux_kernel_parameters(grid)
    launch!(architecture(grid), grid, parameters, _compute_bounded_face_fluxes!, fluxes, grid, scheme, ρ, U, c)
    compute_flux_fractions!(limiter.credit, r, grid, fluxes, c, Δt)
    fill_halo_regions!(r)
    return nothing
end

#####
##### Model interface
#####

"""
$(TYPEDSIGNATURES)

The velocities that advect the tracer `name` of `model`: its `velocities` unless the model extends this,
for instance with drift velocities or advective forcing.
"""
tracer_transport_velocities(model, ::Val{name}) where name = model.velocities

update_flux_limiter!(scheme, model, val_name) = nothing

# `adaptive_advection_timestep` is the Δt of the coming stage, which the limiter takes as a forward Euler update.
function update_flux_limiter!(scheme::FluxLimitedWENO, model, val_name::Val{name}) where name
    c = model.tracers[name]
    U = tracer_transport_velocities(model, val_name)
    Δt = adaptive_advection_timestep(model.timestepper, model.clock)
    return update_flux_limiter!(scheme, model.grid, c, U, Δt)
end
