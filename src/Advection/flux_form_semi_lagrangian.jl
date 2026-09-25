using Oceananigans.Grids: SerialFoldedTopology, inactive_cell
using Oceananigans.Operators: Vᶜᶜᶜ, σⁿ, flux_div_xyᶜᶜᶜ

#####
##### Limiters for the piecewise-parabolic reconstruction
#####

"""
    MonotonePPMLimiter()

Colella & Woodward (1984) limiter for the piecewise-parabolic reconstruction used by
[`FluxFormSemiLagrangian`](@ref): edge values are bounded by their neighbouring cell means and
the parabola in each cell is flattened or steepened so that it creates no new extrema.
"""
struct MonotonePPMLimiter end

Base.summary(::MonotonePPMLimiter) = "MonotonePPMLimiter"

limiter_from_keyword(::Nothing) = nothing
limiter_from_keyword(limiter::MonotonePPMLimiter) = limiter

function limiter_from_keyword(limiter::Symbol)
    limiter === :monotone && return MonotonePPMLimiter()
    throw(ArgumentError("limiter = :$limiter is not supported. Use `limiter = :monotone` or `limiter = nothing`."))
end

limiter_from_keyword(limiter) = throw(ArgumentError("limiter = $limiter is not supported. Use `limiter = :monotone` or `limiter = nothing`."))

#####
##### The scheme
#####

struct FluxFormSemiLagrangian{B, FT, TD, Cmax, L, V, W} <: AbstractAdvectionScheme{B, FT, TD}
    limiter :: L
    vertical_scheme :: V
    workspace :: W

    function FluxFormSemiLagrangian{B, FT, TD, Cmax}(limiter::L, vertical_scheme::V, workspace::W) where {B, FT, TD, Cmax, L, V, W}
        return new{B, FT, TD, Cmax, L, V, W}(limiter, vertical_scheme, workspace)
    end
end

const FFSL = FluxFormSemiLagrangian

"""
    FluxFormSemiLagrangian([FT = Oceananigans.defaults.FloatType];
                           maximum_courant_number = 3,
                           limiter = :monotone,
                           vertical_scheme = WENO(FT; order=5))

Return a flux-form semi-Lagrangian (FFSL) tracer advection scheme that remains stable and
conservative at horizontal Courant numbers larger than one.

Horizontal advection uses the dimensionally-split scheme of Lin & Rood (1996) and Putman & Lin (2007):
one-dimensional flux-form operators in ``x`` and ``y`` are combined with advective-form inner
operators, so that a uniform tracer stays uniform. The flux through a face is the tracer mass in
the region swept through the face during one time step. The swept region is found in index space
from the volume flux `Δt * u * Ax` and the actual cell volumes: it consists of `⌊C⌋` whole cells
plus a fraction of the next cell, integrated with a piecewise-parabolic (PPM) reconstruction.
This works on any logically rectangular grid (`RectilinearGrid`, `LatitudeLongitudeGrid`,
`TripolarGrid`) with or without an `ImmersedBoundaryGrid`. The swept region stops at the
first inactive (dry or boundary) cell; the flux is then the volume flux times the mean tracer
value over the part of the swept region that is available.

The horizontal step is operator-split and forward in time: it is taken once per time step (not
once per Runge-Kutta stage), with the transport velocities of the final stage, which are the ones
that advance the free surface. Vertical advection stays in the tendency and uses `vertical_scheme`
(which may be an `AdaptiveImplicitVerticalAdvection` scheme).

Only `HydrostaticFreeSurfaceModel` with a `SplitRungeKuttaTimeStepper` supports this scheme.
The grid halo must be at least `maximum_courant_number + 3` in every horizontal direction that
is not `Flat`.

Keyword arguments
=================

- `maximum_courant_number`: largest horizontal Courant number (swept volume divided by cell volume)
  that is resolved. Swept regions longer than `maximum_courant_number + 1` cells are truncated.
  Must be a positive integer. Default: 3.
- `limiter`: `:monotone` for the Colella & Woodward (1984) monotone PPM limiter, or `nothing` for
  unlimited PPM. Default: `:monotone`.
- `vertical_scheme`: advection scheme for the vertical direction. Default: `WENO(order=5)`.

Limitations
===========

Only the horizontal components of the model's transport velocities are used by the horizontal step:
horizontal biogeochemical drift velocities, closure-induced velocities (for example Gent-McWilliams
bolus velocities) and horizontal advective forcing are not supported for FFSL tracers.

Example
=======

```jldoctest
julia> using Oceananigans

julia> FluxFormSemiLagrangian()
FluxFormSemiLagrangian with maximum Courant number 3
├── limiter: MonotonePPMLimiter
└── vertical_scheme: WENO{3, Float64, Nothing}(order=5)
```

References
==========

- Lin, S.-J. and Rood, R. B. (1996). Multidimensional flux-form semi-Lagrangian transport schemes.
  Monthly Weather Review, 124, 2046–2070.
- Putman, W. M. and Lin, S.-J. (2007). Finite-volume transport on various cubed-sphere grids.
  Journal of Computational Physics, 227, 55–78.
- Colella, P. and Woodward, P. R. (1984). The piecewise parabolic method (PPM) for gas-dynamical
  simulations. Journal of Computational Physics, 54, 174–201.
"""
function FluxFormSemiLagrangian(FT::DataType = Oceananigans.defaults.FloatType;
                                maximum_courant_number = 3,
                                limiter = :monotone,
                                vertical_scheme = WENO(FT; order=5))

    if !(maximum_courant_number isa Integer) || maximum_courant_number < 1
        throw(ArgumentError("maximum_courant_number must be a positive integer, got $maximum_courant_number"))
    end

    Cmax = Int(maximum_courant_number)
    limiter = limiter_from_keyword(limiter)

    return FluxFormSemiLagrangian{FT}(Cmax, limiter, vertical_scheme, nothing)
end

function FluxFormSemiLagrangian{FT}(Cmax, limiter, vertical_scheme, workspace) where FT
    B  = max(Cmax + 3, required_halo_size_z(vertical_scheme))
    TD = typeof(ffsl_time_discretization(vertical_scheme))
    return FluxFormSemiLagrangian{B, FT, TD, Cmax}(limiter, vertical_scheme, workspace)
end

ffsl_time_discretization(vertical_scheme) = TimeSteppers.time_discretization(vertical_scheme)
ffsl_time_discretization(::Nothing) = ExplicitTimeDiscretization()

@inline maximum_courant_number(::FFSL{B, FT, TD, Cmax}) where {B, FT, TD, Cmax} = Cmax

with_workspace(scheme::FFSL{B, FT, TD, Cmax}, workspace) where {B, FT, TD, Cmax} =
    FluxFormSemiLagrangian{B, FT, TD, Cmax}(scheme.limiter, scheme.vertical_scheme, workspace)

with_vertical_scheme(scheme::FFSL{B, FT, TD, Cmax}, vertical_scheme) where {B, FT, TD, Cmax} =
    FluxFormSemiLagrangian{FT}(Cmax, scheme.limiter, vertical_scheme, scheme.workspace)

TimeSteppers.time_discretization(scheme::FFSL) = ffsl_time_discretization(scheme.vertical_scheme)

@inline vertical_scheme(scheme::FFSL) = scheme.vertical_scheme

@inline Grids.required_halo_size_x(scheme::FFSL) = maximum_courant_number(scheme) + 3
@inline Grids.required_halo_size_y(scheme::FFSL) = maximum_courant_number(scheme) + 3
@inline Grids.required_halo_size_z(scheme::FFSL) = required_halo_size_z(scheme.vertical_scheme)

Base.summary(scheme::FFSL) = string("FluxFormSemiLagrangian(maximum_courant_number=", maximum_courant_number(scheme),
                                    ", limiter=", summary(scheme.limiter),
                                    ", vertical_scheme=", summary(scheme.vertical_scheme), ")")

Base.show(io::IO, scheme::FFSL) =
    print(io, "FluxFormSemiLagrangian with maximum Courant number ", maximum_courant_number(scheme), '\n',
              "├── limiter: ", summary(scheme.limiter), '\n',
              "└── vertical_scheme: ", summary(scheme.vertical_scheme))

# The workspace is only used by the horizontal step, which receives it explicitly,
# so kernels computing tendencies only see the vertical scheme.
Adapt.adapt_structure(to, scheme::FFSL{B, FT, TD, Cmax}) where {B, FT, TD, Cmax} =
    FluxFormSemiLagrangian{B, FT, TD, Cmax}(scheme.limiter, Adapt.adapt(to, scheme.vertical_scheme), nothing)

Architectures.on_architecture(arch, scheme::FFSL{B, FT, TD, Cmax}) where {B, FT, TD, Cmax} =
    FluxFormSemiLagrangian{B, FT, TD, Cmax}(scheme.limiter,
                                            on_architecture(arch, scheme.vertical_scheme),
                                            on_architecture(arch, scheme.workspace))

materialize_advection(scheme::FFSL, grid) = with_vertical_scheme(scheme, materialize_advection(scheme.vertical_scheme, grid))

function adapt_advection_order(scheme::FFSL, grid::AbstractGrid)
    tz = topology(grid, 3)()
    new_vertical_scheme = adapt_advection_order(tz, scheme.vertical_scheme, size(grid, 3), grid)
    return with_vertical_scheme(scheme, new_vertical_scheme)
end

update_advection!(scheme::FFSL, model, tracer) = update_advection!(scheme.vertical_scheme, model, tracer)

#####
##### Tendency: vertical advection only
#####

# The horizontal step is taken separately. The tendency holds the vertical flux divergence
# and `c ∇ₕ⋅(Aₕ uₕ)`, which cancels the horizontal part of `∂z(A w)` so that the Runge-Kutta
# predictor stages keep a uniform tracer uniform. The horizontal step removes this term again
# on the final stage, see `flux_form_semi_lagrangian_step!`.
@inline function div_Uc(i, j, k, grid, advection::FFSL, U, c)
    scheme = advection.vertical_scheme
    vertical_flux_divergence = δzᵃᵃᶜ(i, j, k, grid, _advective_tracer_flux_z, scheme, U.w, c)
    horizontal_compensation = @inbounds c[i, j, k] * flux_div_xyᶜᶜᶜ(i, j, k, grid, U.u, U.v)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (vertical_flux_divergence + horizontal_compensation)
end

@inline div_Uc(i, j, k, grid, ::FFSL, ::ZeroU, c) = zero(grid)
@inline div_Uc(i, j, k, grid, ::FFSL, U, ::ZeroField) = zero(grid)
@inline div_Uc(i, j, k, grid, ::FFSL, ::ZeroU, ::ZeroField) = zero(grid)

#####
##### Horizontal step: geometry
#####

struct XSweep end
struct YSweep end

@inline offset_indices(::XSweep, i, j, m) = (i + m, j)
@inline offset_indices(::YSweep, i, j, m) = (i, j + m)

# The tripolar grid's southern halo lies outside the domain although its topology is not `Bounded`
@inline ffsl_inactive_cell(i, j, k, grid) = inactive_cell(i, j, k, grid)
@inline ffsl_inactive_cell(i, j, k, grid::AbstractGrid{<:Any, <:Any, <:SerialFoldedTopology}) = inactive_cell(i, j, k, grid) | (j < 1)

@inline static_volume(i, j, k, grid) = Vᶜᶜᶜ(i, j, k, grid) / σⁿ(i, j, k, grid, Center(), Center(), Center())

@inline initial_stretching(i, j, σ⁰) = @inbounds σ⁰[i, j, 1]
@inline initial_stretching(i, j, ::Nothing) = 1

# Cell volume at the beginning of the time step
@inline initial_volume(i, j, k, grid, σ⁰) = static_volume(i, j, k, grid) * initial_stretching(i, j, σ⁰)

"""
$(TYPEDSIGNATURES)

Return the volume swept through the `x`-face `i, j, k` during `Δt`, consistently with
`flux_div_xyᶜᶜᶜ`.
"""
@inline ffsl_volume_flux(i, j, k, grid, ::XSweep, u, Δt) = Δt * Ax_qᶠᶜᶜ(i, j, k, grid, u)
@inline ffsl_volume_flux(i, j, k, grid, ::YSweep, v, Δt) = Δt * Ay_qᶜᶠᶜ(i, j, k, grid, v)

"""
$(TYPEDSIGNATURES)

Return the signed index-space Courant number `± (n + r)` at the face `i, j, k` in the direction `sweep`:
the region swept by the volume flux through the face covers `n` whole upstream cells and the
fraction `r` of the next one. Cell volumes are those at the beginning of the time step. The
swept region is truncated at the first inactive cell, and at `Cmax` whole cells.
"""
@inline function swept_courant_number(i, j, k, grid, sweep, U, Δt, σ⁰, ::Val{Cmax}) where Cmax
    F = ffsl_volume_flux(i, j, k, grid, sweep, U, Δt)
    positive = F > 0
    first_offset = ifelse(positive, -1, 0)
    step = ifelse(positive, -1, 1)

    remaining = abs(F)
    n = zero(remaining)
    r = zero(remaining)
    searching = true

    for m in 0:Cmax
        ii, jj = offset_indices(sweep, i, j, first_offset + step * m)
        V = initial_volume(ii, jj, k, grid, σ⁰)
        active = !ffsl_inactive_cell(ii, jj, k, grid)
        whole = searching & active & (remaining >= V) & (m < Cmax)
        partial = searching & active & !whole
        r = ifelse(partial, min(remaining / V, one(r)), r)
        n = ifelse(whole, n + 1, n)
        remaining = ifelse(whole, remaining - V, remaining)
        searching = whole
    end

    return ifelse(positive, n + r, - n - r)
end

@kernel function _compute_swept_courant_numbers!(s, grid, sweep, U, Δt, σ⁰, Cmax)
    i, j, k = @index(Global, NTuple)
    @inbounds s[i, j, k] = swept_courant_number(i, j, k, grid, sweep, U, Δt, σ⁰, Cmax)
end

compute_swept_courant_numbers!(s, grid::XFlatGrid, ::XSweep, args...) = nothing
compute_swept_courant_numbers!(s, grid::YFlatGrid, ::YSweep, args...) = nothing

function compute_swept_courant_numbers!(s, grid, sweep, U, Δt, σ⁰, Cmax)
    Nx, Ny, Nz = size(grid)
    worksize = sweep isa XSweep ? (Nx+1, Ny, Nz) : (Nx, Ny+1, Nz)
    launch!(architecture(grid), grid, worksize, _compute_swept_courant_numbers!, s, grid, sweep, U, Δt, σ⁰, Cmax)
    return nothing
end

#####
##### Swept regions
#####

"""
$(TYPEDSIGNATURES)

Return the swept region through the face `i, j, k` in the direction `sweep`, as a `NamedTuple` holding
the volume flux `F`, the number of whole cells `n`, the fraction `r`, the upstream direction, and
the upstream cell volumes. The region is shared by every tracer advected through the face.
"""
@inline function swept_region(i, j, k, grid, sweep, s, U, Δt, σ⁰, ::Val{Cmax}) where Cmax
    F = ffsl_volume_flux(i, j, k, grid, sweep, U, Δt)
    sᵢ = @inbounds s[i, j, k]
    positive = sᵢ > 0
    a = abs(sᵢ)
    n = unsafe_trunc(Int, a)
    r = a - n
    first_offset = ifelse(positive, -1, 0)
    step = ifelse(positive, -1, 1)

    volumes = ntuple(Val(Cmax + 1)) do μ
        ii, jj = offset_indices(sweep, i, j, first_offset + step * (μ - 1))
        initial_volume(ii, jj, k, grid, σ⁰)
    end

    return (; F, n, r, positive, first_offset, step, volumes)
end

@inline swept_region(i, j, k, grid::XFlatGrid, ::XSweep, args...) = nothing
@inline swept_region(i, j, k, grid::YFlatGrid, ::YSweep, args...) = nothing

"""
$(TYPEDSIGNATURES)

Return the flux of `q` through a face: the volume flux times the average of `q` over the swept `region`.
"""
@inline function swept_flux(i, j, k, grid, sweep, region, q, limiter, ::Val{Cmax}) where Cmax
    (; F, n, r, positive, first_offset, step, volumes) = region

    mass = zero(r)
    volume = zero(r)
    Vⁿ = zero(r)

    for m in 0:Cmax
        ii, jj = offset_indices(sweep, i, j, first_offset + step * m)
        Vᵐ = @inbounds volumes[m+1]
        whole = m < n
        mass += ifelse(whole, Vᵐ * @inbounds(q[ii, jj, k]), zero(r))
        volume += ifelse(whole, Vᵐ, zero(r))
        Vⁿ = ifelse(m == n, Vᵐ, Vⁿ)
    end

    ii, jj = offset_indices(sweep, i, j, first_offset + step * n)
    q̄ = fractional_average(ii, jj, k, sweep, q, r, positive, limiter)
    mass += r * Vⁿ * q̄
    volume += r * Vⁿ

    i₀, j₀ = offset_indices(sweep, i, j, first_offset)
    q₀ = @inbounds q[i₀, j₀, k]
    q̂ = ifelse(volume > 0, mass / volume, q₀)

    return F * q̂
end

@inline swept_flux(i, j, k, grid, sweep, ::Nothing, args...) = zero(grid)

@inline volume_flux(region) = region.F
@inline volume_flux(::Nothing) = 0

#####
##### Piecewise-parabolic reconstruction in index space
#####

@inline function neighbour(sweep, q, i, j, k, m)
    ii, jj = offset_indices(sweep, i, j, m)
    return @inbounds q[ii, jj, k]
end

@inline function ppm_edges(i, j, k, sweep, q, ::Nothing)
    q₋₂ = neighbour(sweep, q, i, j, k, -2)
    q₋₁ = neighbour(sweep, q, i, j, k, -1)
    q₀  = neighbour(sweep, q, i, j, k, 0)
    q₊₁ = neighbour(sweep, q, i, j, k, 1)
    q₊₂ = neighbour(sweep, q, i, j, k, 2)
    qᴸ = (7 * (q₋₁ + q₀) - (q₋₂ + q₊₁)) / 12
    qᴿ = (7 * (q₀ + q₊₁) - (q₋₁ + q₊₂)) / 12
    return qᴸ, qᴿ
end

@inline function monotonized_slope(q₋, q₀, q₊)
    δ = (q₊ - q₋) / 2
    δₘ = min(abs(δ), 2 * abs(q₀ - q₋), 2 * abs(q₊ - q₀))
    return ifelse((q₊ - q₀) * (q₀ - q₋) > 0, copysign(δₘ, δ), zero(δ))
end

@inline function ppm_edges(i, j, k, sweep, q, ::MonotonePPMLimiter)
    q₋₂ = neighbour(sweep, q, i, j, k, -2)
    q₋₁ = neighbour(sweep, q, i, j, k, -1)
    q₀  = neighbour(sweep, q, i, j, k, 0)
    q₊₁ = neighbour(sweep, q, i, j, k, 1)
    q₊₂ = neighbour(sweep, q, i, j, k, 2)

    δ₋ = monotonized_slope(q₋₂, q₋₁, q₀)
    δ₀ = monotonized_slope(q₋₁, q₀, q₊₁)
    δ₊ = monotonized_slope(q₀, q₊₁, q₊₂)

    qᴸ = (q₋₁ + q₀) / 2 - (δ₀ - δ₋) / 6
    qᴿ = (q₀ + q₊₁) / 2 - (δ₊ - δ₀) / 6

    # Flatten the parabola at extrema, and steepen it where it would overshoot
    extremum = (qᴿ - q₀) * (q₀ - qᴸ) <= 0
    qᴸ = ifelse(extremum, q₀, qᴸ)
    qᴿ = ifelse(extremum, q₀, qᴿ)

    Δq = qᴿ - qᴸ
    q₆ = 6 * q₀ - 3 * (qᴸ + qᴿ)
    overshoot_left  = Δq * q₆ > Δq^2
    overshoot_right = - Δq^2 > Δq * q₆
    qᴸ′ = ifelse(overshoot_left,  3 * q₀ - 2 * qᴿ, qᴸ)
    qᴿ′ = ifelse(overshoot_right, 3 * q₀ - 2 * qᴸ, qᴿ)

    return qᴸ′, qᴿ′
end

"""
$(TYPEDSIGNATURES)

Return the mean of the parabola reconstructed in cell `i, j, k` over the fraction `r` of the cell
adjacent to its downstream face: the right part of the cell if `positive`, otherwise the left part.
"""
@inline function fractional_average(i, j, k, sweep, q, r, positive, limiter)
    q₀ = neighbour(sweep, q, i, j, k, 0)
    qᴸ, qᴿ = ppm_edges(i, j, k, sweep, q, limiter)
    Δq = qᴿ - qᴸ
    q₆ = 6 * q₀ - 3 * (qᴸ + qᴿ)
    curvature = (1 - 2r / 3) * q₆
    right_average = qᴿ - r / 2 * (Δq - curvature)
    left_average  = qᴸ + r / 2 * (Δq + curvature)
    return ifelse(positive, right_average, left_average)
end

#####
##### Horizontal step: inner (advective-form) and outer (flux-form) operators
#####

@inline function swept_regions(i, j, k, grid, geometry, U, Δt, Cmax)
    (; sˣ, sʸ, σ⁰) = geometry
    Rˣ⁻ = swept_region(i,   j,   k, grid, XSweep(), sˣ, U.u, Δt, σ⁰, Cmax)
    Rˣ⁺ = swept_region(i+1, j,   k, grid, XSweep(), sˣ, U.u, Δt, σ⁰, Cmax)
    Rʸ⁻ = swept_region(i,   j,   k, grid, YSweep(), sʸ, U.v, Δt, σ⁰, Cmax)
    Rʸ⁺ = swept_region(i,   j+1, k, grid, YSweep(), sʸ, U.v, Δt, σ⁰, Cmax)
    return Rˣ⁻, Rˣ⁺, Rʸ⁻, Rʸ⁺
end

@inline function advective_form_update(i, j, k, grid, sweep, R⁻, R⁺, q, qᵢ, V⁰, limiter, Cmax)
    i⁺, j⁺ = offset_indices(sweep, i, j, 1)
    δF = volume_flux(R⁺) - volume_flux(R⁻)
    δX = swept_flux(i⁺, j⁺, k, grid, sweep, R⁺, q, limiter, Cmax) -
         swept_flux(i,  j,  k, grid, sweep, R⁻, q, limiter, Cmax)
    return - (δX - qᵢ * δF) / V⁰
end

# qˣ = q + ½ aˣ(q) and qʸ = q + ½ aʸ(q), where a is the one-dimensional advective-form update
@kernel function _ffsl_inner_step!(qˣ, qʸ, tracers, grid, geometry, U, Δt, limiter, Cmax)
    i, j, k = @index(Global, NTuple)

    Rˣ⁻, Rˣ⁺, Rʸ⁻, Rʸ⁺ = swept_regions(i, j, k, grid, geometry, U, Δt, Cmax)
    V⁰ = initial_volume(i, j, k, grid, geometry.σ⁰)
    inactive = ffsl_inactive_cell(i, j, k, grid)

    for n in 1:length(tracers)
        q = tracers[n]
        qᵢ = @inbounds q[i, j, k]
        aˣ = advective_form_update(i, j, k, grid, XSweep(), Rˣ⁻, Rˣ⁺, q, qᵢ, V⁰, limiter, Cmax)
        aʸ = advective_form_update(i, j, k, grid, YSweep(), Rʸ⁻, Rʸ⁺, q, qᵢ, V⁰, limiter, Cmax)
        @inbounds qˣ[n][i, j, k] = ifelse(inactive, qᵢ, qᵢ + aˣ / 2)
        @inbounds qʸ[n][i, j, k] = ifelse(inactive, qᵢ, qᵢ + aʸ / 2)
    end
end

# σc ← σc - [δx(F q̂[qʸ]) + δy(F q̂[qˣ])] / Vₛ, where Vₛ = V / σ is the static cell volume
@kernel function _ffsl_outer_step!(σc, qˣ, qʸ, grid, geometry, U, Δt, limiter, Cmax)
    i, j, k = @index(Global, NTuple)

    Rˣ⁻, Rˣ⁺, Rʸ⁻, Rʸ⁺ = swept_regions(i, j, k, grid, geometry, U, Δt, Cmax)
    Vₛ = static_volume(i, j, k, grid)
    inactive = ffsl_inactive_cell(i, j, k, grid)

    for n in 1:length(σc)
        δX = swept_flux(i+1, j, k, grid, XSweep(), Rˣ⁺, qʸ[n], limiter, Cmax) -
             swept_flux(i,   j, k, grid, XSweep(), Rˣ⁻, qʸ[n], limiter, Cmax)
        δY = swept_flux(i, j+1, k, grid, YSweep(), Rʸ⁺, qˣ[n], limiter, Cmax) -
             swept_flux(i, j,   k, grid, YSweep(), Rʸ⁻, qˣ[n], limiter, Cmax)
        @inbounds σc[n][i, j, k] -= ifelse(inactive, zero(Vₛ), (δX + δY) / Vₛ)
    end
end

"""
$(TYPEDSIGNATURES)

Take one horizontal flux-form semi-Lagrangian step of length `Δt` for several tracers at once.

`σc` is a tuple of the (stretched) tracer contents `σ * c` at the beginning of the step, which are updated in place.
`tracers` holds the tracer concentrations at the beginning of the step, with filled halos. `qˣ` and `qʸ` are
tuples of scratch fields for the inner advective-form operators. `geometry` holds the shared swept Courant
numbers `sˣ`, `sʸ` and the grid stretching `σ⁰` at the beginning of the step (`nothing` for static grids).
`U` holds the volume-conserving horizontal transport velocities `u` and `v`.
"""
function flux_form_semi_lagrangian_step!(σc, tracers, qˣ, qʸ, geometry, grid, U, Δt, limiter, Cmax::Val)
    arch = architecture(grid)
    FT = eltype(grid)
    Δt = convert(FT, Δt)

    compute_swept_courant_numbers!(geometry.sˣ, grid, XSweep(), U.u, Δt, geometry.σ⁰, Cmax)
    compute_swept_courant_numbers!(geometry.sʸ, grid, YSweep(), U.v, Δt, geometry.σ⁰, Cmax)

    launch!(arch, grid, :xyz, _ffsl_inner_step!, qˣ, qʸ, tracers, grid, geometry, U, Δt, limiter, Cmax)

    fill_halo_regions!(qˣ)
    fill_halo_regions!(qʸ)

    launch!(arch, grid, :xyz, _ffsl_outer_step!, σc, qˣ, qʸ, grid, geometry, U, Δt, limiter, Cmax)

    return nothing
end
