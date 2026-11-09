using Oceananigans.Grids: AbstractGrid
using Oceananigans.Fields: CenterField
using Oceananigans.BoundaryConditions: fill_bottom_and_top_halo!
using Oceananigans.Utils: launch!
using KernelAbstractions: @index, @kernel

"""
    BoundsPreservation(minimum_value, maximum_value; adaptive_weight=false, maximum_cfl=nothing, flux_limiter=nothing)

The interval a tracer is restricted to and the options of the limiters that keep it there, for the `bounds` of
[`WENO`](@ref): the maximum-principle-satisfying rescaling of [Zhang and Shu (2010)](@cite ZhangShu2010), which
guarantees the interval for a forward Euler or strong-stability-preserving update whose Courant numbers satisfy
`λˣ + λʸ + λᶻ ≤ maximum_cfl`, and optionally a [`FluxLimiter`](@ref) that limits the fluxes between cells so
that the tracer stays within the interval for any time step.

The rescaling splits each cell mean into its two face values and a remainder, `c̄ = ω̂₁ c⁻ + ω̂₁ c⁺ + (1 - 2ω̂₁) p̃`,
with the quadrature weight `ω̂₁ ∈ (0, 1/2)` on the face values, and scales the face values towards `c̄` until all
three lie in the interval. A forward Euler update then stays in the interval wherever the cell's outgoing Courant
number `Λ`, summed over the three directions, satisfies `Λ ≤ ω̂₁`. The weight can be fixed at `maximum_cfl`, or
adapt to each cell as `ω̂₁ = clamp(Λ, minimum_weight, maximum_cfl)`, the smallest weight that guarantees the
interval there, which spares smooth extrema that touch a bound from being clipped.

Keyword arguments
=================
- `adaptive_weight`: Whether the weight adapts to each cell's outgoing Courant number `Λ`. Below the lower clamp
                     `minimum_weight = 4 / ((N + 1) (N + 3))` of WENO order `N` (the end weight of the Gauss–Lobatto
                     quadrature that is exact for the reconstruction) the rescaling clips smooth profiles, so the
                     weight is held there; [`WENO`](@ref) sets it from the order, and near the boundaries of a
                     `Bounded` direction, where a lower-order buffer scheme reconstructs the faces, that scheme's
                     larger end weight applies instead. Default: `false`.
- `maximum_cfl`: The Courant number up to which the rescaling alone guarantees the interval: the fixed weight
                 `ω̂₁`, or the upper clamp of the adaptive weight. Raising it admits a larger time step and
                 rescales more aggressively. With a `flux_limiter` the interval is guaranteed for any time step,
                 and `maximum_cfl` only sets how hard the rescaling clips the reconstructions. Default: `5//18`
                 for the fixed weight, and `1//6` for the adaptive weight, the largest weight at which the
                 rescaling leaves smooth extrema alone. With the adaptive weight and this default, the rescaling
                 alone therefore no longer guarantees the interval at Courant numbers above `1/6`: to keep the
                 guarantee there, raise `maximum_cfl` or add a `flux_limiter`.
- `flux_limiter`: `nothing` or `false` for none, `true` or a [`FluxLimiter`](@ref) for one. Default: `nothing`.

The materialized struct also holds the field `reconstruction_limiter` with the cell-wise rescaling factor ``θ``,
which is `nothing` until the scheme is materialized on a grid, like the fields of `flux_limiter`.
"""
struct BoundsPreservation{FT, W, L, F}
    minimum_value :: FT
    maximum_value :: FT
    maximum_cfl :: FT           # the fixed quadrature weight ω̂₁, or the upper clamp of the adaptive weight
    minimum_weight :: W         # the lower clamp of the adaptive weight, unresolved until WENO sets it from the order, or nothing
    reconstruction_limiter :: L
    flux_limiter :: F
end

# The `minimum_weight` of an adaptive weight before the order is known. Bounds carrying it are neither
# fixed- nor adaptive-weight bounds, so they cannot enter a kernel before `WENO` resolves them.
struct UnresolvedMinimumWeight end

function BoundsPreservation(minimum_value, maximum_value; adaptive_weight=false, maximum_cfl=nothing, flux_limiter=nothing)
    minimum_value < maximum_value || throw(ArgumentError("minimum_value must be below maximum_value, found ($minimum_value, $maximum_value)"))
    adaptive_weight isa Bool || throw(ArgumentError("adaptive_weight must be a Bool, found $adaptive_weight"))

    maximum_cfl = something(maximum_cfl, adaptive_weight ? 1//6 : 5//18)
    0 < maximum_cfl < 1//2 || throw(ArgumentError("maximum_cfl must lie in (0, 1/2), found $maximum_cfl"))

    flux_limiter isa Union{Bool, Nothing, FluxLimiter} || throw(ArgumentError("flux_limiter must be a Bool, nothing or a FluxLimiter, found $flux_limiter"))
    flux_limiter = flux_limiter === true ? FluxLimiter() : flux_limiter === false ? nothing : flux_limiter

    minimum_weight = adaptive_weight ? UnresolvedMinimumWeight() : nothing
    minimum_value, maximum_value, maximum_cfl = promote(minimum_value, maximum_value, maximum_cfl)
    return BoundsPreservation(minimum_value, maximum_value, maximum_cfl, minimum_weight, nothing, flux_limiter)
end

# The bounds with their numbers converted to the float type of a WENO scheme of the given order, which
# also fixes the lower clamp of an adaptive weight: the end weight 4 / ((N + 1) (N + 3)) of the Gauss–Lobatto
# quadrature with (N + 3) / 2 points, which is exact for the degree N - 1 reconstruction of WENO order N.
# It cannot exceed the upper clamp.
with_order_and_float_type(order, FT, ::Nothing) = nothing

function with_order_and_float_type(order, FT, bounds::BoundsPreservation)
    maximum_cfl = convert(FT, bounds.maximum_cfl)
    minimum_weight = minimum_weight_for_order(order, FT, bounds.minimum_weight, maximum_cfl)
    return BoundsPreservation(convert(FT, bounds.minimum_value), convert(FT, bounds.maximum_value),
                              maximum_cfl, minimum_weight, bounds.reconstruction_limiter, bounds.flux_limiter)
end

minimum_weight_for_order(order, FT, ::Nothing, maximum_cfl) = nothing
minimum_weight_for_order(order, FT, ::UnresolvedMinimumWeight, maximum_cfl) = min(convert(FT, 4 // ((order + 1) * (order + 3))), maximum_cfl)
minimum_weight_for_order(order, FT, minimum_weight::Number, maximum_cfl) = min(convert(FT, minimum_weight), maximum_cfl)

const FixedWeightBounds = BoundsPreservation{<:Any, Nothing}
const AdaptiveWeightBounds = BoundsPreservation{<:Any, <:Number}

Base.show(io::IO, bounds::BoundsPreservation) =
    print(io, "(", bounds.minimum_value, ", ", bounds.maximum_value, ")", weight_summary(bounds), flux_limiter_summary(bounds))

weight_summary(::FixedWeightBounds) = ""
weight_summary(::AdaptiveWeightBounds) = ", adaptive_weight=true"
weight_summary(::BoundsPreservation{<:Any, UnresolvedMinimumWeight}) = ", adaptive_weight=true"

flux_limiter_summary(::BoundsPreservation) = ""

Adapt.adapt_structure(to, bounds::BoundsPreservation) =
    BoundsPreservation(Adapt.adapt(to, bounds.minimum_value),
                       Adapt.adapt(to, bounds.maximum_value),
                       Adapt.adapt(to, bounds.maximum_cfl),
                       Adapt.adapt(to, bounds.minimum_weight),
                       Adapt.adapt(to, bounds.reconstruction_limiter),
                       Adapt.adapt(to, bounds.flux_limiter))

Architectures.on_architecture(to, bounds::BoundsPreservation) =
    BoundsPreservation(on_architecture(to, bounds.minimum_value),
                       on_architecture(to, bounds.maximum_value),
                       on_architecture(to, bounds.maximum_cfl),
                       on_architecture(to, bounds.minimum_weight),
                       on_architecture(to, bounds.reconstruction_limiter),
                       on_architecture(to, bounds.flux_limiter))

const _ε₂ = 1e-20

# Note: this can probably be generalized to include UpwindBiased
const BoundsPreservingWENO = WENO{<:Any, <:Any, <:Any, <:Any, <:BoundsPreservation}
const FixedWeightWENO = WENO{<:Any, <:Any, <:Any, <:Any, <:FixedWeightBounds}
const AdaptiveWeightWENO = WENO{<:Any, <:Any, <:Any, <:Any, <:AdaptiveWeightBounds}

# The limiter is also computed in the first horizontal halo layer, whose reconstructions reach one cell further
@inline Grids.required_halo_size_x(::WENO{N, <:Any, <:Any, <:Any, <:BoundsPreservation}) where N = N + 1
@inline Grids.required_halo_size_y(::WENO{N, <:Any, <:Any, <:Any, <:BoundsPreservation}) where N = N + 1

#####
##### Quadrature weight ω̂₁ on the face values, from cᵢ = ω̂₁ c⁻ + ω̂₁ c⁺ + (1 - 2ω̂₁) p̃ in each direction
#####
#
# With the face values rescaled into [m, M], a forward Euler update of cell i with the upwind fluxes
# leaves c⁻ and c⁺ with the coefficients ω̂₁ - Δt A⁻ max(0, -u⁻) / V and ω̂₁ - Δt A⁺ max(0, u⁺) / V,
# where u⁻ and u⁺ are the velocities through its two faces, and p̃ with 1 - 2ω̂₁: a convex combination
# of values in [m, M], and so in [m, M], as long as the larger of the two outflows in each direction,
# summed over the directions into the cell's outgoing Courant number Λᵢ, satisfies Λᵢ ≤ ω̂₁. Every
# ω̂₁ ≥ Λᵢ therefore guarantees the bounds, and the smallest one rescales least: for a smooth profile
# p̃ - cᵢ = Δx² c″ (1 - 6ω̂₁) / (24 (1 - 2ω̂₁)) + ..., so at an extremum touching a bound p̃ lies outside
# the interval and θ clips the reconstruction, unless ω̂₁ ≤ 1/6. Below the Gauss–Lobatto end weight
# 4 / ((N + 1) (N + 3)) of WENO order N, though, the face values of a smooth profile themselves leave
# the interval that p̃ and cᵢ span, and θ clips again. The adaptive weight is ω̂₁ = clamp(Λᵢ, ω̂ₘᵢₙ, ω̂ₘₐₓ)
# with these two limits. Only the cell's own ω̂₁ enters its own update, so the weight may differ between cells.

# The volume leaving cell (i, j, k) per unit time through its faces in one direction: the larger of the
# outflows through the two faces, whose coefficients in the update must each stay non-negative.
@inline outgoing_volume_flux_x(i, j, k, grid::XFlatGrid, u) = zero(grid)
@inline outgoing_volume_flux_y(i, j, k, grid::YFlatGrid, v) = zero(grid)
@inline outgoing_volume_flux_z(i, j, k, grid::ZFlatGrid, w) = zero(grid)

@inline outgoing_volume_flux_x(i, j, k, grid, u) = @inbounds max(0, Axᶠᶜᶜ(i+1, j, k, grid) * u[i+1, j, k], - Axᶠᶜᶜ(i, j, k, grid) * u[i, j, k])
@inline outgoing_volume_flux_y(i, j, k, grid, v) = @inbounds max(0, Ayᶜᶠᶜ(i, j+1, k, grid) * v[i, j+1, k], - Ayᶜᶠᶜ(i, j, k, grid) * v[i, j, k])
@inline outgoing_volume_flux_z(i, j, k, grid, w) = @inbounds max(0, Azᶜᶜᶠ(i, j, k+1, grid) * w[i, j, k+1], - Azᶜᶜᶠ(i, j, k, grid) * w[i, j, k])

# The outgoing Courant number Λᵢ of cell (i, j, k) over the time step Δt of the coming forward Euler update
@inline function outgoing_courant_number(i, j, k, grid, U, Δt)
    outgoing = outgoing_volume_flux_x(i, j, k, grid, U.u) +
               outgoing_volume_flux_y(i, j, k, grid, U.v) +
               outgoing_volume_flux_z(i, j, k, grid, U.w)

    return Δt * V⁻¹ᶜᶜᶜ(i, j, k, grid) * outgoing
end

# Near the boundaries of a Bounded direction the faces are reconstructed by a lower-order buffer scheme
# (see topologically_conditional_interpolation.jl), whose end weight 1 / (N (N + 1)) for WENO{N}, of order
# 2N - 1, is larger, so in each direction the lower clamp is that of the scheme reconstructing both faces of
# the cell there. The buffer schemes carry no bounds once materialized, so their weights come from their
# order, capped at the upper clamp like the outer scheme's; the lowest-order fall-backs reconstruct at most
# linearly, with end weight 1/2, and take the cap. Each direction may use its own weight: writing
# cᵢ = Σ_d a_d [ω̂₁ᵈ c⁻ᵈ + ω̂₁ᵈ c⁺ᵈ + (1 - 2ω̂₁ᵈ) p̃ᵈ] with a_d = Λᵈ / Λᵢ, the coefficient of each face value in
# the update is at least a_d (ω̂₁ᵈ - Λᵢ), so the update stays in [m, M] whenever every ω̂₁ᵈ ≥ Λᵢ.
for (dir, d) in zip((:x, :y, :z), (1, 2, 3))
    minimum_weight_ξ = Symbol(:minimum_weight_, dir)
    buffer_minimum_weight_ξ = Symbol(:buffer_minimum_weight_, dir)
    outside_buffer = Symbol(:outside_biased_halo_, dir, :ᶠ)
    index = (:i, :j, :k)[d]
    Nξ = (:Nx, :Ny, :Nz)[d]
    AGξ = (:AGX, :AGY, :AGZ)[d]

    @eval begin
        # Not Bounded: the outer scheme reconstructs every face
        @inline $minimum_weight_ξ(i, j, k, grid, scheme::AdaptiveWeightWENO) = scheme.bounds.minimum_weight

        @inline function $minimum_weight_ξ(i, j, k, grid::$AGξ, scheme::AdaptiveWeightWENO)
            bounds = scheme.bounds
            outer = $outside_buffer($index, topology(grid, $d), grid.$Nξ, scheme) &
                    $outside_buffer($index + 1, topology(grid, $d), grid.$Nξ, scheme)
            return ifelse(outer, bounds.minimum_weight, $buffer_minimum_weight_ξ(i, j, k, grid, scheme.buffer_scheme, bounds.maximum_cfl))
        end

        @inline $buffer_minimum_weight_ξ(i, j, k, grid, scheme, maximum_cfl) = maximum_cfl

        @inline function $buffer_minimum_weight_ξ(i, j, k, grid, scheme::WENO{N, FT}, maximum_cfl) where {N, FT}
            outer = $outside_buffer($index, topology(grid, $d), grid.$Nξ, scheme) &
                    $outside_buffer($index + 1, topology(grid, $d), grid.$Nξ, scheme)
            weight = min(convert(FT, 1 // (N * (N + 1))), maximum_cfl)
            return ifelse(outer, weight, $buffer_minimum_weight_ξ(i, j, k, grid, scheme.buffer_scheme, maximum_cfl))
        end
    end
end

# The weights of the three directions. The fixed weight reads neither the velocities nor the time step.
@inline function quadrature_weights(i, j, k, grid, scheme::FixedWeightWENO, U, Δt)
    ω̂₁ = scheme.bounds.maximum_cfl
    return ω̂₁, ω̂₁, ω̂₁
end

@inline function quadrature_weights(i, j, k, grid, scheme::AdaptiveWeightWENO, U, Δt)
    bounds = scheme.bounds
    Λ = convert(eltype(grid), outgoing_courant_number(i, j, k, grid, U, Δt))
    # Before the first time step the clock's Δt is infinite, and Λ is not finite where the velocity vanishes
    Λ = ifelse(isfinite(Λ), Λ, bounds.maximum_cfl)
    ω̂₁ˣ = clamp(Λ, minimum_weight_x(i, j, k, grid, scheme), bounds.maximum_cfl)
    ω̂₁ʸ = clamp(Λ, minimum_weight_y(i, j, k, grid, scheme), bounds.maximum_cfl)
    ω̂₁ᶻ = clamp(Λ, minimum_weight_z(i, j, k, grid, scheme), bounds.maximum_cfl)
    return ω̂₁ˣ, ω̂₁ʸ, ω̂₁ᶻ
end

#####
##### Cell-wise rescaling factor θ
#####

@inline function reconstruction_extrema_x(i, j, k, grid, scheme, c, m, M, ω̂₁)
    c⁻ = _biased_interpolate_xᶠᵃᵃ(i,   j, k, grid, scheme, RightBias, c)
    c⁺ = _biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, LeftBias,  c)
    cᵢ = @inbounds c[i, j, k]
    p̃  = (cᵢ - ω̂₁ * (c⁻ + c⁺)) / (1 - 2ω̂₁)

    return min(m, p̃, c⁻, c⁺), max(M, p̃, c⁻, c⁺)
end

@inline function reconstruction_extrema_y(i, j, k, grid, scheme, c, m, M, ω̂₁)
    c⁻ = _biased_interpolate_yᵃᶠᵃ(i, j,   k, grid, scheme, RightBias, c)
    c⁺ = _biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, LeftBias,  c)
    cᵢ = @inbounds c[i, j, k]
    p̃  = (cᵢ - ω̂₁ * (c⁻ + c⁺)) / (1 - 2ω̂₁)

    return min(m, p̃, c⁻, c⁺), max(M, p̃, c⁻, c⁺)
end

@inline function reconstruction_extrema_z(i, j, k, grid, scheme, c, m, M, ω̂₁)
    c⁻ = _biased_interpolate_zᵃᵃᶠ(i, j, k,   grid, scheme, RightBias, c)
    c⁺ = _biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, LeftBias,  c)
    cᵢ = @inbounds c[i, j, k]
    p̃  = (cᵢ - ω̂₁ * (c⁻ + c⁺)) / (1 - 2ω̂₁)

    return min(m, p̃, c⁻, c⁺), max(M, p̃, c⁻, c⁺)
end

# Support for Flat directions
@inline reconstruction_extrema_x(i, j, k, ::AbstractGrid{FT, Flat, TY, TZ}, scheme, c, m, M, ω̂₁) where {FT, TY, TZ} = (m, M)
@inline reconstruction_extrema_y(i, j, k, ::AbstractGrid{FT, TX, Flat, TZ}, scheme, c, m, M, ω̂₁) where {FT, TX, TZ} = (m, M)
@inline reconstruction_extrema_z(i, j, k, ::AbstractGrid{FT, TX, TY, Flat}, scheme, c, m, M, ω̂₁) where {FT, TX, TY} = (m, M)

# `U` and `Δt` are the transport velocities and time step of the coming forward Euler update, which only
# an adaptive weight reads (see `quadrature_weights`); a fixed weight accepts `nothing` for both.
@inline function bounds_preserving_limiter(i, j, k, grid, scheme::BoundsPreservingWENO, c, U, Δt)
    FT = eltype(c)
    ε₂ = convert(FT, _ε₂)

    cᵐⁱⁿ = scheme.bounds.minimum_value
    cᵐᵃˣ = scheme.bounds.maximum_value
    ω̂₁ˣ, ω̂₁ʸ, ω̂₁ᶻ = quadrature_weights(i, j, k, grid, scheme, U, Δt)

    cᵢ = @inbounds c[i, j, k]

    m, M = cᵢ, cᵢ
    m, M = reconstruction_extrema_x(i, j, k, grid, scheme, c, m, M, ω̂₁ˣ)
    m, M = reconstruction_extrema_y(i, j, k, grid, scheme, c, m, M, ω̂₁ʸ)
    m, M = reconstruction_extrema_z(i, j, k, grid, scheme, c, m, M, ω̂₁ᶻ)

    θᵐᵃˣ = abs((cᵐᵃˣ - cᵢ) / (M - cᵢ + ε₂))
    θᵐⁱⁿ = abs((cᵐⁱⁿ - cᵢ) / (m - cᵢ - ε₂))

    return min(θᵐᵃˣ, θᵐⁱⁿ, one(FT))
end

@kernel function _compute_bounds_preserving_limiter!(θ, grid, scheme, c, U, Δt)
    i, j, k = @index(Global, NTuple)
    @inbounds θ[i, j, k] = bounds_preserving_limiter(i, j, k, grid, scheme, c, U, Δt)
end

update_bounds_preserving_limiter!(scheme, grid, c, U, Δt) = nothing

update_bounds_preserving_limiter!(scheme::FluxFormAdvection, grid, c, U, Δt) = update_bounds_preserving_limiter!(scheme.z, grid, c, U, Δt)

"""
$(TYPEDSIGNATURES)

Fill the rescaling factor ``θ`` carried by a bounds-preserving `scheme` from the tracer `c`, for the coming
forward Euler update over `Δt` with the transport velocities `U`, which only an adaptive quadrature weight
reads (see [`BoundsPreservation`](@ref)). ``θ`` is computed in the first horizontal halo layer too, from
the halos of `c` and `U`, so that only its bottom and top halos are filled afterwards and no halo is
communicated.
"""
function update_bounds_preserving_limiter!(scheme::BoundsPreservingWENO, grid, c, U, Δt)
    isnothing(c) && return nothing # the `momentum` entry has no tracer to limit
    θ = scheme.bounds.reconstruction_limiter
    Δt = limiter_timestep(eltype(grid), Δt)
    parameters = horizontally_extended_interior_kernel_parameters(grid)
    launch!(architecture(grid), grid, parameters, _compute_bounds_preserving_limiter!, θ, grid, scheme, c, U, Δt)
    fill_bottom_and_top_halo!(θ.data, θ.boundary_conditions, θ.indices, (Center(), Center(), Center()), grid)
    return nothing
end

limiter_timestep(FT, ::Nothing) = nothing
limiter_timestep(FT, Δt) = convert(FT, Δt)

#####
##### Model interface
#####

"""
$(TYPEDSIGNATURES)

The velocities that advect the tracer `name` of `model`: its `velocities` unless the model extends this,
for instance with drift velocities or advective forcing.
"""
tracer_transport_velocities(model, ::Val{name}) where name = model.velocities

# Refresh the limiters of the scheme advecting the tracer `name` of `model` for the coming forward Euler
# update: the rescaling factor θ, and the fractions of a flux limiter. `adaptive_advection_timestep` is the
# Δt of the coming stage.
update_bounds_preserving_limiters!(scheme, model, val_name) = nothing

update_bounds_preserving_limiters!(scheme::FluxFormAdvection, model, val_name) =
    update_bounds_preserving_limiters!(scheme.z, model, val_name)

function update_bounds_preserving_limiters!(scheme::BoundsPreservingWENO, model, val_name::Val{name}) where name
    c = model.tracers[name]
    U = tracer_transport_velocities(model, val_name)
    Δt = adaptive_advection_timestep(model.timestepper, model.clock)
    update_bounds_preserving_limiter!(scheme, model.grid, c, U, Δt)
    return update_flux_limiter!(scheme, model.grid, c, U, Δt)
end

# The scheme-wise hook of `update_advection!(scheme, model, tracer)` knows no tracer name, so an adaptive
# weight transports with the model's `velocities`; a fixed weight needs neither velocities nor a time step.
update_bounds_preserving_limiter!(scheme, model, tracer) = nothing

update_bounds_preserving_limiter!(scheme::FluxFormAdvection, model, tracer) = update_bounds_preserving_limiter!(scheme.z, model, tracer)

update_bounds_preserving_limiter!(scheme::FixedWeightWENO, model, tracer) =
    update_bounds_preserving_limiter!(scheme, model.grid, tracer, nothing, nothing)

function update_bounds_preserving_limiter!(scheme::AdaptiveWeightWENO, model, tracer)
    Δt = adaptive_advection_timestep(model.timestepper, model.clock)
    return update_bounds_preserving_limiter!(scheme, model.grid, tracer, model.velocities, Δt)
end

@inline function rescaled_reconstruction(ĉ, i, j, k, grid, θ, c)
    cᵢ = @inbounds c[i, j, k]
    θᵢ = @inbounds θ[i, j, k]
    return θᵢ * (ĉ - cᵢ) + cᵢ
end

@inline without_bounds_preservation(scheme) = scheme

@inline function without_bounds_preservation(scheme::WENO{N, FT, WCT}) where {N, FT, WCT}
    return WENO{N, FT, WCT}(nothing, scheme.buffer_scheme, scheme.advecting_velocity_scheme, scheme.time_discretization)
end

#####
##### Bounded face fluxes and their divergence
#####

@inline div_Uc(i, j, k, grid, advection::BoundsPreservingWENO, U, ::ZeroField) = zero(grid)
@inline div_Uc(i, j, k, grid, advection::BoundsPreservingWENO, ::ZeroU, c) = zero(grid)
@inline div_Uc(i, j, k, grid, advection::BoundsPreservingWENO, ::ZeroU, ::ZeroField) = zero(grid)

# Is this immersed-boundary safe without having to extend it in ImmersedBoundaries.jl? I think so... (velocity on immmersed boundaries is masked to 0)
# For bounds preserving advection, we need fluxes at both cell-faces to compute the flux on one face.
# So we extend div_Uc in order to compute the fluxes at i and i+1 in one go and avoid recomputation.
@inline function div_Uc(i, j, k, grid, advection::BoundsPreservingWENO, U, c)
    div_x = bounded_tracer_flux_divergence_x(i, j, k, grid, advection, 1, U.u, c)
    div_y = bounded_tracer_flux_divergence_y(i, j, k, grid, advection, 1, U.v, c)
    div_z = bounded_tracer_flux_divergence_z(i, j, k, grid, advection, 1, U.w, c)

    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (div_x + div_y + div_z)
end

# Support for Flat directions
@inline bounded_face_flux_x(i, j, k, ::AbstractGrid{FT, Flat, TY, TZ}, advection::BoundsPreservingWENO, args...) where {FT, TY, TZ} = zero(FT)
@inline bounded_face_flux_y(i, j, k, ::AbstractGrid{FT, TX, Flat, TZ}, advection::BoundsPreservingWENO, args...) where {FT, TX, TZ} = zero(FT)
@inline bounded_face_flux_z(i, j, k, ::AbstractGrid{FT, TX, TY, Flat}, advection::BoundsPreservingWENO, args...) where {FT, TX, TY} = zero(FT)

@inline bounded_tracer_flux_divergence_x(i, j, k, ::AbstractGrid{FT, Flat, TY, TZ}, advection::BoundsPreservingWENO, args...) where {FT, TY, TZ} = zero(FT)
@inline bounded_tracer_flux_divergence_y(i, j, k, ::AbstractGrid{FT, TX, Flat, TZ}, advection::BoundsPreservingWENO, args...) where {FT, TX, TZ} = zero(FT)
@inline bounded_tracer_flux_divergence_z(i, j, k, ::AbstractGrid{FT, TX, TY, Flat}, advection::BoundsPreservingWENO, args...) where {FT, TX, TY} = zero(FT)

# The advective flux of `c` through the face (i, j, k), weighted by the density `ρ` (`1` for `div_Uc`),
# built from the reconstructions rescaled by θ on either side of the face.
@inline function bounded_face_flux_x(i, j, k, grid, advection::BoundsPreservingWENO, ρ, u, c)
    θ = advection.bounds.reconstruction_limiter
    cᴸ = _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, advection, LeftBias,  c)
    cᴿ = _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, advection, RightBias, c)
    cᴸ = rescaled_reconstruction(cᴸ, i-1, j, k, grid, θ, c)
    cᴿ = rescaled_reconstruction(cᴿ, i,   j, k, grid, θ, c)
    ũ = @inbounds u[i, j, k]
    return ℑxᶠᵃᵃ(i, j, k, grid, ρ) * Axᶠᶜᶜ(i, j, k, grid) * upwind_biased_product(ũ, cᴸ, cᴿ)
end

@inline function bounded_face_flux_y(i, j, k, grid, advection::BoundsPreservingWENO, ρ, v, c)
    θ = advection.bounds.reconstruction_limiter
    cᴸ = _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, advection, LeftBias,  c)
    cᴿ = _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, advection, RightBias, c)
    cᴸ = rescaled_reconstruction(cᴸ, i, j-1, k, grid, θ, c)
    cᴿ = rescaled_reconstruction(cᴿ, i, j,   k, grid, θ, c)
    ṽ = @inbounds v[i, j, k]
    return ℑyᵃᶠᵃ(i, j, k, grid, ρ) * Ayᶜᶠᶜ(i, j, k, grid) * upwind_biased_product(ṽ, cᴸ, cᴿ)
end

@inline function bounded_face_flux_z(i, j, k, grid, advection::BoundsPreservingWENO, ρ, w, c)
    θ = advection.bounds.reconstruction_limiter
    cᴸ = _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, advection, LeftBias,  c)
    cᴿ = _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, advection, RightBias, c)
    cᴸ = rescaled_reconstruction(cᴸ, i, j, k-1, grid, θ, c)
    cᴿ = rescaled_reconstruction(cᴿ, i, j, k,   grid, θ, c)
    w̃ = @inbounds w[i, j, k]
    return ℑzᵃᵃᶠ(i, j, k, grid, ρ) * Azᶜᶜᶠ(i, j, k, grid) * upwind_biased_product(w̃, cᴸ, cᴿ)
end

@inline bounded_tracer_flux_divergence_x(i, j, k, grid, advection::BoundsPreservingWENO, ρ, u, c) =
    bounded_face_flux_x(i+1, j, k, grid, advection, ρ, u, c) - bounded_face_flux_x(i, j, k, grid, advection, ρ, u, c)

@inline bounded_tracer_flux_divergence_y(i, j, k, grid, advection::BoundsPreservingWENO, ρ, v, c) =
    bounded_face_flux_y(i, j+1, k, grid, advection, ρ, v, c) - bounded_face_flux_y(i, j, k, grid, advection, ρ, v, c)

@inline bounded_tracer_flux_divergence_z(i, j, k, grid, advection::BoundsPreservingWENO, ρ, w, c) =
    bounded_face_flux_z(i, j, k+1, grid, advection, ρ, w, c) - bounded_face_flux_z(i, j, k, grid, advection, ρ, w, c)
