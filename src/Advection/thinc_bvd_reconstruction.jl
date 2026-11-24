#####
##### WENO-THINC-BVD: per cell, WENO or THINC face values, whichever jumps less across the cell's faces
#####

using Oceananigans.Utils: newton_div, BackendOptimizedDivision, launch!, KernelParameters
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: Field, CenterField
using Oceananigans.Grids: halo_size, topology
using Oceananigans.Operators: V⁻¹ᶜᶜᶜ
using KernelAbstractions: @index, @kernel

struct THINCBVD{N, FT, TD, W, S, R} <: AbstractUpwindBiasedAdvectionScheme{N, FT, TD}
    buffer_scheme :: W   # WENO, the smooth reconstruction
    steepnesses :: S     # (Val(n), β, tanh β, 1 / tanh β, 1 / cosh β) of each THINC steepness β = nβ₀
    base_steepness :: FT # β₀, the largest steepness of which every β is an integer multiple
    moments :: Bool      # whether a model gives each advected tracer its first moments as tracers
    reconstructions :: R # once materialized, the advected tracer with ghost values, the WENO edges of its cells and the flux limiter

    THINCBVD{N, FT, TD}(buffer_scheme::W, steepnesses::S, base_steepness, moments, reconstructions::R) where {N, FT, TD, W, S, R} =
        new{N, FT, TD, W, S, R}(buffer_scheme, steepnesses, base_steepness, moments, reconstructions)
end

"""
$(SIGNATURES)

Construct a WENO-THINC-BVD tracer advection scheme. A cell whose value lies between its neighbours' reconstructs its face
values with `WENO(FT; order)` or with THINC, a hyperbolic tangent step of one of `steepnesses` between the neighbours'
values, and takes the reconstruction with the smallest total jump across its two faces (boundary variation). The other cells
use WENO. The stencils read the tracer with every inactive cell taking the values of the nearest active column, continued linearly
below its bed and above a closed top. Each face flux is the upwind flux plus the largest fraction of the remainder (the
antidiffusive flux) that keeps the stage update of both cells of the face within the range of the tracer's point values (each cell
widened by half its minmod slope, one-sided next to a wall) at the first time step, widened by each cell and the values flowing
into it (Zalesak, 1979). A model materializes the scheme with up to nine
fields per tracer: the tracer with these ghost values, the two WENO edges of every cell and the fractions of the antidiffusive
fluxes that raise and that lower every cell, refreshed by `update_tracer_advection!` before the tendencies are
computed. The scheme needs halos one cell wider than `WENO(FT; order)`. All steepnesses share one exponential per cell,
cheapest when they are small integer multiples of a common value, as 1.6 and 2.5 are of 1/10. With `steepnesses=()` every
cell takes WENO's face values, read from the tracer with ghost values. With `time_discretization=AdaptiveVerticallyImplicitDiscretization()`
the vertical transport beyond its `cfl` is implicit and first-order upwind: the range then bounds the tracer after the explicit
transport, which the implicit solve keeps within it. With `moments=true`, a `HydrostaticFreeSurfaceModel` gives each tracer `T`
advected by the scheme the tracers `Tˣ`, `Tʸ` and `Tᶻ`, its first moments ⟨ξ T⟩ over the wet part of each cell along x, y and z (ξ
from -1/2 to 1/2 across the cell, or across its level for a cut cell), carried by the fluxes of `T`. A cell between two water masses,
where the range of its 3³ neighbourhood exceeds 3/4 of that of its 5³ neighbourhood, keeps its moments and transports the tanh
profile with its mean and moments. Every other cell takes at every stage the moments of the linear profile with the gradient of the
degree-6 polynomial through the 7 neighbouring means, and transports the least-variance profile with its mean and moments within the
range of its 3³ neighbourhood.

```jldoctest
julia> using Oceananigans

julia> THINCBVD()
THINCBVD{5, Float64}(order=7, steepnesses=(1.6, 2.5), moments=false)
```

References
==========
* Deng, X., Inaba, S., Xie, B., Shyue, K.-M., & Xiao, F. (2018). "High fidelity discontinuity-resolving reconstruction for
  compressible multiphase flows with moving interfaces." Journal of Computational Physics, 371, 945-966.
* Deng, X., Shimizu, Y., & Xiao, F. (2019). "A fifth-order shock capturing scheme with two-stage boundary variation
  diminishing algorithm." Journal of Computational Physics, 386, 323-349.
* Zalesak, S. T. (1979). "Fully multidimensional flux-corrected transport algorithms for fluids." Journal of Computational
  Physics, 31(3), 335-362.
"""
function THINCBVD(FT::DataType=Oceananigans.defaults.FloatType; order=7, steepnesses=(1.6, 2.5), moments=false,
                  time_discretization=ExplicitTimeDiscretization())
    weno = WENO(FT; order, time_discretization)
    N  = required_halo_size_x(weno) + 1
    β₀ = reduce(gcd, map(rationalize, steepnesses); init=0//1)
    thinc = map(steepnesses) do steepness
        n = Int(rationalize(steepness) // β₀)
        β = convert(FT, n * β₀)
        return Val(n), β, tanh(β), 1 / tanh(β), 1 / cosh(β)
    end
    return THINCBVD{N, FT, typeof(weno.time_discretization)}(weno, thinc, convert(FT, β₀), moments, nothing)
end

Adapt.adapt_structure(to, s::THINCBVD{N, FT, TD}) where {N, FT, TD} =
    THINCBVD{N, FT, TD}(Adapt.adapt(to, s.buffer_scheme), s.steepnesses, s.base_steepness, s.moments, Adapt.adapt(to, s.reconstructions))

@inline TimeSteppers.time_discretization(s::THINCBVD) = TimeSteppers.time_discretization(s.buffer_scheme)

Base.summary(s::THINCBVD{N, FT}) where {N, FT} = string("THINCBVD{$N, $FT}(order=", weno_order(s.buffer_scheme), ", steepnesses=",
                                                         map(t -> t[2], s.steepnesses), ", moments=", s.moments, ")")
Base.show(io::IO, s::THINCBVD) = print(io, summary(s))

# The tanh step of a cell of value c from min(c₋, c₊) to max(c₋, c₊): base, height, direction and the cell's place in it, ξ ∈ [-1, 1]
@inline function thinc_step(c₋, c, c₊)
    rising = c₋ < c₊
    low = ifelse(rising, c₋, c₊)
    Δ = ifelse(rising, c₊, c₋) - low
    return low, Δ, sign(c₊ - c₋), 2 * newton_div(BackendOptimizedDivision, c - low, Δ) - 1
end

# Eⁿ by repeated squaring, unrolled at compile time
@inline integer_power(E, ::Val{1}) = E
@inline integer_power(E, ::Val{n}) where n = (P = integer_power(E * E, Val(n ÷ 2)); ifelse(isodd(n), P * E, P))

# Face values (left, right) of the step whose cell mean is c, with B = exp(θβξ); T = tanh β, T⁻¹ = 1 / tanh β, K⁻¹ = 1 / cosh β
@inline function thinc_face_values((low, Δ, θ, ξ), B, (n, β, T, T⁻¹, K⁻¹))
    A = (B * K⁻¹ - 1) * T⁻¹
    return low + Δ * (1 + θ * A) / 2, low + Δ * (1 + θ * newton_div(BackendOptimizedDivision, T + A, 1 + A * T)) / 2
end

# Face values (left, right) of c₀ in c = (c₋₂, c₋₁, c₀, c₊₁, c₊₂), given WENO's values R₋, L₀ on either side of its left face and
# R₀, L₊ on either side of its right face. THINC replaces WENO in every cell lying between its neighbours.
@inline function bvd_face_values(c, R₋, L₀, R₀, L₊, scheme)
    @inline between(n) = (c[n+1] - c[n]) * (c[n] - c[n-1]) > 0
    steps = (thinc_step(c[1], c[2], c[3]), thinc_step(c[2], c[3], c[4]), thinc_step(c[3], c[4], c[5]))
    # exp(θβξ) = Eⁿ for every steepness β = nβ₀; a Float32 E keeps the face values within ~1e-6 of the jump Δ
    E = map(((low, Δ, θ, ξ),) -> convert(typeof(ξ), @fastmath exp(Float32(θ * scheme.base_steepness * ξ))), steps)

    @inline function thinc_candidate(steepness)
        B = map(e -> integer_power(e, first(steepness)), E)
        r₋ = ifelse(between(2), last(thinc_face_values(steps[1], B[1], steepness)), R₋)
        l₀, r₀ = thinc_face_values(steps[2], B[2], steepness)
        l₀ = ifelse(between(3), l₀, L₀)
        r₀ = ifelse(between(3), r₀, R₀)
        l₊ = ifelse(between(4), first(thinc_face_values(steps[3], B[3], steepness)), L₊)
        return abs(r₋ - l₀) + abs(r₀ - l₊), l₀, r₀
    end

    weno_candidate = (abs(R₋ - L₀) + abs(R₀ - L₊), L₀, R₀)
    _, L, R = foldl((a, b) -> ifelse(b[1] < a[1], b, a), map(thinc_candidate, scheme.steepnesses); init=weno_candidate)
    return L, R
end

#####
##### Stored reconstructions: the tracer with ghost values (an inactive cell takes the value of the nearest active column, continued
##### linearly below its bed and above a closed top), the two WENO edges of every cell and the flux limiter
#####

struct Reconstructions{C, X, Y, Z, W, F, M, P, L, E}
    ghost_filled :: C
    x :: X                       # (left, right) WENO edges of every cell, or nothing in a Flat direction
    y :: Y
    z :: Z
    wet_column :: W              # (i, j) of the column whose values each column takes, and the lowest active level of that column
    antidiffusive_fractions :: F # (R⁺, R⁻), the fractions of the antidiffusive fluxes that raise and that lower each cell
    moments :: M                 # the tracer's first moments (Mˣ, Mʸ, Mᶻ), or nothing
    profiles :: P                # (c⁻, c⁺, w₁, w₂, w₃, w₄, state) of the profile of each cell, or nothing
    switched_cells :: L          # (places of the listed cells, the list of cells to fit, reach, geometry band, face integrals of the listed
                                 # tanh profiles) over the moments' update range, or nothing
    extrema :: E                 # the range of the tracer's point values at the first time step, empty before
end

Adapt.adapt_structure(to, r::Reconstructions) = Reconstructions(Adapt.adapt(to, r.ghost_filled), Adapt.adapt(to, r.x), Adapt.adapt(to, r.y),
                                                                Adapt.adapt(to, r.z), Adapt.adapt(to, r.wet_column),
                                                                Adapt.adapt(to, r.antidiffusive_fractions), Adapt.adapt(to, r.moments),
                                                                Adapt.adapt(to, r.profiles), Adapt.adapt(to, r.switched_cells), nothing)

@inline Base.getindex(r::Reconstructions, i, j, k) = @inbounds r.ghost_filled[i, j, k]

function Reconstructions(grid, radius)
    edges(d) = topology(grid, d) === Flat ? nothing : (CenterField(grid), CenterField(grid))
    R⁺, R⁻ = CenterField(grid), CenterField(grid)
    fill!(R⁺, 1)
    fill!(R⁻, 1)
    FT = eltype(grid)
    empty_range = Ref((convert(FT, Inf), convert(FT, -Inf)))
    return Reconstructions(CenterField(grid), edges(1), edges(2), edges(3), wet_columns(grid, radius), (R⁺, R⁻), nothing, nothing, nothing,
                           empty_range)
end

function wet_columns(grid, radius)
    iˢ, jˢ, kˢ = ntuple(_ -> Field{Center, Center, Nothing}(grid, Int32), 3)
    Nx, Ny = size(grid)
    Hx, Hy = halo_size(grid)
    launch!(architecture(grid), grid, KernelParameters((Nx + 2Hx, Ny + 2Hy), (-Hx, -Hy)), _compute_wet_columns!, iˢ, jˢ, kˢ, grid, radius)
    return iˢ, jˢ, kˢ
end

# The nearest column with an active top cell within `radius` of each column, or the column itself, and its lowest active level
@kernel function _compute_wet_columns!(iˢ, jˢ, kˢ, grid, radius)
    i, j = @index(Global, NTuple)
    Nx, Ny, Nz = size(grid)
    Hx, Hy = halo_size(grid)
    i′, j′, d² = i, j, typemax(Int)
    for di in -radius:radius, dj in -radius:radius
        iⁿ = clamp(i + di, 1 - Hx, Nx + Hx)
        jⁿ = clamp(j + dj, 1 - Hy, Ny + Hy)
        dⁿ = (iⁿ - i)^2 + (jⁿ - j)^2
        nearer = !inactive_cell(iⁿ, jⁿ, Nz, grid) & (dⁿ < d²)
        i′ = ifelse(nearer, iⁿ, i′)
        j′ = ifelse(nearer, jⁿ, j′)
        d² = ifelse(nearer, dⁿ, d²)
    end
    kᵇ = Nz
    for k in Nz:-1:1
        kᵇ = ifelse(inactive_cell(i′, j′, k, grid), kᵇ, k)
    end
    @inbounds iˢ[i, j, 1] = i′
    @inbounds jˢ[i, j, 1] = j′
    @inbounds kˢ[i, j, 1] = kᵇ
end

@inline function ghost_value(i, j, k, grid, c, (iˢ, jˢ, kˢ))
    i′, j′, kᵇ = @inbounds iˢ[i, j, 1], jˢ[i, j, 1], kˢ[i, j, 1]
    k′ = ifelse(inactive_cell(i′, j′, k, grid), clamp(k, kᵇ, size(grid, 3)), k)
    n  = k - k′ # levels past the nearest active one: < 0 under the bed, > 0 above a closed top
    cᵃ = @inbounds c[i′, j′, k′]
    return cᵃ + abs(n) * (cᵃ - @inbounds c[i′, j′, clamp(k′ - sign(n), kᵇ, size(grid, 3))])
end

@kernel function _fill_ghost_values!(ĉ, grid, c, wet_column)
    i, j, k = @index(Global, NTuple)
    @inbounds ĉ[i, j, k] = ghost_value(i, j, k, grid, c, wet_column)
end

@kernel function _compute_weno_edges!(edges, grid, weno::WENO{N}, ĉ, stencil, shift) where N
    i, j, k = @index(Global, NTuple)
    S  = stencil(i + shift[1], j + shift[2], k + shift[3], grid, weno, LeftBias, ĉ)
    δ  = weno_differences(weno, S)
    ψ₀ = weno_anchor(weno, S)
    β  = beta_loop(weno, δ)
    τ  = global_smoothness_indicator(Val(N), β)
    αᴸ = zweno_alpha_loop(weno, reverse(β), τ)
    αᴿ = zweno_alpha_loop(weno, β, τ)
    L  = weno_reconstruction(weno, ψ₀, map(-, reverse(δ)), αᴸ .* weight_normalization(weno, αᴸ))
    R  = weno_reconstruction(weno, ψ₀, δ, αᴿ .* weight_normalization(weno, αᴿ))
    @inbounds edges[1][i, j, k] = L
    @inbounds edges[2][i, j, k] = R
end

# Half the largest minmod slope of cell i, j, k along d, one-sided next to a cell without data
@inline function half_slope(i, j, k, grid, ::Val{d}, c) where d
    i⁻, j⁻, k⁻ = d == 1 ? (i - 1, j, k) : d == 2 ? (i, j - 1, k) : (i, j, k - 1)
    i⁺, j⁺, k⁺ = d == 1 ? (i + 1, j, k) : d == 2 ? (i, j + 1, k) : (i, j, k + 1)
    cᵢ = @inbounds c[i, j, k]
    Δ⁻ = cᵢ - @inbounds c[i⁻, j⁻, k⁻]
    Δ⁺ = @inbounds(c[i⁺, j⁺, k⁺]) - cᵢ
    l  = !inactive_node(i⁻, j⁻, k⁻, grid, Center(), Center(), Center())
    r  = !inactive_node(i⁺, j⁺, k⁺, grid, Center(), Center(), Center())
    minmod = ifelse(Δ⁻ * Δ⁺ > 0, min(abs(Δ⁻), abs(Δ⁺)), zero(cᵢ))
    return ifelse(l & r, minmod, ifelse(l, abs(Δ⁻), ifelse(r, abs(Δ⁺), zero(cᵢ)))) / 2
end

@inline half_slope(i, j, k, grid::XFlatGrid, ::Val{1}, c) = zero(grid)
@inline half_slope(i, j, k, grid::YFlatGrid, ::Val{2}, c) = zero(grid)
@inline half_slope(i, j, k, grid::ZFlatGrid, ::Val{3}, c) = zero(grid)

# The cell value widened by half its largest minmod slope, below for s = -1 and above for s = +1
@inline function point_value(i, j, k, grid, c, s)
    σ = max(half_slope(i, j, k, grid, Val(1), c), half_slope(i, j, k, grid, Val(2), c), half_slope(i, j, k, grid, Val(3), c))
    return @inbounds c[i, j, k] + s * σ
end

"""
$(TYPEDSIGNATURES)

Fill the tracer with ghost values, the WENO edges of its cells and the antidiffusive fractions that a materialized THINC-BVD
`scheme` reads, from the tracer `c` and the velocities and next stage time step of `model`, and update the first moments of `c`
that `scheme` carries: fit a tanh profile to those of the cells between two water masses, reset the others and fit a least-variance
ramp to those whose linear profile leaves their bounds. The first update with a finite time step sets the range of `c`.
"""
function update_tracer_advection!(scheme::THINCBVD, model, c)
    grid = model.grid
    r, weno = scheme.reconstructions, scheme.buffer_scheme
    ĉ = r.ghost_filled
    arch = architecture(grid)
    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)
    launch!(arch, grid, KernelParameters((Nx + 2Hx, Ny + 2Hy, Nz + 2Hz), (-Hx, -Hy, -Hz)), _fill_ghost_values!, ĉ, grid, c, r.wet_column)
    isnothing(r.x) || launch!(arch, grid, KernelParameters((Nx + 4, Ny, Nz), (-2, 0, 0)), _compute_weno_edges!, r.x, grid, weno, ĉ, weno_stencil_x, (1, 0, 0))
    isnothing(r.y) || launch!(arch, grid, KernelParameters((Nx, Ny + 4, Nz), (0, -2, 0)), _compute_weno_edges!, r.y, grid, weno, ĉ, weno_stencil_y, (0, 1, 0))
    isnothing(r.z) || launch!(arch, grid, KernelParameters((Nx, Ny, Nz + 4), (0, 0, -2)), _compute_weno_edges!, r.z, grid, weno, ĉ, weno_stencil_z, (0, 0, 1))

    if !isnothing(r.moments)
        positions, list, reach, geometry, face_integrals = r.switched_cells
        state = last(r.profiles)
        update_range = KernelParameters(size(positions), .-reach)
        listing = (size(positions), reach)
        launch!(arch, grid, update_range, _update_moments!, r.moments, r.profiles, grid, ĉ, scheme, model.velocities.w)
        ramps = list_cells!(list, positions, state, ramp_fit_bit, reach, arch, grid, update_range)
        ramps > 0 && launch!(arch, grid, view(first(list.values), 1:ramps), _fit_least_variance_ramps!, listing, r.moments, r.profiles, grid, ĉ, geometry)
        tanhs = list_cells!(list, positions, state, tanh_fit_bit, reach, arch, grid, update_range)
        grow!(face_integrals, tanhs)
        tanhs > 0 && launch!(arch, grid, view(first(list.values), 1:tanhs), _fit_tanh_profiles!, listing, r.moments, r.profiles, grid, ĉ, geometry,
                             (positions, face_integrals))
    end

    Δτ = adaptive_advection_timestep(model.timestepper, model.clock)
    isfinite(Δτ) || return nothing

    if first(r.extrema[]) > last(r.extrema[])
        lower = KernelFunctionOperation{Center, Center, Center}(point_value, grid, c, -1)
        upper = KernelFunctionOperation{Center, Center, Center}(point_value, grid, c, +1)
        r.extrema[] = (minimum(lower), maximum(upper))
    end

    R⁺, R⁻ = r.antidiffusive_fractions
    launch!(arch, grid, :xyz, _compute_antidiffusive_fractions!, R⁺, R⁻, grid, scheme, model.velocities, r, convert(eltype(grid), Δτ), r.extrema[])
    fill_halo_regions!((R⁺, R⁻))
    return nothing
end

for (interpolate, d) in ((:biased_interpolate_xᶠᵃᵃ, 1), (:biased_interpolate_yᵃᶠᵃ, 2), (:biased_interpolate_zᵃᵃᶠ, 3))
    index(s) = d == 1 ? (:(i + o + $s), :j, :k) : d == 2 ? (:i, :(j + o + $s), :k) : (:i, :j, :(k + o + $s))
    edges = (:x, :y, :z)[d]

    @eval @inline function $interpolate(i, j, k, grid, scheme::THINCBVD, bias, r::Reconstructions)
        L, R = r.$edges
        o = ifelse(bias == LeftBias, -1, 0) # the upwind cell is i + o
        c = @inbounds ($((:(r.ghost_filled[$(index(s)...)]) for s in -2:2)...),)
        Lₘ, Rₘ = @inbounds bvd_face_values(c, R[$(index(-1)...)], L[$(index(0)...)], R[$(index(0)...)], L[$(index(1)...)], scheme)
        return ifelse(bias == LeftBias, Rₘ, Lₘ)
    end
end

#####
##### Flux limiter (Zalesak, 1979): a face carries the upwind flux q cᵘ plus θ A, the fraction θ of the antidiffusive flux A that
##### keeps the stage update of both cells of the face within their bounds
#####

@inline positive_part(x) = ifelse(signbit(x), zero(x), x)

for (flux, interpolate, area, FlatGrid, d) in ((:_advective_tracer_flux_x, :biased_interpolate_xᶠᵃᵃ, :Axᶠᶜᶜ, :XFlatGrid, 1),
                                               (:_advective_tracer_flux_y, :biased_interpolate_yᵃᶠᵃ, :Ayᶜᶠᶜ, :YFlatGrid, 2),
                                               (:_advective_tracer_flux_z, :biased_interpolate_zᵃᵃᶠ, :Azᶜᶜᶠ, :ZFlatGrid, 3))
    minus_side = d == 1 ? (:(i - 1), :j, :k) : d == 2 ? (:i, :(j - 1), :k) : (:i, :j, :(k - 1))
    velocity_scale = d == 3 ? :(explicit_velocity_scaleᶜᶜᶠ(i, j, k, grid, scheme, TimeSteppers.time_discretization(scheme), u)) : 1

    @eval begin
        # explicit transport q, upwind value cᵘ and antidiffusive flux A = q (ψ - cᵘ) through the face at i, j, k
        @inline function face_flux_parts(i, j, k, grid, scheme, u, r, ::Val{$d})
            ũ  = @inbounds u[i, j, k]
            q  = $area(i, j, k, grid) * ũ * $velocity_scale
            ψ  = two_water_masses_face_value($interpolate(i, j, k, grid, scheme, bias(ũ), r), upwind_face_profile(i, j, k, r.profiles, r, ũ, Val($d)))
            cᵘ = ifelse(ũ > 0, r[$(minus_side...)], r[i, j, k])
            return q, cᵘ, q * (ψ - cᵘ)
        end

        @inline face_flux_parts(i, j, k, grid::$FlatGrid, scheme, u, r, ::Val{$d}) = (zero(grid), zero(grid), zero(grid))

        # explicit transport q, flux q cᵘ + θ A and the fraction θ of the antidiffusive flux that the face carries
        @inline function limited_face_flux(i, j, k, grid, scheme, u, r, ::Val{$d})
            q, cᵘ, A = face_flux_parts(i, j, k, grid, scheme, u, r, Val($d))
            R⁺, R⁻ = r.antidiffusive_fractions
            R₁, R₂ = @inbounds ifelse(A > 0, (R⁺[i, j, k], R⁻[$(minus_side...)]), (R⁺[$(minus_side...)], R⁻[i, j, k]))
            θ = ifelse(R₁ < R₂, R₁, R₂)
            return q, q * cᵘ + θ * A, θ
        end

        @inline limited_face_flux(i, j, k, grid::$FlatGrid, scheme, u, r, ::Val{$d}) = (zero(grid), zero(grid), zero(grid))

        @inline $flux(i, j, k, grid, scheme::THINCBVD, u, r::Reconstructions) = limited_face_flux(i, j, k, grid, scheme, u, r, Val($d))[2]
    end
end

# R± = min(1, Q± / P±): P± is the stage change that the antidiffusive fluxes into or out of a cell would make, Q± the room between the
# upwind update cᵗᵈ and the bounds, the range (m, M) widened by the cell and the values that flow into it. Vᵗᵈ is the cell volume, in
# units of V, after the explicit transport, which differs from 1 by the implicit vertical transport
@kernel function _compute_antidiffusive_fractions!(R⁺, R⁻, grid, scheme, U, r, Δτ, (m, M))
    i, j, k = @index(Global, NTuple)
    c = r[i, j, k]

    qʷ, cʷ, Aʷ = face_flux_parts(i,     j,     k,     grid, scheme, U.u, r, Val(1))
    qᵉ, cᵉ, Aᵉ = face_flux_parts(i + 1, j,     k,     grid, scheme, U.u, r, Val(1))
    qˢ, cˢ, Aˢ = face_flux_parts(i,     j,     k,     grid, scheme, U.v, r, Val(2))
    qⁿ, cⁿ, Aⁿ = face_flux_parts(i,     j + 1, k,     grid, scheme, U.v, r, Val(2))
    qᵇ, cᵇ, Aᵇ = face_flux_parts(i,     j,     k,     grid, scheme, U.w, r, Val(3))
    qᵗ, cᵗ, Aᵗ = face_flux_parts(i,     j,     k + 1, grid, scheme, U.w, r, Val(3))

    κ   = Δτ * V⁻¹ᶜᶜᶜ(i, j, k, grid)
    Vᵗᵈ = 1 + κ * (qʷ - qᵉ + qˢ - qⁿ + qᵇ - qᵗ)
    cᵗᵈ = (c + κ * (qʷ * cʷ - qᵉ * cᵉ + qˢ * cˢ - qⁿ * cⁿ + qᵇ * cᵇ - qᵗ * cᵗ)) / Vᵗᵈ
    P⁺  = κ * (positive_part(Aʷ) + positive_part(-Aᵉ) + positive_part(Aˢ) + positive_part(-Aⁿ) + positive_part(Aᵇ) + positive_part(-Aᵗ))
    P⁻  = κ * (positive_part(-Aʷ) + positive_part(Aᵉ) + positive_part(-Aˢ) + positive_part(Aⁿ) + positive_part(-Aᵇ) + positive_part(Aᵗ))

    inflows = (ifelse(qʷ > 0, cʷ, c), ifelse(qᵉ < 0, cᵉ, c), ifelse(qˢ > 0, cˢ, c), ifelse(qⁿ < 0, cⁿ, c), ifelse(qᵇ > 0, cᵇ, c), ifelse(qᵗ < 0, cᵗ, c))
    Q⁺ = Vᵗᵈ * positive_part(max(M, c, inflows...) - cᵗᵈ)
    Q⁻ = Vᵗᵈ * positive_part(cᵗᵈ - min(m, c, inflows...))

    @inbounds R⁺[i, j, k] = ifelse(P⁺ ≤ Q⁺, one(c), Q⁺ / P⁺)
    @inbounds R⁻[i, j, k] = ifelse(P⁻ ≤ Q⁻, one(c), Q⁻ / P⁻)
end
