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
    buffer_scheme :: W   # WENO: the smooth reconstruction, and the scheme beside boundaries
    steepnesses :: S     # (Val(n), β, tanh β, 1 / tanh β, 1 / cosh β) of each THINC steepness β = nβ₀
    base_steepness :: FT # β₀, the largest steepness of which every β is an integer multiple
    reconstructions :: R # once materialized, the advected tracer with ghost values, the WENO edges of its cells and the flux limiter
end

"""
$(MethodSignatures(; defaults = true))

Construct a WENO-THINC-BVD tracer advection scheme. A cell whose value lies between its neighbours' reconstructs its face
values with `WENO(FT; order)` or with THINC, a hyperbolic tangent step of one of `steepnesses` between the neighbours'
values, and takes the reconstruction with the smallest total jump across its two faces (boundary variation). The other cells
use WENO. On grids with an immersed bottom, a cell under the bed takes the value of its column's bottom cell, and a land
column those of the nearest wet column. Each face flux is the upwind flux plus the largest fraction of the remainder (the
antidiffusive flux) that keeps the stage update of both cells of the face within the range of the tracer at the first time
step, widened by each cell and the values flowing into it (Zalesak, 1979). A model materializes the scheme with up to nine
fields per tracer: the tracer with these ghost values, the two WENO edges of every cell and the fractions of the antidiffusive
fluxes that raise and that lower every cell, refreshed by `update_tracer_advection!` before the tendencies are
computed. The scheme needs halos one cell wider than `WENO(FT; order)`. All steepnesses share one exponential per cell,
cheapest when they are small integer multiples of a common value, as 1.6 and 2.5 are of 1/10. With `steepnesses=()` every
cell takes WENO's face values, read from the tracer with ghost values. With `time_discretization=AdaptiveVerticallyImplicitDiscretization()`
the vertical transport beyond its `cfl` is implicit and first-order upwind: the range then bounds the tracer after the explicit
transport, which the implicit solve keeps within it.

```jldoctest
julia> using Oceananigans

julia> THINCBVD()
THINCBVD{5, Float64}(order=7, steepnesses=(1.6, 2.5))
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
function THINCBVD(FT::DataType=Oceananigans.defaults.FloatType; order=7, steepnesses=(1.6, 2.5), time_discretization=ExplicitTimeDiscretization())
    weno = WENO(FT; order, time_discretization)
    N  = required_halo_size_x(weno) + 1
    β₀ = reduce(gcd, map(rationalize, steepnesses); init=0//1)
    thinc = map(steepnesses) do steepness
        n = Int(rationalize(steepness) // β₀)
        β = convert(FT, n * β₀)
        return Val(n), β, tanh(β), 1 / tanh(β), 1 / cosh(β)
    end
    return THINCBVD{N, FT, typeof(weno.time_discretization), typeof(weno), typeof(thinc), Nothing}(weno, thinc, convert(FT, β₀), nothing)
end

function Adapt.adapt_structure(to, s::THINCBVD{N, FT, TD}) where {N, FT, TD}
    weno = Adapt.adapt(to, s.buffer_scheme)
    reconstructions = Adapt.adapt(to, s.reconstructions)
    return THINCBVD{N, FT, TD, typeof(weno), typeof(s.steepnesses), typeof(reconstructions)}(weno, s.steepnesses, s.base_steepness, reconstructions)
end

@inline TimeSteppers.time_discretization(s::THINCBVD) = TimeSteppers.time_discretization(s.buffer_scheme)

Base.summary(s::THINCBVD{N, FT}) where {N, FT} = string("THINCBVD{$N, $FT}(order=", weno_order(s.buffer_scheme), ", steepnesses=", map(t -> t[2], s.steepnesses), ")")
Base.show(io::IO, s::THINCBVD) = print(io, summary(s))

# The tanh step of a cell of value c from min(c₋, c₊) to max(c₋, c₊): base, height, direction and the cell's place in it, ξ ∈ [-1, 1]
@inline function thinc_step(c₋, c, c₊)
    low = min(c₋, c₊)
    Δ = max(c₋, c₊) - low
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
    between(n) = (c[n+1] - c[n]) * (c[n] - c[n-1]) > 0
    steps = (thinc_step(c[1], c[2], c[3]), thinc_step(c[2], c[3], c[4]), thinc_step(c[3], c[4], c[5]))
    # exp(θβξ) = Eⁿ for every steepness β = nβ₀; a Float32 E keeps the face values within ~1e-6 of the jump Δ
    E = map(((low, Δ, θ, ξ),) -> convert(typeof(ξ), @fastmath exp(Float32(θ * scheme.base_steepness * ξ))), steps)

    function thinc_candidate(steepness)
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
##### Stored reconstructions: the tracer with ghost values (a cell under the bed takes the value of its column's bottom cell, a land
##### column those of the nearest wet column), the two WENO edges of every cell and the flux limiter
#####

const BottomImmersedBoundaryGrid = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Union{GridFittedBottom, PartialCellBottom, CutCellBottom}}

struct Reconstructions{C, X, Y, Z, W, F, E}
    ghost_filled :: C
    x :: X                       # (left, right) WENO edges of every cell, or nothing in a Flat direction
    y :: Y
    z :: Z
    wet_column :: W              # (i, j) of the column whose values each column takes, or nothing without an immersed bottom
    antidiffusive_fractions :: F # (R⁺, R⁻), the fractions of the antidiffusive fluxes that raise and that lower each cell
    extrema :: E                 # the range of the tracer at the first time step, empty before
end

Adapt.adapt_structure(to, r::Reconstructions) = Reconstructions(Adapt.adapt(to, r.ghost_filled), Adapt.adapt(to, r.x), Adapt.adapt(to, r.y),
                                                                Adapt.adapt(to, r.z), Adapt.adapt(to, r.wet_column),
                                                                Adapt.adapt(to, r.antidiffusive_fractions), nothing)

@inline Base.getindex(r::Reconstructions, i, j, k) = @inbounds r.ghost_filled[i, j, k]

function Reconstructions(grid, radius)
    edges(d) = topology(grid, d) === Flat ? nothing : (CenterField(grid), CenterField(grid))
    R⁺, R⁻ = CenterField(grid), CenterField(grid)
    fill!(R⁺, 1)
    fill!(R⁻, 1)
    FT = eltype(grid)
    empty_range = Ref((convert(FT, Inf), convert(FT, -Inf)))
    return Reconstructions(CenterField(grid), edges(1), edges(2), edges(3), wet_columns(grid, radius), (R⁺, R⁻), empty_range)
end

wet_columns(grid, radius) = nothing

function wet_columns(ibg::BottomImmersedBoundaryGrid, radius)
    iˢ = Field{Center, Center, Nothing}(ibg, Int32)
    jˢ = Field{Center, Center, Nothing}(ibg, Int32)
    Nx, Ny = size(ibg)
    Hx, Hy = halo_size(ibg)
    launch!(architecture(ibg), ibg, KernelParameters((Nx + 2Hx, Ny + 2Hy), (-Hx, -Hy)), _compute_wet_columns!, iˢ, jˢ, ibg, radius)
    return iˢ, jˢ
end

# The nearest column with water within `radius` of each column, or the column itself
@kernel function _compute_wet_columns!(iˢ, jˢ, ibg, radius)
    i, j = @index(Global, NTuple)
    grid = ibg.underlying_grid
    i′, j′, d² = i, j, typemax(Int)
    for di in -radius:radius, dj in -radius:radius
        iⁿ = clamp(i + di, 1 - grid.Hx, grid.Nx + grid.Hx)
        jⁿ = clamp(j + dj, 1 - grid.Hy, grid.Ny + grid.Hy)
        dⁿ = (iⁿ - i)^2 + (jⁿ - j)^2
        nearer = (bottom_cell(iⁿ, jⁿ, grid, ibg.immersed_boundary) ≤ grid.Nz) & (dⁿ < d²)
        i′ = ifelse(nearer, iⁿ, i′)
        j′ = ifelse(nearer, jⁿ, j′)
        d² = ifelse(nearer, dⁿ, d²)
    end
    @inbounds iˢ[i, j, 1] = i′
    @inbounds jˢ[i, j, 1] = j′
end

@inline ghost_value(i, j, k, grid, c, args...) = @inbounds c[i, j, k]

@inline function ghost_value(i, j, k, ibg::BottomImmersedBoundaryGrid, c, (iˢ, jˢ))
    i′ = @inbounds iˢ[i, j, 1]
    j′ = @inbounds jˢ[i, j, 1]
    return @inbounds c[i′, j′, max(k, bottom_cell(i′, j′, ibg.underlying_grid, ibg.immersed_boundary))]
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

"""
$(TYPEDSIGNATURES)

Fill the tracer with ghost values, the WENO edges of its cells and the antidiffusive fractions that a materialized THINC-BVD
`scheme` reads, from the tracer `c` and the velocities and next stage time step of `model`. The first update with a finite
time step sets the range of `c`.
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

    Δτ = adaptive_advection_timestep(model.timestepper, model.clock)
    isfinite(Δτ) || return nothing

    if first(r.extrema[]) > last(r.extrema[])
        r.extrema[] = (minimum(c), maximum(c))
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

for interpolate in (:_biased_interpolate_xᶠᵃᵃ, :_biased_interpolate_yᵃᶠᵃ, :_biased_interpolate_zᵃᵃᶠ)
    @eval @inline $interpolate(i, j, k, ibg::BottomImmersedBoundaryGrid, scheme::THINCBVD, bias, r::Reconstructions) = $interpolate(i, j, k, ibg.underlying_grid, scheme, bias, r)
end

#####
##### Flux limiter (Zalesak, 1979): a face carries the upwind flux q cᵘ plus θ A, the fraction θ of the antidiffusive flux A that
##### keeps the stage update of both cells of the face within their bounds
#####

@inline positive_part(x) = ifelse(signbit(x), zero(x), x)

for (flux, interpolate, area, FlatGrid, d) in ((:_advective_tracer_flux_x, :_biased_interpolate_xᶠᵃᵃ, :Axᶠᶜᶜ, :XFlatGrid, 1),
                                               (:_advective_tracer_flux_y, :_biased_interpolate_yᵃᶠᵃ, :Ayᶜᶠᶜ, :YFlatGrid, 2),
                                               (:_advective_tracer_flux_z, :_biased_interpolate_zᵃᵃᶠ, :Azᶜᶜᶠ, :ZFlatGrid, 3))
    minus_side = d == 1 ? (:(i - 1), :j, :k) : d == 2 ? (:i, :(j - 1), :k) : (:i, :j, :(k - 1))
    velocity_scale = d == 3 ? :(explicit_velocity_scaleᶜᶜᶠ(i, j, k, grid, scheme, TimeSteppers.time_discretization(scheme), u)) : 1

    @eval begin
        # explicit transport q, upwind value cᵘ and antidiffusive flux A = q (ψ - cᵘ) through the face at i, j, k
        @inline function face_flux_parts(i, j, k, grid, scheme, u, r, ::Val{$d})
            ũ  = @inbounds u[i, j, k]
            q  = $area(i, j, k, grid) * ũ * $velocity_scale
            ψ  = $interpolate(i, j, k, grid, scheme, bias(ũ), r)
            cᵘ = ifelse(ũ > 0, r[$(minus_side...)], r[i, j, k])
            return q, cᵘ, q * (ψ - cᵘ)
        end

        @inline face_flux_parts(i, j, k, grid::$FlatGrid, scheme, u, r, ::Val{$d}) = (zero(grid), zero(grid), zero(grid))

        @inline function $flux(i, j, k, grid, scheme::THINCBVD, u, r::Reconstructions)
            q, cᵘ, A = face_flux_parts(i, j, k, grid, scheme, u, r, Val($d))
            R⁺, R⁻ = r.antidiffusive_fractions
            θ = @inbounds ifelse(A > 0, min(R⁺[i, j, k], R⁻[$(minus_side...)]), min(R⁺[$(minus_side...)], R⁻[i, j, k]))
            return q * cᵘ + θ * A
        end
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
