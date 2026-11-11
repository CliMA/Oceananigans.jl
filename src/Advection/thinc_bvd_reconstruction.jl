#####
##### WENO-THINC-BVD: per cell, WENO or THINC face values, whichever jumps less across the cell's faces
#####

using Oceananigans.Utils: newton_div, BackendOptimizedDivision

struct THINCBVD{N, FT, TD, W, S} <: AbstractUpwindBiasedAdvectionScheme{N, FT, TD}
    buffer_scheme :: W   # WENO: the smooth reconstruction, and the scheme beside boundaries
    steepnesses :: S     # (Val(n), β, tanh β, 1 / tanh β, 1 / cosh β) of each THINC steepness β = nβ₀
    base_steepness :: FT # β₀, the largest steepness of which every β is an integer multiple
end

"""
$(MethodSignatures(; defaults = true))

Construct a WENO-THINC-BVD tracer advection scheme. A cell whose value lies between its neighbours' reconstructs its face
values with `WENO(FT; order)` or with THINC, a hyperbolic tangent step of one of `steepnesses` between the neighbours'
values, and takes the reconstruction with the smallest total jump across its two faces (boundary variation). The other cells
use WENO. On grids with an immersed bottom, a cell under the bed takes the value of its column's bottom cell. The scheme
needs halos one cell wider than `WENO(FT; order)`. All steepnesses share one exponential per cell, cheapest when they are
small integer multiples of a common value, as 1.6 and 2.5 are of 1/10.

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
"""
function THINCBVD(FT::DataType=Oceananigans.defaults.FloatType; order=7, steepnesses=(1.6, 2.5))
    weno = WENO(FT; order)
    N  = required_halo_size_x(weno) + 1
    β₀ = reduce(gcd, map(rationalize, steepnesses))
    thinc = map(steepnesses) do steepness
        n = Int(rationalize(steepness) // β₀)
        β = convert(FT, n * β₀)
        return Val(n), β, tanh(β), 1 / tanh(β), 1 / cosh(β)
    end
    return THINCBVD{N, FT, typeof(weno.time_discretization), typeof(weno), typeof(thinc)}(weno, thinc, convert(FT, β₀))
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
    E = map(((low, Δ, θ, ξ),) -> exp(θ * scheme.base_steepness * ξ), steps) # exp(θβξ) = Eⁿ for every steepness β = nβ₀

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

for (interpolate, d) in ((:biased_interpolate_xᶠᵃᵃ, 1), (:biased_interpolate_yᵃᶠᵃ, 2), (:biased_interpolate_zᵃᵃᶠ, 3))
    index(s) = d == 1 ? (:(i + o + $s), :j, :k) : d == 2 ? (:i, :(j + o + $s), :k) : (:i, :j, :(k + o + $s))

    @eval @inline function $interpolate(i, j, k, grid, scheme::THINCBVD, bias, ψ, args...)
        weno = scheme.buffer_scheme
        o = ifelse(bias == LeftBias, -1, 0) # the upwind cell is i + o
        R₋ = $interpolate($(index(0)...), grid, weno, LeftBias,  ψ, args...)
        L₀ = $interpolate($(index(0)...), grid, weno, RightBias, ψ, args...)
        R₀ = $interpolate($(index(1)...), grid, weno, LeftBias,  ψ, args...)
        L₊ = $interpolate($(index(1)...), grid, weno, RightBias, ψ, args...)
        c = ($((:(getvalue(ψ, $(index(s)...), grid, args...)) for s in -2:2)...),)
        L, R = bvd_face_values(c, R₋, L₀, R₀, L₊, scheme)
        return ifelse(bias == LeftBias, R, L)
    end
end

#####
##### Immersed bottoms: a cell under the bed takes the value of its column's bottom cell
#####

const BottomImmersedBoundaryGrid = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Union{GridFittedBottom, PartialCellBottom}}

@inline bed_value(i, j, k, grid, ibg, c) = @inbounds c[i, j, max(k, bottom_cell(i, j, grid, ibg.immersed_boundary))]
@inline land_column(i, j, ibg) = bottom_cell(i, j, ibg.underlying_grid, ibg.immersed_boundary) > ibg.Nz

# A stencil reaching a column of land falls back on WENO as on any immersed grid
for (interpolate, d) in ((:_biased_interpolate_xᶠᵃᵃ, 1), (:_biased_interpolate_yᵃᶠᵃ, 2), (:_biased_interpolate_zᵃᵃᶠ, 3))
    near_land = d == 1 ? :(any(ntuple(n -> land_column(i - N - 1 + n, j, ibg), Val(2N)))) :
                d == 2 ? :(any(ntuple(n -> land_column(i, j - N - 1 + n, ibg), Val(2N)))) : false

    @eval @inline function $interpolate(i, j, k, ibg::BottomImmersedBoundaryGrid, scheme::THINCBVD{N}, bias, c::AbstractArray) where N
        ψ̂ = $interpolate(i, j, k, ibg.underlying_grid, scheme, bias, bed_value, ibg, c)
        return $(near_land == false ? :ψ̂ : :(ifelse($near_land, $interpolate(i, j, k, ibg, scheme.buffer_scheme, bias, c), ψ̂)))
    end
end
