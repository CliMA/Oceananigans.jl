using Oceananigans.Grids: inactive_node

#####
##### Stencils of 2N nodes along one direction, with ghosts beyond the data
#####

@inline unit_vector(::Val{1}) = (1, 0, 0)
@inline unit_vector(::Val{2}) = (0, 1, 0)
@inline unit_vector(::Val{3}) = (0, 0, 1)

@inline stencil_value(i, j, k, grid, ψ, args...)           = @inbounds ψ[i, j, k]
@inline stencil_value(i, j, k, grid, ψ::Callable, args...) = ψ(i, j, k, grid, args...)

# The node o places from (i, j, k) along `dir` has data when its cell is active, or at a Face location the cell towards (i, j, k)
@inline function has_data(i, j, k, grid, dir::Val{d}, ℓ, o) where d
    di, dj, dk = unit_vector(dir)
    c = o - ((ℓ[d] isa Face) & (o ≥ 0))
    return !inactive_node(i + c * di, j + c * dj, k + c * dk, grid, Base.setindex(ℓ, Center(), d)...)
end

# Every node along a Periodic direction has data when the grid is not immersed
@inline has_data(i, j, k, grid::AbstractUnderlyingGrid{<:Any, Periodic, <:Any, <:Any}, ::Val{1}, ℓ, o) = true
@inline has_data(i, j, k, grid::AbstractUnderlyingGrid{<:Any, <:Any, Periodic, <:Any}, ::Val{2}, ℓ, o) = true
@inline has_data(i, j, k, grid::AbstractUnderlyingGrid{<:Any, <:Any, <:Any, Periodic}, ::Val{3}, ℓ, o) = true

"""
    MinmodExtrapolation()

Ghosts that continue the data with the minmod of its last two differences.
"""
struct MinmodExtrapolation end

"""
    VanLeerExtrapolation()

Ghosts that continue the data with the van Leer mean `2ab / (a + b)` of its last two differences `a` and `b`.
"""
struct VanLeerExtrapolation end

"""
    LinearExtrapolation()

Ghosts that continue the data with its last difference.
"""
struct LinearExtrapolation end

"""
    ContinuedExtrapolation()

Ghosts that continue the data with its last value.
"""
struct ContinuedExtrapolation end

"""
    BottomExtrapolation()

Ghosts under an immersed bottom take the value of the bottom cell of their own column; other ghosts continue the data with its last
value.
"""
struct BottomExtrapolation end

Base.summary(::MinmodExtrapolation)    = "MinmodExtrapolation()"
Base.summary(::VanLeerExtrapolation)   = "VanLeerExtrapolation()"
Base.summary(::LinearExtrapolation)    = "LinearExtrapolation()"
Base.summary(::ContinuedExtrapolation) = "ContinuedExtrapolation()"
Base.summary(::BottomExtrapolation)    = "BottomExtrapolation()"

@inline limited_slope(::MinmodExtrapolation,    a, b) = ifelse(a * b > 0, ifelse(abs(a) < abs(b), a, b), zero(a))
@inline limited_slope(::VanLeerExtrapolation,   a, b) = ifelse(a * b > 0, 2a * b / (a + b), zero(a))
@inline limited_slope(::LinearExtrapolation,    a, b) = a
@inline limited_slope(::ContinuedExtrapolation, a, b) = zero(a)
@inline limited_slope(::BottomExtrapolation,    a, b) = zero(a)

@inline ghost_value(i, j, k, grid, extrapolation, g, ℓ, ψ, args...) = g

@inline function ghost_value(i, j, k, grid::AGFBIBG, ::BottomExtrapolation, g, ::Tuple{Center, Center, Center}, ψ, args...)
    kᵇ = bottom_cell(i, j, grid.underlying_grid, grid.immersed_boundary)
    Nz = size(grid, 3)
    ψᵇ = stencil_value(i, j, min(kᵇ, Nz), grid, ψ, args...)
    return ifelse((k < kᵇ) & (kᵇ ≤ Nz), ψᵇ, g)
end

# The data at the three nodes with data nearest the boundary, outermost first, and the number of nodes with data
@inline function data_at_boundary_nodes(S::NTuple{M}, D) where M
    N = M ÷ 2
    S₀, S₁, S₂ = S[N], S[N + 1], S[N + 2]
    χ, m = true, 1
    for n in N-1:-1:1
        χ &= D[n]
        m += χ
        S₀, S₁, S₂ = ifelse(χ, (S[n], S₀, S₁), (S₀, S₁, S₂))
    end
    return (S₀, S₁, S₂), m
end

@inline function boundary_slope(extrapolation, S₀, S₁, S₂, data_length)
    δ₁, δ₂ = S₀ - S₁, S₁ - S₂
    s = limited_slope(extrapolation, δ₁, ifelse(data_length ≥ 3, δ₂, δ₁))
    return ifelse(data_length ≥ 2, s, zero(s))
end

# The values at the 2N nodes along `dir`, node N + 1 at (i, j, k), about the location between nodes N and N + 1
@inline function stencil_values(i, j, k, grid, scheme::AbstractAdvectionScheme{N}, dir, ℓ, ψ, args...) where N
    di, dj, dk = unit_vector(dir)

    S = ntuple(Val(2N)) do n
        @inline
        o = n - N - 1
        stencil_value(i + o * di, j + o * dj, k + o * dk, grid, ψ, args...)
    end

    D = ntuple(@inline(n -> has_data(i, j, k, grid, dir, ℓ, n - N - 1)), Val(2N))

    S⁻, N⁻ = data_at_boundary_nodes(S, D)
    S⁺, N⁺ = data_at_boundary_nodes(reverse(S), reverse(D))

    s⁻ = boundary_slope(scheme.extrapolation, S⁻..., N⁻ + N⁺)
    s⁺ = boundary_slope(scheme.extrapolation, S⁺..., N⁻ + N⁺)

    FT = eltype(S)
    n⁻, n⁺ = N + 1 - N⁻, N + N⁺

    G = ntuple(Val(2N)) do n
        @inline
        o = n - N - 1
        iₙ, jₙ, kₙ = i + o * di, j + o * dj, k + o * dk
        g⁻ = ghost_value(iₙ, jₙ, kₙ, grid, scheme.extrapolation, S⁻[1] + (convert(FT, n⁻) - n) * s⁻, ℓ, ψ, args...)
        g⁺ = ghost_value(iₙ, jₙ, kₙ, grid, scheme.extrapolation, S⁺[1] + (n - convert(FT, n⁺)) * s⁺, ℓ, ψ, args...)
        ifelse(n < n⁻, g⁻, ifelse(n > n⁺, g⁺, S[n]))
    end

    Sᴮ = ifelse(D[N], S[N], S[N + 1])

    return ifelse(D[N] & D[N + 1], G, ifelse(D[N] | D[N + 1], ntuple(@inline(n -> Sᴮ), Val(2N)), S))
end

@inline function stencil_values(i, j, k, grid, ::AbstractAdvectionScheme{1}, dir, ℓ, ψ, args...)
    di, dj, dk = unit_vector(dir)
    return stencil_value(i - di, j - dj, k - dk, grid, ψ, args...), stencil_value(i, j, k, grid, ψ, args...)
end

@generated function linear_reconstruction(S::NTuple{M}, ::Type{FT}, ::Val{shift}) where {M, FT, shift}
    reconstruction = calc_reconstruction_stencil(FT, M ÷ 2, shift, :stencil)
    return quote
        $(Expr(:meta, :inline))
        @inbounds @muladd $reconstruction
    end
end

#####
##### Interpolations named by the location they reconstruct at
#####

const XFG = Union{<:XFlatGrid, <:ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:XFlatGrid}}
const YFG = Union{<:YFlatGrid, <:ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:YFlatGrid}}
const ZFG = Union{<:ZFlatGrid, <:ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:ZFlatGrid}}

for (d, ξ, FlatGrid) in zip(1:3, (:x, :y, :z), (:XFG, :YFG, :ZFG))
    for ℓ in Iterators.product((Center(), Face()), (Center(), Face()), (Center(), Face()))
        target       = Base.setindex(ℓ, ℓ[d] isa Center ? Face() : Center(), d)
        superscripts = map(L -> L isa Center ? :ᶜ : :ᶠ, target)
        biased       = Symbol(:biased_interpolate_, ξ, superscripts...)
        symmetric    = Symbol(:symmetric_interpolate_, ξ, superscripts...)
        shift        = ℓ[d] isa Face ? 1 : 0
        indices      = Base.setindex((:i, :j, :k), :($((:i, :j, :k)[d]) + $shift), d)

        @eval begin
            @inline $biased(i, j, k, grid, scheme, bias, ψ, args...) = biased_interpolate($(indices...), grid, scheme, bias, $(Val(d)), $ℓ, ψ, args...)
            @inline $symmetric(i, j, k, grid, scheme, ψ, args...)    = symmetric_interpolate($(indices...), grid, scheme, $(Val(d)), $ℓ, ψ, args...)

            # A Flat direction has a single node
            @inline $biased(i, j, k, grid::$FlatGrid, scheme, bias, ψ, args...) = stencil_value(i, j, k, grid, ψ, args...)
            @inline $symmetric(i, j, k, grid::$FlatGrid, scheme, ψ, args...)    = stencil_value(i, j, k, grid, ψ, args...)
        end
    end
end
