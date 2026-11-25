using Oceananigans.Grids: inactive_node

#####
##### Stencils of 2N nodes along one direction, with ghosts beyond the data
#####

@inline unit_vector(::Val{1}) = (1, 0, 0)
@inline unit_vector(::Val{2}) = (0, 1, 0)
@inline unit_vector(::Val{3}) = (0, 0, 1)

@inline stencil_value(i, j, k, grid, ψ, args...)           = @inbounds ψ[i, j, k]
@inline stencil_value(i, j, k, grid, ψ::Callable, args...) = ψ(i, j, k, grid, args...)

@inline function has_data(node, grid, dir::Val{d}, ℓ, beyond_point) where d
    cell_towards_point = ifelse((ℓ[d] isa Face) & beyond_point, node .- unit_vector(dir), node)
    return !inactive_node(cell_towards_point..., grid, Base.setindex(ℓ, Center(), d)...)
end

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

Base.summary(::MinmodExtrapolation)    = "MinmodExtrapolation()"
Base.summary(::VanLeerExtrapolation)   = "VanLeerExtrapolation()"
Base.summary(::LinearExtrapolation)    = "LinearExtrapolation()"
Base.summary(::ContinuedExtrapolation) = "ContinuedExtrapolation()"

@inline limited_slope(::MinmodExtrapolation,    a, b) = ifelse(a * b > 0, ifelse(abs(a) < abs(b), a, b), zero(a))
@inline limited_slope(::VanLeerExtrapolation,   a, b) = ifelse(a * b > 0, 2a * b / (a + b), zero(a))
@inline limited_slope(::LinearExtrapolation,    a, b) = a
@inline limited_slope(::ContinuedExtrapolation, a, b) = zero(a)

# The data at the three nodes with data nearest the boundary, outermost first, and the number of nodes with data
@inline function data_at_boundary_nodes(S::NTuple{M}, D) where M
    N = M ÷ 2
    S₀, S₁, S₂ = S[N], S[N + 1], S[N + 2]
    joined, count = true, 1
    for n in N-1:-1:1
        joined &= D[n]
        count  += joined
        S₀, S₁, S₂ = ifelse(joined, (S[n], S₀, S₁), (S₀, S₁, S₂))
    end
    return (S₀, S₁, S₂), count
end

@inline function boundary_slope(extrapolation, S₀, S₁, S₂, data_length)
    δ₁, δ₂ = S₀ - S₁, S₁ - S₂
    slope  = limited_slope(extrapolation, δ₁, ifelse(data_length ≥ 3, δ₂, δ₁))
    return ifelse(data_length ≥ 2, slope, zero(slope))
end

# The values at the 2N nodes about the point between nodes N and N + 1
@inline function stencil_values(i, j, k, grid, scheme::AbstractAdvectionScheme{N}, dir, ℓ, ψ, args...) where N
    node(n) = (i, j, k) .+ (n - N - 1) .* unit_vector(dir)
    S = ntuple(n -> stencil_value(node(n)..., grid, ψ, args...),    Val(2N))
    D = ntuple(n -> has_data(node(n),         grid, dir, ℓ, n > N), Val(2N))

    S⁻, N⁻ = data_at_boundary_nodes(S, D)
    S⁺, N⁺ = data_at_boundary_nodes(reverse(S), reverse(D))

    slope⁻ = boundary_slope(scheme.extrapolation, S⁻..., N⁻ + N⁺)
    slope⁺ = boundary_slope(scheme.extrapolation, S⁺..., N⁻ + N⁺)

    first,  last  = N + 1 - N⁻, N + N⁺
    first′, last′ = convert(eltype(S), first), convert(eltype(S), last)
    ghosted = ntuple(Val(2N)) do n
        ifelse(n < first, S⁻[1] + (first′ - n) * slope⁻,
        ifelse(n > last,  S⁺[1] + (n - last′)  * slope⁺, S[n]))
    end

    wall_value = ifelse(D[N], S[N], S[N + 1])
    at_wall    = ntuple(n -> wall_value, Val(2N))

    return ifelse(D[N] & D[N + 1], ghosted, ifelse(D[N] | D[N + 1], at_wall, S))
end

@inline function stencil_values(i, j, k, grid, ::AbstractAdvectionScheme{1}, dir, ℓ, ψ, args...)
    i⁻, j⁻, k⁻ = (i, j, k) .- unit_vector(dir)
    return stencil_value(i⁻, j⁻, k⁻, grid, ψ, args...), stencil_value(i, j, k, grid, ψ, args...)
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
        target    = Base.setindex(ℓ, ℓ[d] isa Center ? Face() : Center(), d)
        code      = map(L -> L isa Center ? :ᶜ : :ᶠ, target)
        biased    = Symbol(:biased_interpolate_, ξ, code...)
        symmetric = Symbol(:symmetric_interpolate_, ξ, code...)
        shift     = ℓ[d] isa Face ? 1 : 0
        point     = Base.setindex((:i, :j, :k), :($((:i, :j, :k)[d]) + $shift), d)

        @eval begin
            @inline $biased(i, j, k, grid, scheme, bias, ψ, args...) = biased_interpolate($(point...), grid, scheme, bias, $(Val(d)), $ℓ, ψ, args...)
            @inline $symmetric(i, j, k, grid, scheme, ψ, args...)    = symmetric_interpolate($(point...), grid, scheme, $(Val(d)), $ℓ, ψ, args...)

            # A Flat direction has a single node
            @inline $biased(i, j, k, grid::$FlatGrid, scheme, bias, ψ, args...) = stencil_value(i, j, k, grid, ψ, args...)
            @inline $symmetric(i, j, k, grid::$FlatGrid, scheme, ψ, args...)    = stencil_value(i, j, k, grid, ψ, args...)
        end
    end
end
