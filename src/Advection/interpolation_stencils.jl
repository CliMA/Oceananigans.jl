using Oceananigans.Grids: inactive_node

#####
##### Reconstruction stencils: the 2N nodes about a point along one direction, continued beyond boundaries
#####

@inline unit_vector(::Val{1}) = (1, 0, 0)
@inline unit_vector(::Val{2}) = (0, 1, 0)
@inline unit_vector(::Val{3}) = (0, 0, 1)

@inline stencil_value(i, j, k, grid, ψ, args...)           = @inbounds ψ[i, j, k]
@inline stencil_value(i, j, k, grid, ψ::Callable, args...) = ψ(i, j, k, grid, args...)

# A node at `ℓ = (ℓx, ℓy, ℓz)` has data when it is active, with the cell between it and the point standing for it along `dir`
@inline function has_data(node, grid, dir::Val{d}, ℓ, beyond_point) where d
    cell = ifelse((ℓ[d] isa Face) & beyond_point, node .- unit_vector(dir), node)
    return !inactive_node(cell..., grid, Base.setindex(ℓ, Center(), d)...)
end

"""
    MinmodExtrapolation()

Ghosts beyond a boundary that continue the stencil nodes with data by the minmod of the two differences at their end.
"""
struct MinmodExtrapolation end

"""
    VanLeerExtrapolation()

Ghosts beyond a boundary that continue the stencil nodes with data by the van Leer mean `2ab / (a + b)` of the two
differences `a` and `b` at their end, or by zero when `a` and `b` differ in sign.
"""
struct VanLeerExtrapolation end

Base.summary(::MinmodExtrapolation)  = "MinmodExtrapolation()"
Base.summary(::VanLeerExtrapolation) = "VanLeerExtrapolation()"

@inline limited_slope(::MinmodExtrapolation,  a, b) = ifelse(a * b > 0, ifelse(abs(a) < abs(b), a, b), zero(a))
@inline limited_slope(::VanLeerExtrapolation, a, b) = ifelse(a * b > 0, 2a * b / (a + b), zero(a))

# The slope that continues the nodes `S` beyond the end `n` of the nodes with data, which continue `available` nodes past `n`
# in the direction `s`: the limited slope of the two differences there, the one difference, or none
@inline function boundary_slope(extrapolation, S::NTuple{M}, n, s, available) where M
    δ₁ = S[n]                  - S[clamp(n + s, 1, M)]
    δ₂ = S[clamp(n + s, 1, M)] - S[clamp(n + 2s, 1, M)]
    return ifelse(available ≥ 2, limited_slope(extrapolation, δ₁, δ₂), ifelse(available == 1, δ₁, zero(δ₁)))
end

# The 2N nodes i - N to i + N - 1 along `dir` about the point between nodes N and N + 1. Beyond the nodes with data next to the
# point they continue the end node by the limited slope there; a point with data on one side only reads the node next to it on
# that side everywhere, and a point with no data on either side reads its nodes as they are.
@inline function stencil_nodes(i, j, k, grid, scheme::AbstractAdvectionScheme{N}, dir, ℓ, ψ, args...) where N
    node(n) = (i, j, k) .+ (n - N - 1) .* unit_vector(dir)
    S = ntuple(n -> stencil_value(node(n)..., grid, ψ, args...),    Val(2N))
    D = ntuple(n -> has_data(node(n),         grid, dir, ℓ, n > N), Val(2N))

    # the nodes with data next to the point
    first = max(0,      ntuple(n -> ifelse(D[n], 0, n), Val(N))...) + 1
    last  = min(2N + 1, ntuple(n -> ifelse(D[N + n], 2N + 1, N + n), Val(N))...) - 1

    slope⁻ = boundary_slope(scheme.extrapolation, S, first, +1, last - first)
    slope⁺ = boundary_slope(scheme.extrapolation, S, last,  -1, last - first)

    continued = ntuple(Val(2N)) do n
        ifelse(n < first, S[first] + (first - n) * slope⁻,
        ifelse(n > last,  S[last]  + (n - last)  * slope⁺, S[n]))
    end

    wall      = ifelse(D[N], S[N], S[N + 1])
    one_sided = ntuple(n -> wall, Val(2N))

    return ifelse(D[N] & D[N + 1], continued, ifelse(D[N] | D[N + 1], one_sided, S))
end

# A two-node stencil reads the two nodes next to the point
@inline function stencil_nodes(i, j, k, grid, ::AbstractAdvectionScheme{1}, dir, ℓ, ψ, args...)
    i⁻, j⁻, k⁻ = (i, j, k) .- unit_vector(dir)
    return stencil_value(i⁻, j⁻, k⁻, grid, ψ, args...), stencil_value(i, j, k, grid, ψ, args...)
end

# The reconstruction of `Centered` (`:symmetric`) or `UpwindBiased` (`:left`, `:right`) from the nodes `S`, with coefficients of type `FT`
@generated function linear_reconstruction(S::NTuple{M}, ::Type{FT}, ::Val{shift}) where {M, FT, shift}
    reconstruction = calc_reconstruction_stencil(FT, M ÷ 2, shift, :stencil)
    return quote
        $(Expr(:meta, :inline))
        @inbounds @muladd $reconstruction
    end
end

#####
##### Interpolations along one direction to the location in their name, e.g. `biased_interpolate_xᶠᶠᶜ` reconstructs at
##### (Face, Face, Center) from the nodes at (Center, Face, Center)
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
