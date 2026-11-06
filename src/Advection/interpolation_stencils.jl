using Oceananigans.Grids: inactive_node, flip_location, topology, Open, TwoSided

#####
##### Interpolations to a three-dimensional location along one direction
#####
##### `biased_interpolate_xᶠᶜᶜ(i, j, k, grid, scheme, bias, ψ, args...)` reconstructs `ψ` at `(Face, Center, Center)`
##### from the `(Center, Center, Center)` nodes along `x`. Every scheme reconstructs from the stencil of `load_stencil`.
#####

for (d, ξ) in enumerate((:x, :y, :z)), ℓx in (:ᶜ, :ᶠ), ℓy in (:ᶜ, :ᶠ), ℓz in (:ᶜ, :ᶠ)
    ℓ = map(l -> l == :ᶜ ? Center() : Face(), (ℓx, ℓy, ℓz))
    biased_interpolation    = Symbol(:biased_interpolate_, ξ, ℓx, ℓy, ℓz)
    symmetric_interpolation = Symbol(:symmetric_interpolate_, ξ, ℓx, ℓy, ℓz)

    @eval begin
        @inline $biased_interpolation(i, j, k, grid, scheme, args...)    = biased_interpolate(i, j, k, grid, $(Val(d)), $ℓ, scheme, args...)
        @inline $symmetric_interpolation(i, j, k, grid, scheme, args...) = symmetric_interpolate(i, j, k, grid, $(Val(d)), $ℓ, scheme, args...)
    end
end

#####
##### Stencils
#####

@inline stencil_node(i, j, k, ::Val{1}, s) = (i + s, j, k)
@inline stencil_node(i, j, k, ::Val{2}, s) = (i, j + s, k)
@inline stencil_node(i, j, k, ::Val{3}, s) = (i, j, k + s)

@inline node_location(::Val{1}, (ℓx, ℓy, ℓz)) = (flip_location(ℓx), ℓy, ℓz)
@inline node_location(::Val{2}, (ℓx, ℓy, ℓz)) = (ℓx, flip_location(ℓy), ℓz)
@inline node_location(::Val{3}, (ℓx, ℓy, ℓz)) = (ℓx, ℓy, flip_location(ℓz))

@inline stencil_value(i, j, k, grid, ψ, args...)           = @inbounds ψ[i, j, k]
@inline stencil_value(i, j, k, grid, ψ::Callable, args...) = @inbounds ψ(i, j, k, grid, args...)

# beyond an Open side the boundary condition fills the first halo cell only
@inline unfilled_halo(i, N, T) = ((T <: TwoSided{Open}) & (i < 0)) | ((T <: TwoSided{<:Any, Open}) & (i > N + 1))

"""
$(TYPEDSIGNATURES)

Return the `M` values `S` of `ψ` along `dir` about the point `(i, j, k)` located at `ℓ`, the first `N` of them before the point, with
the walls `nᴸ`, the last node without data in `1:N` (`0` when there is none), and `nᴿ`, the first node without data in `M-N+1:M`
(`M + 1` when there is none).
"""
@inline function load_stencil(i, j, k, grid, dir::Val{d}, ℓ, ::Val{N}, ::Val{M}, ψ, args...) where {d, N, M}
    σ  = ℓ[d] isa Center

    s(n) = n - N - 1 + σ
    # a node lies past a wall when its cell along `dir` is inactive or an unfilled halo cell, a face node when the cell between it and the point is
    cell(n) = stencil_node(i, j, k, dir, s(n) - (σ & (n > N)))
    no_data(n) = inactive_node(cell(n)..., grid, Center(), Center(), Center()) | unfilled_halo(cell(n)[d], size(grid, d), topology(grid, d))

    nᴸ = max(Int32(0),     ntuple(n -> ifelse(no_data(n),         Int32(n),         Int32(0)),     Val(N))...)
    nᴿ = min(Int32(M + 1), ntuple(n -> ifelse(no_data(M - N + n), Int32(M - N + n), Int32(M + 1)), Val(N))...)

    # evaluated from the last node to the first, which lowers the register peak of the GPU momentum kernels
    Sʳ = ntuple(n -> stencil_value(stencil_node(i, j, k, dir, s(M + 1 - n))..., grid, ψ, args...), Val(M))
    S  = reverse(Sʳ)

    return S, nᴸ, nᴿ
end

# the 2J values of the stencil `S` about its point
@inline central_nodes(S::NTuple{M}, ::Val{J}) where {M, J} = ntuple(n -> @inbounds(S[M ÷ 2 - J + n]), Val(2J))

#####
##### Next to walls
#####

# the largest half-width with `L` nodes with data on the left and `R` on the right: a biased stencil reads `N` nodes upwind and `N - 1`
# downwind, a symmetric one `N` on both sides
@inline stencil_reach(L, R, bias) = ifelse(bias == LeftBias, min(L, R + Int32(1)), min(L + Int32(1), R))
@inline stencil_reach(L, R, ::Nothing) = min(L, R)

"""
$(TYPEDSIGNATURES)

Return the reconstruction of `scheme` from the stencil `(S, nᴸ, nᴿ)` of `load_stencil`, with the smoothness stencils `Sˢ` of WENO.
The point takes the first scheme of the `buffer_scheme` chain of `scheme`, `scheme` included, whose half-width does not exceed the
`stencil_reach` of the nodes with data. Schemes of half-width one read their two nodes whatever lies past them.
"""
@inline function buffered_interpolate(scheme::AbstractAdvectionScheme{N}, bias, (S, nᴸ, nᴿ), Sˢ = ()) where N
    K = stencil_reach(Int32(N) - nᴸ, nᴿ - Int32(N + 1), bias)
    return buffered_interpolate(scheme, bias, K, S, Sˢ)
end

@inline function buffered_interpolate(scheme::AbstractAdvectionScheme{N}, bias, K, S, Sˢ) where N
    ψ = biased_interpolate(scheme, bias, central_nodes(S, Val(N)), map(s -> central_nodes(s, Val(N)), Sˢ))
    # BFloat16 WENO reconstructions are Float32, the lowest-order buffer schemes BFloat16
    ψᵇ = convert(typeof(ψ), buffered_interpolate(scheme.buffer_scheme, bias, K, S, Sˢ))
    return ifelse(K ≥ N, ψ, ψᵇ)
end

@inline buffered_interpolate(scheme::AbstractAdvectionScheme{1}, bias, K, S, Sˢ) = biased_interpolate(scheme, bias, central_nodes(S, Val(1)), ())
@inline buffered_interpolate(::Nothing, bias, K, S, Sˢ) = zero(eltype(S))

# the smoothness stencils `Sˢ` are used by WENO only
@inline biased_interpolate(scheme::AbstractAdvectionScheme, bias, S, Sˢ) = biased_interpolate(scheme, bias, S)

# main's chains of `Centered` and `UpwindBiased` from half-width `J` down to half-width one
@inline decreasing_order_chain(::Type{Centered},     ::Val{0}, FT) = nothing
@inline decreasing_order_chain(::Type{UpwindBiased}, ::Val{0}, FT) = nothing
@inline decreasing_order_chain(::Type{Centered},     ::Val{J}, FT) where J = Centered{J, FT}(decreasing_order_chain(Centered, Val(J - 1), FT), nothing)
@inline decreasing_order_chain(::Type{UpwindBiased}, ::Val{J}, FT) where J = UpwindBiased{J, FT}(decreasing_order_chain(UpwindBiased, Val(J - 1), FT), nothing, nothing)

"""
$(TYPEDSIGNATURES)

Return the tracer value at a face from its stencil `S`, whose nodes up to `nᴸ` and from `nᴿ` lie past walls. With `U` nodes with data
upwind of the face (the donor included) and `D` downwind, the face takes the reconstruction of `scheme` when `U ≥ N` and `D ≥ N - 1`,
and otherwise the linear reconstruction on the largest stencil with data: `Centered` on `2U` nodes when `U ≤ D` or `D = 1`, and
`UpwindBiased` on `D + 1` upwind and `D` downwind nodes when `D < U`. A face with data on one side only takes the value of that side.
"""
@inline function wall_interpolate(scheme::AbstractAdvectionScheme{N}, bias, (S, nᴸ, nᴿ)) where N
    FT = eltype(S)
    L  = Int32(N) - nᴸ
    R  = nᴿ - Int32(N + 1)
    K  = min(L, R)
    U  = ifelse(bias == LeftBias, L, R)
    D  = ifelse(bias == LeftBias, R, L)

    ψ  = biased_interpolate(scheme, bias, S)
    ψᶜ = buffered_interpolate(decreasing_order_chain(Centered,     Val(N - 1), FT), nothing, K,            S, ())
    ψᵘ = buffered_interpolate(decreasing_order_chain(UpwindBiased, Val(N - 1), FT), bias,    D + Int32(1), S, ())
    ψ¹ = ifelse(iszero(L), S[N + 1], S[N])

    native = (U ≥ Int32(N)) & (D ≥ Int32(N - 1))
    return ifelse(native, ψ, convert(typeof(ψ), ifelse(iszero(K), ψ¹, ifelse((U ≤ D) | (K == Int32(1)), ψᶜ, ψᵘ))))
end

@inline function wall_interpolate(scheme::ACAS{N}, bias, (S, nᴸ, nᴿ)) where N
    L = Int32(N) - nᴸ
    K = min(L, nᴿ - Int32(N + 1))
    return ifelse(iszero(K), ifelse(iszero(L), S[N + 1], S[N]), buffered_interpolate(scheme, nothing, K, S, ()))
end

@inline tracer_interpolate(i, j, k, grid, dir, ℓ, scheme::AbstractAdvectionScheme{N}, bias, c) where N =
    wall_interpolate(scheme, bias, load_stencil(i, j, k, grid, dir, ℓ, Val(N), Val(2N), c))
