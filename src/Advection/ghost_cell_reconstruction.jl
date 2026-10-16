using Base.FastMath: min_fast, max_fast, minmax_fast
using KernelAbstractions: @index, @kernel
using KernelAbstractions.Extras.LoopInfo: @unroll
using Oceananigans.Grids: peripheral_node, topology, halo_size
using Oceananigans.Utils: KernelParameters, launch!, newton_div

"""
    GhostCells([FT=Oceananigans.defaults.FloatType])

Boundary scheme of a `WENO` reconstruction of order 5 or higher, `WENO(order = 7, boundary_scheme = GhostCells())`.
The peripheral nodes of a stencil, which are inactive or lie on a boundary, are filled by linear extrapolation of the
other nodes in index coordinates, the coordinates of the WENO reconstruction, which keeps the reconstruction exact next to
boundaries for data that are linear in the grid index, and the reconstructed values next to boundaries are limited.
"""
struct GhostCells{S, B}
    scheme :: S
    boundary_distances :: B
end

GhostCells(FT::DataType = Oceananigans.defaults.FloatType) = GhostCells(nothing, nothing)
GhostCells(scheme) = GhostCells(scheme, nothing)

Base.summary(::GhostCells) = "GhostCells()"

const GhostCellWENO = WENO{<:Any, <:Any, <:Any, <:Any, <:Any, <:GhostCells}

Adapt.adapt_structure(to, scheme::GhostCells) = GhostCells(Adapt.adapt(to, scheme.scheme), Adapt.adapt(to, scheme.boundary_distances))

Oceananigans.Architectures.on_architecture(to, scheme::GhostCells) =
    GhostCells(on_architecture(to, scheme.scheme), on_architecture(to, scheme.boundary_distances))

#####
##### Boundary distances of the stencil nodes, computed once when the scheme is materialized on a grid
#####

# The boundary distances of a node pack whether the node is peripheral in bit 0 and, for x, y and z, the number of nodes
# after it going backward and going forward before the first peripheral one, three bits each
@inline boundary_distance_shift(::Val{D}, s) where D = 1 + 3 * (2 * (D - 1) + (s > 0))

@inline node_is_peripheral(B) = isodd(B)
@inline boundary_distance(B, d, s) = Int((B >> boundary_distance_shift(d, s)) & 0x00000007)

# Whether the forward and the backward neighbors of the node along x, y and z are peripheral
@inline function peripheral_neighbors(B)
    pˣ = (boundary_distance(B, Val(1), +1) == 0, boundary_distance(B, Val(1), -1) == 0)
    pʸ = (boundary_distance(B, Val(2), +1) == 0, boundary_distance(B, Val(2), -1) == 0)
    pᶻ = (boundary_distance(B, Val(3), +1) == 0, boundary_distance(B, Val(3), -1) == 0)
    return pˣ, pʸ, pᶻ
end

# Of the M nodes after the node along s d, the number before the first peripheral one. The halo bounds M, so that the
# nodes beyond the outermost faces stay within it, and Flat directions, without halo, count none
@inline function boundary_distance(i, j, k, grid, ℓx, ℓy, ℓz, d::Val{D}, s, M) where D
    di, dj, dk = unit_vector(d)
    M = min(M, halo_size(grid, D) - 1, 7)
    n = 0
    reached = false
    @unroll for m in 1:M
        reached |= peripheral_node(i + s * m * di, j + s * m * dj, k + s * m * dk, grid, ℓx, ℓy, ℓz)
        n += !reached
    end
    return n
end

@inline function boundary_distance_bits(i, j, k, grid, ℓx, ℓy, ℓz, d, M)
    n⁻ = boundary_distance(i, j, k, grid, ℓx, ℓy, ℓz, d, -1, M)
    n⁺ = boundary_distance(i, j, k, grid, ℓx, ℓy, ℓz, d, +1, M)
    return (UInt32(n⁻) << boundary_distance_shift(d, -1)) | (UInt32(n⁺) << boundary_distance_shift(d, +1))
end

# Mˣ, Mʸ and Mᶻ nodes are counted along x, y and z
@inline function node_boundary_distances(i, j, k, grid, ℓx, ℓy, ℓz, Mˣ, Mʸ, Mᶻ)
    p  = UInt32(peripheral_node(i, j, k, grid, ℓx, ℓy, ℓz))
    Bˣ = boundary_distance_bits(i, j, k, grid, ℓx, ℓy, ℓz, Val(1), Mˣ)
    Bʸ = boundary_distance_bits(i, j, k, grid, ℓx, ℓy, ℓz, Val(2), Mʸ)
    Bᶻ = boundary_distance_bits(i, j, k, grid, ℓx, ℓy, ℓz, Val(3), Mᶻ)
    return p | Bˣ | Bʸ | Bᶻ
end

@kernel function _compute_boundary_distances!(boundary_distances, grid, ℓx, ℓy, ℓz)
    i, j, k = @index(Global, NTuple)
    @inbounds boundary_distances[i, j, k] = node_boundary_distances(i, j, k, grid, ℓx, ℓy, ℓz, 7, 7, 7)
end

# The faces of the interior nodes read the boundary distances of the nodes one beyond them in non-flat directions
boundary_distances_range(N, H) = ifelse(H > 0, 0:N+1, 1:1)

function boundary_distances_array(grid, ℓx, ℓy, ℓz)
    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)
    ranges = (boundary_distances_range(Nx, Hx), boundary_distances_range(Ny, Hy), boundary_distances_range(Nz, Hz))
    data = on_architecture(architecture(grid), zeros(UInt32, length.(ranges)...))
    boundary_distances = OffsetArray(data, (first.(ranges) .- 1)...)
    launch!(architecture(grid), grid, KernelParameters(ranges...), _compute_boundary_distances!, boundary_distances, grid, ℓx, ℓy, ℓz)
    return boundary_distances
end

# the locations of the advected fields: tracers at (c, c, c) and the velocities at (f, c, c), (c, f, c) and (c, c, f)
const boundary_distances_locations = ((Center(), Center(), Center()), (Face(), Center(), Center()), (Center(), Face(), Center()), (Center(), Center(), Face()))

ghost_cell_boundary_distances(grid) = map(ℓ -> boundary_distances_array(grid, ℓ...), boundary_distances_locations)

@inline located_boundary_distances(B, ℓx, ℓy, ℓz) = nothing
@inline located_boundary_distances(B::Tuple, ::Center, ::Center, ::Center) = @inbounds B[1]
@inline located_boundary_distances(B::Tuple, ::Face,   ::Center, ::Center) = @inbounds B[2]
@inline located_boundary_distances(B::Tuple, ::Center, ::Face,   ::Center) = @inbounds B[3]
@inline located_boundary_distances(B::Tuple, ::Center, ::Center, ::Face)   = @inbounds B[4]

# Fields read the boundary distances built at materialization. Functions, and locations without arrays, compute in place
# the N - 1 nodes of the stencil along the face direction d and the neighbors across it
@inline stencil_boundary_distances(B::AbstractArray, ψ::AbstractArray, i, j, k, grid, ℓx, ℓy, ℓz, d, ::Val) = @inbounds B[i, j, k]

@inline stencil_boundary_distances(B, ψ, i, j, k, grid, ℓx, ℓy, ℓz, d, ::Val{N}) where N =
    node_boundary_distances(i, j, k, grid, ℓx, ℓy, ℓz, ifelse(d === Val(1), N - 1, 1), ifelse(d === Val(2), N - 1, 1), ifelse(d === Val(3), N - 1, 1))

@inline face_boundary_distances(scheme::GhostCells{<:WENO{N}}, ψ, i, j, k, grid, ℓx, ℓy, ℓz, d) where N =
    stencil_boundary_distances(located_boundary_distances(scheme.boundary_distances, ℓx, ℓy, ℓz), ψ, i, j, k, grid, ℓx, ℓy, ℓz, d, Val(N))

#####
##### Reconstruction
#####

# Differences between consecutive stencil nodes, ΔSₙ = Sₙ₊₁ - Sₙ
@inline node_differences(S::NTuple{M}) where M = ntuple(n -> @inbounds(S[n + 1] - S[n]), Val(M - 1))

# The difference between the last two nodes before the boundary on the side s of the face, of which n nodes beyond L or R
# come before it: the difference across the face when none does
@inline function boundary_difference(ΔS, n, s, ::Val{N}) where N
    Δ = @inbounds ΔS[N]
    @unroll for m in 1:N-2
        Δ = @inbounds ifelse(m ≤ n, ΔS[N + m * s], Δ)
    end
    return Δ
end

# Whether the upwind stencil reads a node beyond the boundary of L or R: nodes 1 to 2N - 1 form the stencil
# of a LeftBias reconstruction, nodes 2 to 2N that of a RightBias one
@inline function ghosted(nᴸ, nᴿ, bias, ::Val{N}) where N
    aᴸ = (nᴸ ≥ N - 1) & (nᴿ ≥ N - 2)
    aᴿ = (nᴸ ≥ N - 2) & (nᴿ ≥ N - 1)
    return !ifelse(bias === LeftBias, aᴸ, aᴿ)
end

# WENO reconstruction of the face between the stencil nodes N and N + 1, given in index order with their differences,
# after the peripheral nodes of each side are moved onto the line through the last two nodes before the boundary. The
# line is drawn in index coordinates, the coordinates of the WENO reconstruction: the differences that reach a peripheral
# node take the difference between the last two nodes before the boundary
@inline function ghost_cell_reconstruction(weno::WENO{N}, S, ΔS, nᴸ, nᴿ, bias, δˢ) where N
    Δᴸᵇ = boundary_difference(ΔS, nᴸ, -1, Val(N))
    Δᴿᵇ = boundary_difference(ΔS, nᴿ, +1, Val(N))

    Δᴸ = ntuple(Val(N - 1)) do m
        @inline
        @inbounds ifelse(m ≤ nᴸ, ΔS[N - m], Δᴸᵇ)
    end

    Δᴿ = ntuple(Val(N - 1)) do m
        @inline
        @inbounds ifelse(m ≤ nᴿ, ΔS[N + m], Δᴿᵇ)
    end

    Δ = (reverse(Δᴸ)..., @inbounds(ΔS[N]), Δᴿ...)

    # the upwind stencil is made of the first 2N - 1 nodes for a LeftBias, of the last 2N - 1 reversed for a RightBias
    δ = ntuple(Val(2N - 2)) do n
        @inline
        @inbounds ifelse(bias === LeftBias, Δ[n], -Δ[2N - n])
    end

    ψᵁ = @inbounds ifelse(bias === LeftBias, S[N], S[N + 1])
    γ  = ghosted(nᴸ, nᴿ, bias, Val(N))

    ω = biased_weno_weights(smoothness(γ, δ, δˢ), nothing, weno)
    ψ̂ = weno_reconstruction(weno, ψᵁ, δ, ω)

    return ψ̂, γ
end

@inline smoothness(γ, δ,  ::Nothing) = δ
@inline smoothness(γ, δ, δˢ::NTuple) = ifelse(γ, δ, δˢ)
@inline smoothness(γ, δ, δˢ::Tuple{Tuple, Tuple}) = ifelse(γ, (δ, δ), δˢ)

# 2⁻ᵉ for the binary exponent e of x, so that x 2⁻ᵉ lies in [1, 2); zero for zero and subnormal x. E holds the exponent bits of x, E⁻¹ those of 2⁻ᵉ
@inline function inverse_power_of_two(x::FT) where FT
    U = Base.uinttype(FT)
    E = reinterpret(U, x) & Base.exponent_mask(FT)
    E⁻¹ = (U(2 * Base.exponent_bias(FT)) << Base.significand_bits(FT)) - E
    return ifelse(E > zero(U), reinterpret(FT, E⁻¹), zero(FT))
end

# Value at the boundary half a node beyond ψ₀, extrapolated with the differences δ₁ = ψ₀ - ψ₁, δ₂ = ψ₁ - ψ₂, δ₃ = ψ₂ - ψ₃
# of ψ₀ and the three nodes after it when none of them is peripheral (χ), and ψ₀ otherwise
@inline @muladd function boundary_value(ψ₀, δ₁, δ₂, δ₃, χ, WCT)
    # minmod: the difference closest to zero when all three share a sign, zero otherwise
    δ⁻ = min_fast(δ₁, δ₂, δ₃)
    δ⁺ = max_fast(δ₁, δ₂, δ₃)
    δᵐ = max_fast(δ⁻, zero(δ₁)) + min_fast(δ⁺, zero(δ₁))

    # Ratio of the harmonic to the arithmetic mean of the squared differences, which do not depend on their scale:
    # the power of two σ below the largest difference keeps the products in range
    σ = inverse_power_of_two(max_fast(-δ⁻, δ⁺))
    a = (δ₁ * σ)^2
    b = (δ₂ * σ)^2
    c = (δ₃ * σ)^2
    D = (a + b + c) * (a * b + b * c + c * a)

    # the agreement times half the minmod difference
    Δψ = newton_div(WCT, 9 * a * b * c * δᵐ, 2 * ifelse(D > 0, D, one(D)))

    return ψ₀ + ifelse(χ, Δψ, zero(ψ₀))
end

# The two neighbors of the node along a direction, each replaced by the node value if it is peripheral
@inline neighbors_or_node(i, j, k, ψ₀, ::Val, ::Type{Flat}, p, grid, ψ, args...) = ψ₀, ψ₀, false

@inline function neighbors_or_node(i, j, k, ψ₀, d, topology, p, grid, ψ, args...)
    di, dj, dk = unit_vector(d)
    p₊₁, p₋₁ = p
    ψ₊₁ = ifelse(p₊₁, ψ₀, stencil_value(i + di, j + dj, k + dk, grid, ψ, args...))
    ψ₋₁ = ifelse(p₋₁, ψ₀, stencil_value(i - di, j - dj, k - dk, grid, ψ, args...))
    return ψ₊₁, ψ₋₁, p₊₁ | p₋₁
end

# Extrema of the node and of its neighbors that are not peripheral, the same whichever face asks for them, and whether a neighbor is peripheral
@inline function local_extrema(i, j, k, grid, B, ψ, args...)
    ψ₀ = stencil_value(i, j, k, grid, ψ, args...)
    pˣ, pʸ, pᶻ = peripheral_neighbors(B)

    ψᵢ₊₁, ψᵢ₋₁, bˣ = neighbors_or_node(i, j, k, ψ₀, Val(1), topology(grid, 1), pˣ, grid, ψ, args...)
    ψⱼ₊₁, ψⱼ₋₁, bʸ = neighbors_or_node(i, j, k, ψ₀, Val(2), topology(grid, 2), pʸ, grid, ψ, args...)
    ψₖ₊₁, ψₖ₋₁, bᶻ = neighbors_or_node(i, j, k, ψ₀, Val(3), topology(grid, 3), pᶻ, grid, ψ, args...)

    ψˣ⁻, ψˣ⁺ = minmax_fast(ψᵢ₊₁, ψᵢ₋₁)
    ψʸ⁻, ψʸ⁺ = minmax_fast(ψⱼ₊₁, ψⱼ₋₁)
    ψᶻ⁻, ψᶻ⁺ = minmax_fast(ψₖ₊₁, ψₖ₋₁)

    ψ⁻ = min_fast(ψ₀, ψˣ⁻, ψʸ⁻, ψᶻ⁻)
    ψ⁺ = max_fast(ψ₀, ψˣ⁺, ψʸ⁺, ψᶻ⁺)

    return ψ⁻, ψ⁺, bˣ | bʸ | bᶻ
end

# The face between L and R along d: R is the node i, j, k and L the node behind it
@inline function ghost_cell_interpolate(i, j, k, grid, scheme, bias, d, ℓx, ℓy, ℓz, stencil, ψ, args...)
    di, dj, dk = unit_vector(d)
    Bᴸ = face_boundary_distances(scheme, ψ, i - di, j - dj, k - dk, grid, ℓx, ℓy, ℓz, d)
    Bᴿ = face_boundary_distances(scheme, ψ, i, j, k, grid, ℓx, ℓy, ℓz, d)
    ℰᴸ = local_extrema(i - di, j - dj, k - dk, grid, Bᴸ, ψ, args...)
    ℰᴿ = local_extrema(i, j, k, grid, Bᴿ, ψ, args...)
    return limited_interpolate(i, j, k, grid, scheme, bias, d, Bᴸ, Bᴿ, ℰᴸ, ℰᴿ, stencil, ψ, args...)
end

# The field ψ with the location of its nodes and the local extrema to be satisfied
struct FaceExtrema{F, P, L, R}
     field :: F
  location :: P
      left :: L
     right :: R
end

@inline function ghost_cell_interpolate(i, j, k, grid, scheme, bias, d, ℓx, ℓy, ℓz, stencil, ψ::FaceExtrema)
    di, dj, dk = unit_vector(d)
    Bᴸ = face_boundary_distances(scheme, ψ.field, i - di, j - dj, k - dk, grid, ℓx, ℓy, ℓz, d)
    Bᴿ = face_boundary_distances(scheme, ψ.field, i, j, k, grid, ℓx, ℓy, ℓz, d)
    return limited_interpolate(i, j, k, grid, scheme, bias, d, Bᴸ, Bᴿ, ψ.left, ψ.right, stencil, ψ.field)
end

@inline stencil_location(ψ, ℓx, ℓy, ℓz)               = (ℓx, ℓy, ℓz)
@inline stencil_location(ψ::FaceExtrema,  ℓx, ℓy, ℓz) = ψ.location
# the vorticity is the only function operand whose nodes are not at the interpolation location
@inline stencil_location(::typeof(ζ₃ᶠᶠᶜ), ℓx, ℓy, ℓz) = (Face(), Face(), Center())

# The face value between L and R, clamped within bounds built from the local extrema ℰᴸ and ℰᴿ of L and R
@inline @muladd function limited_interpolate(i, j, k, grid, scheme::GhostCells{<:WENO{N, FT, WCT}}, bias, d, Bᴸ, Bᴿ, ℰᴸ, ℰᴿ, stencil, ψ, args...) where {N, FT, WCT}
    weno = scheme.scheme
    e = unit_vector(d)

    # The stencil in index order: node N is the cell L on the left of the face, node N + 1 the cell R on its right
    S  = upwind_stencil(i, j, k, grid, weno, LeftBias, e, ψ, args...)
    ΔS = node_differences(S)
    nᴸ = boundary_distance(Bᴸ, d, -1)
    nᴿ = boundary_distance(Bᴿ, d, +1)

    ψᴸ⁻, ψᴸ⁺, bᴸ = ℰᴸ
    ψᴿ⁻, ψᴿ⁺, bᴿ = ℰᴿ

    # One boundary value serves the face: beyond L (βᴸ) when node N + 2 comes before the boundary of R, beyond R otherwise
    βᴸ = nᴿ ≥ 1

    # none of the three nodes after the one next to the boundary is peripheral
    χ = ifelse(βᴸ, !node_is_peripheral(Bᴿ) & (nᴿ ≥ 2), !node_is_peripheral(Bᴸ) & (nᴸ ≥ 2))

    # differences going toward the boundary, from L, R and the nodes after them beyond L, and their mirror beyond R
    ψ₀ = @inbounds ifelse(βᴸ, S[N], S[N + 1])
    δ₁ = @inbounds ifelse(βᴸ, -ΔS[N],     ΔS[N])
    δ₂ = @inbounds ifelse(βᴸ, -ΔS[N + 1], ΔS[N - 1])
    δ₃ = @inbounds ifelse(βᴸ, -ΔS[N + 2], ΔS[N - 2])
    ψᵇ = boundary_value(ψ₀, δ₁, δ₂, δ₃, χ, WCT)

    # the extrema also contain the nodes beyond L and R along the line, or the boundary value if they are peripheral
    ψᴸᴸ = @inbounds ifelse(nᴸ ≥ 1, S[N - 1], ifelse(βᴸ, ψᵇ, S[N]))
    ψᴿᴿ = @inbounds ifelse(βᴸ, S[N + 2], ψᵇ)

    ψᴸ⁻ = min_fast(ψᴸ⁻, ψᴸᴸ)
    ψᴸ⁺ = max_fast(ψᴸ⁺, ψᴸᴸ)
    ψᴿ⁻ = min_fast(ψᴿ⁻, ψᴿᴿ)
    ψᴿ⁺ = max_fast(ψᴿ⁺, ψᴿᴿ)

    ψᵁ  = @inbounds ifelse(bias === LeftBias, S[N], S[N + 1])
    ψᵁ⁻ = ifelse(bias === LeftBias, ψᴸ⁻, ψᴿ⁻)
    ψᵁ⁺ = ifelse(bias === LeftBias, ψᴸ⁺, ψᴿ⁺)
    ψᴰ⁻ = ifelse(bias === LeftBias, ψᴿ⁻, ψᴸ⁻)
    ψᴰ⁺ = ifelse(bias === LeftBias, ψᴿ⁺, ψᴸ⁺)

    # faces next to boundaries are limited (λ), the others get unbounded limits that the clamp below leaves unchanged
    λ  = ghosted(nᴸ, nᴿ, bias, Val(N)) | bᴸ | bᴿ
    α  = convert(typeof(ψᵁ), 3//2)
    ψ̂⁻ = ifelse(λ, max_fast(ψᵁ - α * (ψᵁ⁺ - ψᵁ),  ψᴰ⁻), typemin(ψᵁ))
    ψ̂⁺ = ifelse(λ, min_fast(ψᵁ + α * (ψᵁ  - ψᵁ⁻), ψᴰ⁺), typemax(ψᵁ))

    δˢ   = smoothness_differences(i, j, k, grid, weno, bias, e, stencil, args...)
    ψ̂, _ = ghost_cell_reconstruction(weno, S, ΔS, nᴸ, nᴿ, bias, δˢ)

    return clamp(ψ̂, ψ̂⁻, ψ̂⁺)
end

for (dir, ξ) in enumerate((:x, :y, :z))
    e = unit_vector(Val(dir))
    d = Val(dir)
    ℓᶠ = ntuple(3) do n
        n == dir ? :(Face()) : :(Center())
    end
    ℓᶜ = (:(Center()), :(Center()), :(Center()))
    interpolateᶠ = Symbol(:biased_interpolate_, ξ, ntuple(d -> d == dir ? :ᶠ : :ᵃ, 3)...)
    interpolateᶜ = Symbol(:biased_interpolate_, ξ, ntuple(d -> d == dir ? :ᶜ : :ᵃ, 3)...)

    @eval begin
        @inline $interpolateᶠ(i, j, k, grid, scheme::GhostCells, bias, ψ, args...) = ghost_cell_interpolate(i, j, k,              grid, scheme, bias, $d, stencil_location(ψ, $(ℓᶜ...))..., DefaultStencil(), ψ, args...)
        @inline $interpolateᶜ(i, j, k, grid, scheme::GhostCells, bias, ψ, args...) = ghost_cell_interpolate(((i, j, k) .+ $e)..., grid, scheme, bias, $d, stencil_location(ψ, $(ℓᶠ...))..., DefaultStencil(), ψ, args...)

        @inline $interpolateᶠ(i, j, k, grid, scheme::GhostCells, bias, ψ, stencil::AbstractSmoothnessStencil, args...) = ghost_cell_interpolate(i, j, k,              grid, scheme, bias, $d, stencil_location(ψ, $(ℓᶜ...))..., stencil, ψ, args...)
        @inline $interpolateᶜ(i, j, k, grid, scheme::GhostCells, bias, ψ, stencil::AbstractSmoothnessStencil, args...) = ghost_cell_interpolate(((i, j, k) .+ $e)..., grid, scheme, bias, $d, stencil_location(ψ, $(ℓᶠ...))..., stencil, ψ, args...)
    end

    for location in (:ᶜ, :ᶠ)
        symmetric = Symbol(:symmetric_interpolate_, ξ, ntuple(d -> d == dir ? location : :ᵃ, 3)...)
        @eval @inline $symmetric(i, j, k, grid, ::GhostCells{<:WENO{<:Any, FT}}, args...) where FT = $symmetric(i, j, k, grid, Centered{1, FT}(nothing, ExplicitTimeDiscretization()), args...)
    end
end
