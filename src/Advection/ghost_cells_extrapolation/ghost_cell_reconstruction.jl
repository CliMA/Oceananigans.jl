using Base.FastMath: min_fast, max_fast
using KernelAbstractions.Extras.LoopInfo: @unroll
using Oceananigans.Grids: node_has_data, unit_vector
using Oceananigans.Utils: BackendOptimizedDivision 

#####
##### Reconstruction next to boundaries: the stencil nodes without data are replaced with ghosts
#####

# The number of nodes with data beyond the node next to the face
@inline function boundary_distance(i, j, k, grid, d, ℓx, ℓy, ℓz, s, ::Val{N}) where N
    di, dj, dk = unit_vector(d)
    n = 0
    reached = false
    @unroll for m in 1:N-1
        reached |= !node_has_data(i + s * m * di, j + s * m * dj, k + s * m * dk, grid, d, ℓx, ℓy, ℓz)
        n += !reached
    end
    return n
end

@inline node_differences(S::NTuple{M}) where M = ntuple(n -> @inbounds(S[n + 1] - S[n]), Val(M - 1))

# T[a + s min(m, N - 1)] for m ≥ 0, and T[a] for m < 0, unrolled to avoid a run-time tuple index
@inline function offset_entry(T, a, m, s, ::Val{N}) where N
    t = @inbounds T[a]
    @unroll for p in 1:N-1
        t = @inbounds ifelse(p ≤ m, T[a + s * p], t)
    end
    return t
end

# The differences inward from the last node with data on the side s: along the own side, then across the face, and the
# last one repeated past the boundary of the other side
@inline function inward_differences(ΔS, n, nᵒ, s, ::Val{N}) where N
    inward(j, Δ) = ifelse(j ≤ n + 1, offset_entry(ΔS, N, n + 1 - j, s, Val(N)),
                   ifelse(j - n - 1 ≤ nᵒ, offset_entry(ΔS, N, j - n - 1, -s, Val(N)), Δ))
    Δ₁ = inward(1, zero(eltype(ΔS)))
    Δ₂ = inward(2, Δ₁)
    Δ₃ = inward(3, Δ₂)
    Δ₄ = inward(4, Δ₃)
    return Δ₁, Δ₂, Δ₃, Δ₄
end

# floatmin gives zero agreement to differences whose products underflow
@inline function consecutive_agreement(WCT, a, b, c)
    φφ = newton_div(WCT, 4 * a * b * b * c, (a^2 + b^2) * (b^2 + c^2) + floatmin(typeof(a)))
    return ifelse((a * b > 0) & (b * c > 0), φφ^2, zero(a))
end

@inline minmod(a, b) = max_fast(min_fast(a, b), zero(a)) + min_fast(max_fast(a, b), zero(a))

# Ghost slope and curvature beyond the boundary on the side s of the stencil `S`, whose `n` nodes past the face have data
@inline function ghost_extrapolation(WCT, S, ΔS, n, nᵒ, s, ::HierarchicalExtrapolation, ::Val{N}) where N
    Δ₁, Δ₂, Δ₃, Δ₄ = inward_differences(ΔS, n, nᵒ, s, Val(N))
    G₁ = consecutive_agreement(WCT, Δ₁, Δ₂, Δ₃)
    G₂ = consecutive_agreement(WCT, Δ₁ - Δ₂, Δ₂ - Δ₃, Δ₃ - Δ₄)
    return (G₂ + (1 - G₂) * G₁) * Δ₁, G₂ * (Δ₁ - Δ₂)
end

# With stored wall values: cᵇ is the last node with data and cʷ, the wall value, the node beyond it
@inline function ghost_extrapolation(WCT, S, ΔS, n, nᵒ, s, ::TransportedWallValue, ::Val{N}) where N
    Δ₁, Δ₂, Δ₃, _ = inward_differences(ΔS, n, nᵒ, s, Val(N))
    a  = ifelse(s == 1, N + 1, N)
    cᵇ = offset_entry(S, a, n,     s, Val(N))
    cʷ = offset_entry(S, a, n + 1, s, Val(N))
    Δʷ = 2s * (cʷ - cᵇ)
    K  = 3//2 * (Δʷ - Δ₁)
    G  = consecutive_agreement(WCT, K, Δ₁ - Δ₂, Δ₂ - Δ₃)
    return G * Δ₁ + (1 - G) * minmod(Δʷ, 2Δ₁), G * K
end

@inline weight_computation(::WENO{N, FT, WCT}) where {N, FT, WCT} = WCT
@inline weight_computation(::WENO{N, FT, Nothing}) where {N, FT} = BackendOptimizedDivision
@inline weight_computation(scheme) = BackendOptimizedDivision

# Sᵘ, the stencil reversed for a RightBias, has its node N upwind of the face; the `D` differences downwind of the face are
# N - 2 for an upwind-biased stencil and N - 1 for a centered one. Returns the upwind node, the ghosted differences of the
# stencil and whether any node is a ghost.
@inline function ghosted_differences(scheme::AbstractAdvectionScheme{N}, S, ΔS, nᴸ, nᴿ, bias, upwind_extrapolation, downwind_extrapolation, ::Val{D}) where {N, D}
    left = bias === LeftBias
    Sᵘ  = ifelse(left, S, reverse(S))
    ΔSᵘ = ifelse(left, ΔS, .-reverse(ΔS))
    nᵘ  = ifelse(left, nᴸ, nᴿ)
    nᵈ  = ifelse(left, nᴿ, nᴸ)

    WCT = weight_computation(scheme)
    σᵘ, κᵘ = ghost_extrapolation(WCT, Sᵘ, ΔSᵘ, nᵘ, nᵈ, -1, upwind_extrapolation, Val(N))
    σᵈ, κᵈ = ghost_extrapolation(WCT, Sᵘ, ΔSᵘ, nᵈ, nᵘ, +1, downwind_extrapolation, Val(N))

    Δᵘ = ntuple(Val(N - 1)) do m
        @inline
        @inbounds ifelse(m ≤ nᵘ, ΔSᵘ[N - m], σᵘ + (m - nᵘ) * κᵘ)
    end

    Δᵈ = ntuple(Val(D)) do m
        @inline
        @inbounds ifelse(m ≤ nᵈ, ΔSᵘ[N + m], σᵈ + (m - nᵈ) * κᵈ)
    end

    ghosted = (nᵘ < N - 1) | (nᵈ < D)

    return @inbounds Sᵘ[N], (reverse(Δᵘ)..., ΔSᵘ[N], Δᵈ...), ghosted
end

# Only the upwind side reads the stored wall values; downwind ghosts extrapolate the interior
@inline function ghost_cell_reconstruction(weno::WENO{N}, S, ΔS, nᴸ, nᴿ, bias, δˢ, extrapolation) where N
    ψ₀, δ, ghosted = ghosted_differences(weno, S, ΔS, nᴸ, nᴿ, bias, extrapolation, HierarchicalExtrapolation(), Val(N - 2))
    ω = biased_weno_weights(weight_differences(ghosted, δ, δˢ), weno)
    return weno_reconstruction(weno, ψ₀, δ, ω)
end

@inline function ghost_cell_reconstruction(scheme::UpwindBiased{N}, S, ΔS, nᴸ, nᴿ, bias, δˢ, extrapolation) where N
    ψ₀, δ, _ = ghosted_differences(scheme, S, ΔS, nᴸ, nᴿ, bias, extrapolation, HierarchicalExtrapolation(), Val(N - 2))
    return upwind_reconstruction(scheme, ψ₀, δ)
end

# A centered stencil has no upwind side: both sides read their own wall value
@inline function ghost_cell_reconstruction(scheme::Centered{N}, S, ΔS, nᴸ, nᴿ, bias, δˢ, extrapolation) where N
    ψ₀, δ, _ = ghosted_differences(scheme, S, ΔS, nᴸ, nᴿ, LeftBias, extrapolation, extrapolation, Val(N - 1))
    return centered_reconstruction(scheme, ψ₀, δ)
end

# Two-node stencils: the average, or the upwind node
@inline ghost_cell_reconstruction(::Centered{1}, ψᴸ, ψᴿ, bias) = (ψᴸ + ψᴿ) / 2
@inline ghost_cell_reconstruction(::UpwindBiased{1}, ψᴸ, ψᴿ, bias) = ifelse(bias === LeftBias, ψᴸ, ψᴿ)

@inline weight_differences(ghosted, δ,  ::Nothing) = δ
@inline weight_differences(ghosted, δ, δˢ::NTuple) = ifelse(ghosted, δ, δˢ)
@inline weight_differences(ghosted, δ, δˢ::Tuple{Tuple, Tuple}) = ifelse(ghosted, (δ, δ), δˢ)

# A wall face takes the operand with data, as the conditional ℑ, and zero where neither has
@inline function wall_face_value(i, j, k, grid, d, ℓx, ℓy, ℓz, ψᴸ, ψᴿ, ψ̂)
    di, dj, dk = unit_vector(d)
    dᴸ = node_has_data(i - di, j - dj, k - dk, grid, d, ℓx, ℓy, ℓz)
    dᴿ = node_has_data(i, j, k, grid, d, ℓx, ℓy, ℓz)
    return ifelse(dᴸ & dᴿ, ψ̂, ifelse(dᴸ, ψᴸ, ifelse(dᴿ, ψᴿ, zero(ψ̂))))
end

# Reconstruction at the face between the nodes `(i, j, k) - e` and `(i, j, k)`; `S` in index order: node N is L, behind the
# face, and node N + 1 is R = (i, j, k)
@inline function ghost_cell_interpolate(i, j, k, grid, scheme::AbstractAdvectionScheme{N}, bias, d, ℓx, ℓy, ℓz, stencil, ψ, args...) where N
    e = unit_vector(d)
    di, dj, dk = e

    S  = index_stencil(i, j, k, grid, scheme, e, ψ, args...)
    ΔS = node_differences(S)
    nᴸ = boundary_distance(i - di, j - dj, k - dk, grid, d, ℓx, ℓy, ℓz, -1, Val(N))
    nᴿ = boundary_distance(i,      j,      k,      grid, d, ℓx, ℓy, ℓz, +1, Val(N))
    δˢ = smoothness_differences(i, j, k, grid, scheme, bias, e, stencil, args...)
    extrapolation = extrapolation_along(scheme.extrapolation, grid, d)
    ψ̂  = ghost_cell_reconstruction(scheme, S, ΔS, nᴸ, nᴿ, bias, δˢ, extrapolation)

    return @inbounds wall_face_value(i, j, k, grid, d, ℓx, ℓy, ℓz, S[N], S[N + 1], ψ̂)
end

@inline function ghost_cell_interpolate(i, j, k, grid, scheme::Union{Centered{1}, UpwindBiased{1}}, bias, d, ℓx, ℓy, ℓz, stencil, ψ, args...)
    di, dj, dk = unit_vector(d)
    ψᴸ = getvalue(ψ, i - di, j - dj, k - dk, grid, args...)
    ψᴿ = getvalue(ψ, i, j, k, grid, args...)
    return wall_face_value(i, j, k, grid, d, ℓx, ℓy, ℓz, ψᴸ, ψᴿ, ghost_cell_reconstruction(scheme, ψᴸ, ψᴿ, bias))
end

# Reconstructions named by the location of their result, as the interpolations: along ξ the operands sit at the flipped
# location, and the reconstruction point lies between the operand nodes (i, j, k) - e and (i, j, k)
operand_location(ℓ) = ℓ === :ᶜ ? :(Center()) : :(Face())
flipped_location(ℓ) = ℓ === :ᶜ ? :(Face()) : :(Center())

for (dir, ξ) in enumerate((:x, :y, :z)), ℓx in (:ᶜ, :ᶠ), ℓy in (:ᶜ, :ᶠ), ℓz in (:ᶜ, :ᶠ)
    ℓ = (ℓx, ℓy, ℓz)
    e = unit_vector(Val(dir))
    d = Val(dir)
    biased    = Symbol(:biased_interpolate_, ξ, ℓ...)
    symmetric = Symbol(:symmetric_interpolate_, ξ, ℓ...)
    operand   = ntuple(n -> n == dir ? flipped_location(ℓ[n]) : operand_location(ℓ[n]), 3)
    node      = ℓ[dir] === :ᶠ ? :((i, j, k)) : :((i, j, k) .+ $e)

    @eval begin
        @inline $biased(i, j, k, grid, scheme::AbstractAdvectionScheme, bias, ψ, args...) = ghost_cell_interpolate($node..., grid, scheme, bias, $d, $(operand...), DefaultStencil(), ψ, args...)
        @inline $biased(i, j, k, grid, scheme::AbstractAdvectionScheme, bias, ψ, stencil::AS, args...) = ghost_cell_interpolate($node..., grid, scheme, bias, $d, $(operand...), stencil, ψ, args...)

        # A centered scheme has no bias; an upwind scheme interpolates symmetrically with its advecting velocity scheme
        @inline $symmetric(i, j, k, grid, scheme::AbstractCenteredAdvectionScheme, args...) = $biased(i, j, k, grid, scheme, LeftBias, args...)
        @inline $symmetric(i, j, k, grid, scheme::UpwindScheme, args...) = $symmetric(i, j, k, grid, scheme.advecting_velocity_scheme, args...)
    end
end
