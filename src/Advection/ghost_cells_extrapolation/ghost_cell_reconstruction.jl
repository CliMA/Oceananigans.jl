using Base.FastMath: min_fast, max_fast
using KernelAbstractions.Extras.LoopInfo: @unroll
using Oceananigans.Grids: peripheral_node

"""
    HierarchicalExtrapolation()

Ghosts that continue the nodes before the boundary with a polynomial of degree up to two:

                        boundary
        │   c₋₁   │    c₀   ┃    c₁   │    c₂   │    c₃   │    c₄   │    c₅   │
                            ┃         Δ₁        Δ₂        Δ₃        Δ₄        │
               ghosts                    nodes before the boundary

```math
    \\begin{align}
    φ(a, b) & = 2ab / (a² + b²) \\\\
    G₁  & = [φ(Δ₁, Δ₂) ⋅ φ(Δ₂, Δ₃)]² \\\\
    G₂  & = [φ(κ₁, κ₂) ⋅ φ(κ₂, κ₃)]² \\\\
    σ   & = [G₂ + (1 - G₂) G₁] Δ₁ \\\\
    κ   & = G₂ κ₁ \\\\
    c₀  & = c₁ - σ - κ \\\\
    c₋₁ & = c₁ - 2σ - 3κ \\\\
    c₋₂ & = c₁ - 3σ - 6κ
    \\end{align}
```

with `Δⱼ = cⱼ₊₁ - cⱼ`, `κⱼ = Δⱼ - Δⱼ₊₁`, `φ = 0` unless `a` and `b` share a sign, and in general the g-th ghost `c₁₋g = c₁ - g σ - g (g + 1) κ / 2`.
Linear data, and quadratic data on five active nodes, are continued exactly, and a front next to the boundary (`G₁ = G₂ = 0`) has constant ghosts.
"""
struct HierarchicalExtrapolation end

"""
    TransportedWallValue()

Ghosts built from a wall value `cʷ`, stored in the solid node beyond each wall cell and transported as

```math
V ∂ₜcʷ = Az max(-w, 0) (ĉ - cʷ) + Ax [u⁻ (ĉ⁻ - cʷ) - u⁺ (ĉ⁺ - cʷ)]
```

                       interior fluid
          ┌────────────┬────────────┬────────────┐
          │            │            │            │
          │            │     c₂     │            │
          │            │            │            │
          ├────────────┼──── ↓ w ───┼────────────┤
          │            │            │            │
          │            →     c₁     →            │     wall cells: the fluid cells next to the wall
          │           u⁻           u⁺            │
        ━━━━━━━━━━━━━━━━━━━━ ĉ ━━━━━━━━━━━━━━━━━━━━━   wall, where c₁, c₂, … extrapolate to ĉ
               cʷ⁻    ĉ⁻     cʷ    ĉ⁺    cʷ⁺           wall values, stored in the solid nodes beyond it

with `V`, `Ax` and `Az` the volume and face areas of the wall cell of `c₁`, `w` the velocity at its top face, so that
only the fluid entering from the interior relaxes `cʷ` towards `ĉ = c₁ - σʰ/2`, the wall value of the
`HierarchicalExtrapolation` slope `σʰ` of the line `c₁, c₂, …`, and `u⁻`, `u⁺` the velocities at its faces along the
wall, which carry the upwind wall values `ĉ⁻`, `ĉ⁺`. On other walls the velocities normal to and along the wall take the
places of `w` and `u`, and a wall with two directions along it sums the last term over both.

Upwind of a face, with `Δⱼ = cⱼ₊₁ - cⱼ`, `κⱼ = Δⱼ - Δⱼ₊₁` and `φ` as in `HierarchicalExtrapolation`, the ghosts beyond
the wall are

```math
    \\begin{align}
    Δʷ  & = 2 (c₁ - cʷ) \\\\
    K   & = 3/2 (Δʷ - Δ₁) \\\\
    G   & = [φ(K, κ₁) ⋅ φ(κ₁, κ₂)]² \\\\
    σ   & = G Δ₁ + (1 - G) \\, \\mathrm{minmod}(Δʷ, 2Δ₁) \\\\
    κ   & = G K \\\\
    c₀  & = c₁ - σ - κ \\\\
    c₋₁ & = c₁ - 2σ - 3κ \\\\
    c₋₂ & = c₁ - 3σ - 6κ
    \\end{align}
```

with `Δʷ` the slope towards `cʷ`, half a node beyond `c₁`, `K` the curvature of the quadratic through `cʷ`, `c₁` and
`c₂`, and `G`, which is 1 when `K`, `κ₁` and `κ₂` are equal and falls to 0 as they differ: the ghosts follow the
quadratic where the line is smooth and the slope towards `cʷ`, limited by `2Δ₁`, otherwise. Downwind ghosts, and those of
fields without stored wall values, extrapolate as `HierarchicalExtrapolation`.

On the walls of the domain where the tracer has a `Value`, `Gradient` or `Mixed` boundary condition, `cʷ = (c₀ + c₁) / 2`
comes from the halo `c₀` that the condition fills.

!!!NOTE:
    This extraplation technique is used only for tracers and assumes `(Center, Center, Center)` variables.
"""
struct TransportedWallValue{I, D}
    immersed_walls :: I
      domain_walls :: D
end

TransportedWallValue() = TransportedWallValue(nothing, nothing)

"""
    GhostCells([FT=Oceananigans.defaults.FloatType]; extrapolation = TransportedWallValue())

Boundary scheme of a `WENO` reconstruction of order 5 or higher, `WENO(order = 7, boundary_scheme = GhostCells())`, which replaces the
peripheral nodes of a stencil with ghosts that `extrapolation`, `TransportedWallValue()` or `HierarchicalExtrapolation()`, builds in index
coordinates. Data linear in the grid index are reconstructed exactly next to boundaries, and stencils without peripheral nodes as by `WENO`.
"""
struct GhostCells{S, E}
           scheme :: S
    extrapolation :: E
end

GhostCells(FT::DataType = Oceananigans.defaults.FloatType; extrapolation = TransportedWallValue()) = GhostCells(nothing, extrapolation)

Base.summary(::GhostCells) = "GhostCells()"

const GhostCellWENO = WENO{<:Any, <:Any, <:Any, <:Any, <:Any, <:GhostCells}

Adapt.adapt_structure(to, scheme::GhostCells) = GhostCells(Adapt.adapt(to, scheme.scheme), Adapt.adapt(to, scheme.extrapolation))
Oceananigans.Architectures.on_architecture(to, scheme::GhostCells) = GhostCells(on_architecture(to, scheme.scheme), on_architecture(to, scheme.extrapolation))

#####
##### Reconstruction
#####

@inline function boundary_distance(i, j, k, grid, ℓx, ℓy, ℓz, d, s, ::Val{N}) where N
    di, dj, dk = unit_vector(d)
    n = 0
    reached = false
    @unroll for m in 1:N-1
        reached |= peripheral_node(i + s * m * di, j + s * m * dj, k + s * m * dk, grid, ℓx, ℓy, ℓz)
        n += !reached
    end
    return n
end

@inline node_differences(S::NTuple{M}) where M = ntuple(n -> @inbounds(S[n + 1] - S[n]), Val(M - 1))

# ΔS[N + s max(n, 0)], unrolled to avoid a run-time tuple index
@inline function boundary_difference(ΔS, n, s, ::Val{N}) where N
    Δ = @inbounds ΔS[N]
    @unroll for m in 1:N-1
        Δ = @inbounds ifelse(m ≤ n, ΔS[N + m * s], Δ)
    end
    return Δ
end

# ΔS[N + s (n + 1 - j)], or Δ past the boundary on the other side
@inline function inward_difference(ΔS, Δ, n, nᵒ, s, j, ::Val{N}) where N
    own   = boundary_difference(ΔS, n + 1 - j, s, Val(N))
    other = boundary_difference(ΔS, j - n - 1, -s, Val(N))
    return ifelse(j ≤ n + 1, own, ifelse(j - n - 1 ≤ nᵒ, other, Δ))
end

@inline function inward_differences(ΔS, n, nᵒ, s, ::Val{N}) where N
    Δ₁ = inward_difference(ΔS, zero(eltype(ΔS)), n, nᵒ, s, 1, Val(N))
    Δ₂ = inward_difference(ΔS, Δ₁, n, nᵒ, s, 2, Val(N))
    Δ₃ = inward_difference(ΔS, Δ₂, n, nᵒ, s, 3, Val(N))
    Δ₄ = inward_difference(ΔS, Δ₃, n, nᵒ, s, 4, Val(N))
    return Δ₁, Δ₂, Δ₃, Δ₄
end

# floatmin gives zero agreement to differences whose products underflow
@inline function consecutive_agreement(WCT, a, b, c)
    φφ = newton_div(WCT, 4 * a * b * b * c, (a^2 + b^2) * (b^2 + c^2) + floatmin(typeof(a)))
    return ifelse((a * b > 0) & (b * c > 0), φφ^2, zero(a))
end

@inline function boundary_node(S, n, s, ::Val{N}) where N
    a  = ifelse(s == 1, N + 1, N)
    cᵇ = @inbounds S[a]
    @unroll for m in 1:N-2
        cᵇ = @inbounds ifelse(m ≤ n, S[a + s * m], cᵇ)
    end
    return cᵇ
end

@inline minmod(a, b) = max_fast(min_fast(a, b), zero(a)) + min_fast(max_fast(a, b), zero(a))

# Pure extrapolation (no stored wall values)
@inline function ghost_extrapolation(WCT, S, ΔS, n, nᵒ, s, ::Nothing, ::Val{N}) where N
    Δ₁, Δ₂, Δ₃, Δ₄ = inward_differences(ΔS, n, nᵒ, s, Val(N))
    G₁ = consecutive_agreement(WCT, Δ₁, Δ₂, Δ₃)
    G₂ = consecutive_agreement(WCT, Δ₁ - Δ₂, Δ₂ - Δ₃, Δ₃ - Δ₄)
    return (G₂ + (1 - G₂) * G₁) * Δ₁, G₂ * (Δ₁ - Δ₂)
end

@inline function ghost_extrapolation(WCT, S, ΔS, n, nᵒ, s, cʷ, ::Val{N}) where N
    Δ₁, Δ₂, Δ₃, _ = inward_differences(ΔS, n, nᵒ, s, Val(N))
    cᵇ = boundary_node(S, n, s, Val(N))
    Δʷ = 2s * (cʷ - cᵇ)
    K  = 3//2 * (Δʷ - Δ₁)
    G  = consecutive_agreement(WCT, K, Δ₁ - Δ₂, Δ₂ - Δ₃)
    return G * Δ₁ + (1 - G) * minmod(Δʷ, 2Δ₁), G * K
end

# Sᵘ, the stencil reversed for a RightBias, has its node N upwind of the face
@inline function ghost_cell_reconstruction(weno::WENO{N, FT, WCT}, S, ΔS, nᴸ, nᴿ, bias, δˢ, cʷ) where {N, FT, WCT}
    left = bias === LeftBias
    Sᵘ  = ifelse(left, S, reverse(S))
    ΔSᵘ = ifelse(left, ΔS, .-reverse(ΔS))
    nᵘ  = ifelse(left, nᴸ, nᴿ)
    nᵈ  = ifelse(left, nᴿ, nᴸ)

    σᵘ, κᵘ = ghost_extrapolation(WCT, Sᵘ, ΔSᵘ, nᵘ, nᵈ, -1, cʷ, Val(N))
    σᵈ, κᵈ = ghost_extrapolation(WCT, Sᵘ, ΔSᵘ, nᵈ, nᵘ, +1, nothing, Val(N))

    Δᵘ = ntuple(Val(N - 1)) do m
        @inline
        @inbounds ifelse(m ≤ nᵘ, ΔSᵘ[N - m], σᵘ + (m - nᵘ) * κᵘ)
    end

    Δᵈ = ntuple(Val(N - 2)) do m
        @inline
        @inbounds ifelse(m ≤ nᵈ, ΔSᵘ[N + m], σᵈ + (m - nᵈ) * κᵈ)
    end

    δ = @inbounds (reverse(Δᵘ)..., ΔSᵘ[N], Δᵈ...)

    ghosted = (nᵘ < N - 1) | (nᵈ < N - 2)
    ω = biased_weno_weights(smoothness(ghosted, δ, δˢ), nothing, weno)

    return @inbounds weno_reconstruction(weno, Sᵘ[N], δ, ω)
end

@inline smoothness(ghosted, δ,  ::Nothing) = δ
@inline smoothness(ghosted, δ, δˢ::NTuple) = ifelse(ghosted, δ, δˢ)
@inline smoothness(ghosted, δ, δˢ::Tuple{Tuple, Tuple}) = ifelse(ghosted, (δ, δ), δˢ)

@inline function ghost_cell_interpolate(i, j, k, grid, scheme::GhostCells{<:WENO{N}}, bias, d, ℓx, ℓy, ℓz, stencil, ψ, args...) where N
    weno = scheme.scheme
    e = unit_vector(d)
    di, dj, dk = e

    # S in index order: node N is L, behind the face, and node N + 1 is R = (i, j, k)
    S  = upwind_stencil(i, j, k, grid, weno, LeftBias, e, ψ, args...)
    ΔS = node_differences(S)
    nᴸ = boundary_distance(i - di, j - dj, k - dk, grid, ℓx, ℓy, ℓz, d, -1, Val(N))
    nᴿ = boundary_distance(i,      j,      k,      grid, ℓx, ℓy, ℓz, d, +1, Val(N))
    δˢ = smoothness_differences(i, j, k, grid, weno, bias, e, stencil, args...)
    cʷ = stored_wall_value(scheme.extrapolation, i, j, k, grid, d, nᴸ, nᴿ, bias, ψ, Val(N))

    return ghost_cell_reconstruction(weno, S, ΔS, nᴸ, nᴿ, bias, δˢ, cʷ)
end

struct LocatedField{F, L}
       field :: F
    location :: L
end

@inline Base.getindex(ψ::LocatedField, i, j, k) = @inbounds ψ.field[i, j, k]

@inline stencil_location(ψ, ℓx, ℓy, ℓz)               = (ℓx, ℓy, ℓz)
@inline stencil_location(ψ::LocatedField, ℓx, ℓy, ℓz) = ψ.location
@inline stencil_location(::typeof(ζ₃ᶠᶠᶜ), ℓx, ℓy, ℓz) = (Face(), Face(), Center())

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
