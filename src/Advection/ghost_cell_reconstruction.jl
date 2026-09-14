using Oceananigans.ImmersedBoundaries: inactive_node

"""
    GhostCells([FT=Oceananigans.defaults.FloatType;] curvature_weight = 100, monotone = true)

Boundary scheme of a `WENO` reconstruction whose stencil reaches inactive cells. Every inactive cell of the stencil is
replaced by a ghost value, and the full-order reconstruction runs on the completed stencil. The ghost value blends the
mirror image of the active run across the boundary with the quadratic extrapolation of its three cells closest to the
boundary, `c₀, c₁, c₂`, as `(1 - θ) * mirror + θ * extrapolation` with

```math
θ = δ⁴ / (δ⁴ + w κ⁴), \\quad δ = c₁ - c₀, \\quad κ = c₂ - 2c₁ + c₀,
```

so that smooth data are extrapolated, while jumps and data with vanishing gradient at the boundary are mirrored.

- `curvature_weight`: the weight `w` of the curvature in `θ`.
- `monotone`: where the reconstruction uses ghost values, bound it between the upwind and downwind cells and within
  twice the upwind gradient of the upwind cell, as in [SureshHuynh97](@citet).
"""
struct GhostCells{S, FT}
    scheme :: S
    curvature_weight :: FT
    monotone :: Bool
end

GhostCells(FT::DataType = Oceananigans.defaults.FloatType; curvature_weight = 100, monotone = true) =
    GhostCells(nothing, convert(FT, curvature_weight), monotone)

Base.summary(scheme::GhostCells) = string("GhostCells(curvature_weight=", scheme.curvature_weight, ", monotone=", scheme.monotone, ")")

const GhostCellWENO = WENO{<:Any, <:Any, <:Any, <:Any, <:Any, <:GhostCells}

@inline function extension_weight(c₀, c₁, c₂, w)
    δ = c₁ - c₀
    κ = c₂ - 2c₁ + c₀
    δ⁴ = δ^4
    Σ = δ⁴ + w * κ^4
    return δ, κ, ifelse(Σ > 0, δ⁴ / Σ, zero(Σ))
end

# `m` cells beyond the end `c₀` of the active run, whose `m`-th cell inwards is `mirror`
@inline ghost_value(mirror, c₀, δ, κ, θ, m) = mirror + θ * (c₀ - m * δ + (m * (m + 1) ÷ 2) * κ - mirror)

# length of the run of active cells that starts next to `from` and walks along `step`
function active_run_length(from, step, last)
    from == last && return 0
    return :(A[$(from + step)] * (1 + $(active_run_length(from + step, step, last))))
end

for N in advection_buffers[2:end]
    substencils = Tuple(Symbol(:S, Char(0x2080 + r), Char(0x2080 + N)) for r in 0:N-1)

    completed = map(1:2N) do q
        :(ifelse($q < a, ghost_value(S[clamp(2a - $q - 1, a, b)], S[a], δᵃ, κᵃ, θᵃ, a - $q),
          ifelse($q > b, ghost_value(S[clamp(2b - $q + 1, a, b)], S[b], δᵇ, κᵇ, θᵇ, $q - b), S[$q])))
    end

    @eval @inline function ghost_cell_reconstruction(scheme::GhostCells{<:WENO{$N}}, bias, S, A)
        w = scheme.curvature_weight
        a = ifelse(bias == LeftBias, $N - $(active_run_length(N, -1, 1)),  $(N + 1) - $(active_run_length(N + 1, -1, 1)))
        b = ifelse(bias == LeftBias, $N + $(active_run_length(N, +1, 2N)), $(N + 1) + $(active_run_length(N + 1, +1, 2N)))

        δᵃ, κᵃ, θᵃ = @inbounds extension_weight(S[a], S[min(a + 1, b)], S[min(a + 2, b)], w)
        δᵇ, κᵇ, θᵇ = @inbounds extension_weight(S[b], S[max(b - 1, a)], S[max(b - 2, a)], w)

        G = @inbounds $(Expr(:tuple, completed...))
        ψ = $(Expr(:tuple, (:($s(G, bias)) for s in substencils)...))
        ω = biased_weno_weights(ψ, nothing, scheme.scheme, bias)
        ψ̂ = weno_reconstruction(scheme.scheme, bias, ψ, ω)

        U  = @inbounds ifelse(bias == LeftBias, G[$N],       G[$(N + 1)])
        D  = @inbounds ifelse(bias == LeftBias, G[$(N + 1)], G[$N])
        UU = @inbounds ifelse(bias == LeftBias, G[$(N - 1)], G[$(N + 2)])
        Uᴸ = U + 2 * (U - UU)
        lower = max(min(U, D), min(U, Uᴸ))
        upper = min(max(U, D), max(U, Uᴸ))
        ghosted = ifelse(bias == LeftBias, (a > 1) | (b < $(2N - 1)), (a > 2) | (b < $(2N)))

        return ifelse(scheme.monotone & ghosted, clamp(ψ̂, lower, upper), ψ̂), ghosted
    end
end

for (ξ, dir) in ((:x, 1), (:y, 2), (:z, 3))
    face = Symbol(:biased_interpolate_, ξ, [d == dir ? :ᶠ : :ᵃ for d in 1:3]...)
    cent = Symbol(:biased_interpolate_, ξ, [d == dir ? :ᶜ : :ᵃ for d in 1:3]...)
    ghosted = Symbol(:ghost_cell_interpolate_, ξ)
    shifted = dir == 1 ? (:(i + 1), :j, :k) : dir == 2 ? (:i, :(j + 1), :k) : (:i, :j, :(k + 1))
    staggered = Tuple(d == dir ? :(Face()) : :(Center()) for d in 1:3)

    for N in advection_buffers[2:end], (ψtype, callable) in ((:Any, false), (:Callable, true))
        activity = [ξ == :x ? :(!inactive_node(i + $c, j, k, grid, ℓx, ℓy, ℓz)) :
                    ξ == :y ? :(!inactive_node(i, j + $c, k, grid, ℓx, ℓy, ℓz)) :
                              :(!inactive_node(i, j, k + $c, grid, ℓx, ℓy, ℓz)) for c in -N:N-1]

        @eval @inline function $ghosted(i, j, k, grid, scheme::GhostCells{<:WENO{$N}}, bias, ℓx, ℓy, ℓz, ψ::$ψtype, args...)
            S = @inbounds $(load_weno_stencil(N, ξ, callable))
            A = $(Expr(:tuple, activity...))
            return ghost_cell_reconstruction(scheme, bias, S, A)
        end
    end

    @eval begin
        @inline $face(i, j, k, grid, scheme::GhostCells, bias, ψ, args...) = first($ghosted(i, j, k, grid, scheme, bias, Center(), Center(), Center(), ψ, args...))
        @inline $cent(i, j, k, grid, scheme::GhostCells, bias, ψ, args...) = first($ghosted($(shifted...), grid, scheme, bias, $(staggered...), ψ, args...))

        # away from boundaries the smoothness of the reconstruction is diagnosed by `stencil`, as in `WENO`
        @inline function $face(i, j, k, grid, scheme::GhostCells, bias, ψ, stencil::AbstractSmoothnessStencil, args...)
            ψ̂, ghosted = $ghosted(i, j, k, grid, scheme, bias, Center(), Center(), Center(), ψ, args...)
            return ifelse(ghosted, ψ̂, $face(i, j, k, grid, scheme.scheme, bias, ψ, stencil, args...))
        end

        @inline function $cent(i, j, k, grid, scheme::GhostCells, bias, ψ, stencil::AbstractSmoothnessStencil, args...)
            ψ̂, ghosted = $ghosted($(shifted...), grid, scheme, bias, $(staggered...), ψ, args...)
            return ifelse(ghosted, ψ̂, $cent(i, j, k, grid, scheme.scheme, bias, ψ, stencil, args...))
        end
    end

    for loc in (:ᶜ, :ᶠ)
        symmetric = Symbol(:symmetric_interpolate_, ξ, [d == dir ? loc : :ᵃ for d in 1:3]...)
        @eval @inline $symmetric(i, j, k, grid, ::GhostCells{<:Any, FT}, args...) where FT =
            $symmetric(i, j, k, grid, Centered{1, FT}(nothing, ExplicitTimeDiscretization()), args...)
    end
end
