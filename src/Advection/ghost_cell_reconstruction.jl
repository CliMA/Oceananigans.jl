using Oceananigans.ImmersedBoundaries: inactive_node

"""
    GhostCells([FT=Oceananigans.defaults.FloatType;] curvature_weight = 100, monotone = true)

Boundary scheme of a `WENO` reconstruction whose stencil reaches inactive cells. Every inactive cell of the stencil is replaced by a ghost value,
and the full-order reconstruction runs on the completed stencil. The ghost value blends the mirror image of the active run across the boundary with
the quadratic extrapolation of its three cells closest to the boundary, `c₀, c₁, c₂`, as `(1 - θ) * mirror + θ * extrapolation` with

```math
θ = δ⁴ / (δ⁴ + w κ⁴), \\quad δ = c₁ - c₀, \\quad κ = c₂ - 2c₁ + c₀,
```

so that smooth data are extrapolated, while jumps and data with vanishing gradient at the boundary are mirrored.

- `curvature_weight`: the weight `w` of the curvature in `θ`.
- `monotone`: where the reconstruction uses ghost values, bound it between the upwind and downwind cells and within twice the upwind gradient of the upwind cell, 
  as in [SureshHuynh97](@citet).
"""
struct GhostCells{S, FT}
    scheme :: S
    curvature_weight :: FT
    monotone :: Bool
end

GhostCells(FT::DataType = Oceananigans.defaults.FloatType; curvature_weight = 100, monotone = true) = GhostCells(nothing, convert(FT, curvature_weight), monotone)

Base.summary(scheme::GhostCells) = string("GhostCells(curvature_weight=", scheme.curvature_weight, ", monotone=", scheme.monotone, ")")

const GhostCellWENO = WENO{<:Any, <:Any, <:Any, <:Any, <:Any, <:GhostCells}

Adapt.adapt_structure(to, scheme::GhostCells) = GhostCells(Adapt.adapt(to, scheme.scheme), scheme.curvature_weight, scheme.monotone)
Oceananigans.Architectures.on_architecture(to, scheme::GhostCells) = GhostCells(on_architecture(to, scheme.scheme), scheme.curvature_weight, scheme.monotone)

@inline function extrapolation_weight(WCT, c₀, c₁, c₂, w)
    δ = c₁ - c₀
    κ = c₂ - 2c₁ + c₀
    δ² = δ * δ
    κ² = κ * κ
    δ⁴ = δ² * δ²
    Σ = δ⁴ + w * κ² * κ²
    return δ, κ, newton_div(WCT, δ⁴, ifelse(Σ > 0, Σ, one(Σ)))
end

@inline ghost_value(mirror, c₀, δ, κ, θ, distance) = mirror + θ * (c₀ - distance * δ + (distance * (distance + 1) ÷ 2) * κ - mirror)

@inline function monotone_bound(ψ̂, U, D, UU)
    Uᴸ = U + 2 * (U - UU)
    ψᵐᵃˣ = ifelse(D > Uᴸ, D, Uᴸ)
    ψᵐⁱⁿ = ifelse(D > Uᴸ, Uᴸ, D)
    return clamp(ψ̂, ifelse(U < ψᵐᵃˣ, U, ψᵐᵃˣ), ifelse(U > ψᵐⁱⁿ, U, ψᵐⁱⁿ))
end

@inline upwind_stencil_length(N) = max(2N - 1, N + 2)

function clamped_to_run_start(N, q)
    if q < 1 
        return clamped_to_run_start(N, 1)
    elseif q ≥ N 
        return :(S[$q])
    else
        return :(ifelse(Aᵃ[$(N - q)], S[$q], $(clamped_to_run_start(N, q + 1))))
    end
end

function clamped_to_run_end(N, q)
    if q > upwind_stencil_length(N) 
        return clamped_to_run_end(N, upwind_stencil_length(N))
    elseif q ≤ N 
        return :(S[$q])
    else
        return :(ifelse(Aᵇ[$(q - N)], S[$q], $(clamped_to_run_end(N, q - 1))))
    end
end

function run_start_anchor(N, k)
    anchor = clamped_to_run_end(N, N + k)
    for run in 1:N-2
        anchor = :(ifelse(Aᵃ[$run], $(clamped_to_run_end(N, N - run + k)), $anchor))
    end
    return anchor
end

function run_end_anchor(N, k)
    anchor = clamped_to_run_start(N, N - k)
    for run in 1:upwind_stencil_length(N)-N-1
        anchor = :(ifelse(Aᵇ[$run], $(clamped_to_run_start(N, N + run - k)), $anchor))
    end
    return anchor
end

function completed_cell(N, q)
    q == N && return :(S[$N])
    A, g, depth = if q < N
        (:Aᵃ, :gᵃ, N - q) 
    else
        (:Aᵇ, :gᵇ, q - N)
    end
    ghost = :($g[$depth])
    for run in 1:depth-1
        ghost = :(ifelse($A[$run], $g[$(depth - run)], $ghost))
    end
    return :(ifelse($A[$depth], S[$q], $ghost))
end

for N in advection_buffers[2:end]
    @eval begin
        @inline run_start_anchors(S, Aᵃ, Aᵇ, ::Val{$N}) = @inbounds $(Expr(:tuple, (run_start_anchor(N, k) for k in 0:max(2, N - 2))...))
        @inline   run_end_anchors(S, Aᵃ, Aᵇ, ::Val{$N}) = @inbounds $(Expr(:tuple, (  run_end_anchor(N, k) for k in 0:max(2, N - 2))...))
        @inline completed_stencil(S, Aᵃ, Aᵇ, gᵃ, gᵇ, ::Val{$N}) = @inbounds $(Expr(:tuple, (completed_cell(N, q) for q in 1:2N-1)...))
    end
end

@inline function ghost_cell_reconstruction(scheme::GhostCells{<:WENO{N, <:Any, WCT}}, bias, S, Aᵃ, Aᵇ, δˢ) where {N, WCT}
    cᵃ = run_start_anchors(S, Aᵃ, Aᵇ, Val(N)) # Values nearest the (-) boundary for each stencil
    cᵇ =   run_end_anchors(S, Aᵃ, Aᵇ, Val(N)) # Values nearest the (+) boundary for each stencil

    δᵃ, κᵃ, θᵃ = @inbounds extrapolation_weight(WCT, cᵃ[1], cᵃ[2], cᵃ[3], scheme.curvature_weight)
    δᵇ, κᵇ, θᵇ = @inbounds extrapolation_weight(WCT, cᵇ[1], cᵇ[2], cᵇ[3], scheme.curvature_weight)

    gᵃ = ntuple(d -> @inbounds(ghost_value(cᵃ[d], cᵃ[1], δᵃ, κᵃ, θᵃ, d)), Val(N - 1)) # ghost values inside the (-) boundary
    gᵇ = ntuple(d -> @inbounds(ghost_value(cᵇ[d], cᵇ[1], δᵇ, κᵇ, θᵇ, d)), Val(N - 1)) # ghost values inside the (+) boundary
    S̃  = completed_stencil(S, Aᵃ, Aᵇ, gᵃ, gᵇ, Val(N)) # Choose between ``real'' interior values (S) and ghost values (in immersed or boundary nodes)

    ghosted = @inbounds !(Aᵃ[N - 1] & Aᵇ[N - 1])

    δ = weno_differences(scheme.scheme, S̃)
    ω = biased_weno_weights(smoothness(ghosted, δ, δˢ), nothing, scheme.scheme)
    ψ̂ = weno_reconstruction(scheme.scheme, weno_anchor(scheme.scheme, S̃), δ, ω)
    ψ̃ = @inbounds monotone_bound(ψ̂, S[N], S̃[N + 1], S̃[N - 1])

    return ifelse(scheme.monotone & ghosted, ψ̃, ψ̂)
end

# Stencils with ghost points do not obey the `VelocityStencil` and `FunctionStencil`, 
# they always treat the weno reconstruction as if there was a `DefaultStencil` 
@inline smoothness(ghosted, δ, ::Nothing) = δ
@inline smoothness(ghosted, δ, δˢ::NTuple) = ifelse(ghosted, δ, δˢ)
@inline smoothness(ghosted, δ, δˢ::Tuple{Tuple, Tuple}) = ifelse(ghosted, (δ, δ), δˢ)

@inline active_node(i, j, k, grid, ℓx, ℓy, ℓz) = !inactive_node(i, j, k, grid, ℓx, ℓy, ℓz)

@inline function ghost_cell_interpolate(i, j, k, grid, scheme::GhostCells{<:WENO{N}}, bias, e, ℓx, ℓy, ℓz, δˢ, ψ, args...) where N
    S = upwind_stencil(i, j, k, grid, scheme.scheme, bias, e, ψ, args...)
    A = upwind_stencil(i, j, k, grid, scheme.scheme, bias, e, active_node, ℓx, ℓy, ℓz)

    Aᵃ = accumulate(&, ntuple(@inline(run -> @inbounds(A[N - run])), Val(N - 1)))
    Aᵇ = accumulate(&, ntuple(@inline(run -> @inbounds(A[N + run])), Val(upwind_stencil_length(N) - N)))

    return ghost_cell_reconstruction(scheme, bias, S, Aᵃ, Aᵇ, δˢ)
end

for (dir, ξ) in enumerate((:x, :y, :z))
    e = ntuple(d -> Int(d == dir), 3) # The direction
    ℓ = ntuple(d -> d == dir ? :(Face()) : :(Center()), 3) # Face location in direction dir
    biased_face   = Symbol(:biased_interpolate_, ξ, ntuple(d -> d == dir ? :ᶠ : :ᵃ, 3)...)
    biased_center = Symbol(:biased_interpolate_, ξ, ntuple(d -> d == dir ? :ᶜ : :ᵃ, 3)...)

    @eval begin
        @inline   $biased_face(i, j, k, grid, scheme::GhostCells, bias, ψ, args...) = ghost_cell_interpolate(i, j, k, grid, scheme, bias, $e, Center(), Center(), Center(), nothing, ψ, args...)
        @inline $biased_center(i, j, k, grid, scheme::GhostCells, bias, ψ, args...) = ghost_cell_interpolate(((i, j, k) .+ $e)..., grid, scheme, bias, $e, $(ℓ...), nothing, ψ, args...)

        @inline function $biased_face(i, j, k, grid, scheme::GhostCells, bias, ψ, stencil::AbstractSmoothnessStencil, args...)
            δˢ = smoothness_differences(i, j, k, grid, scheme.scheme, bias, $e, stencil, args...)
            return ghost_cell_interpolate(i, j, k, grid, scheme, bias, $e, Center(), Center(), Center(), δˢ, ψ, args...)
        end

        @inline function $biased_center(i, j, k, grid, scheme::GhostCells, bias, ψ, stencil::AbstractSmoothnessStencil, args...)
            δˢ = smoothness_differences(((i, j, k) .+ $e)..., grid, scheme.scheme, bias, $e, stencil, args...)
            return ghost_cell_interpolate(((i, j, k) .+ $e)..., grid, scheme, bias, $e, $(ℓ...), δˢ, ψ, args...)
        end
    end

    for location in (:ᶜ, :ᶠ)
        symmetric = Symbol(:symmetric_interpolate_, ξ, ntuple(d -> d == dir ? location : :ᵃ, 3)...)
        @eval @inline $symmetric(i, j, k, grid, ::GhostCells{<:Any, FT}, args...) where FT = $symmetric(i, j, k, grid, Centered{1, FT}(nothing, ExplicitTimeDiscretization()), args...)
    end
end
