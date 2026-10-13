using Oceananigans.ImmersedBoundaries: inactive_node

"""
    GhostCells([FT=Oceananigans.defaults.FloatType;] bounded = true)

Boundary scheme of a `WENO` reconstruction whose stencil reaches inactive cells, either inside an immersed boundary
or outside a `Bounded` domain. Every inactive cell of the stencil is replaced by a ghost value and the full-order
reconstruction runs on the completed stencil. The ghost value is the Lagrange extrapolation, along the stencil line,
of the active cells closest to the boundary: quadratic when three active cells are available, linear or constant
when fewer are.

- `bounded`: for tracers advected in flux form, project the reconstructed values in the cells whose stencils read
  ghost values onto a cell-wise admissible set, so that the advective update of these cells stays within the range of
  the cell and its neighbors. Next to a boundary the range extends to the value extrapolated to the boundary, within
  the range of the tracer over the domain, so that a stratification evolves next to the boundary without being
  clipped. The projection bounds the update of the state that each stage advances by the stage time step; it is exact
  for forward-Euler, strong-stability-preserving and split Runge-Kutta stages and approximate for other time steppers.
  Where no projection is attached, as for momentum, the ghost values blend the extrapolation with the mirror image of
  the active cells across the boundary, as `(1 - θ) * mirror + θ * extrapolation` with

  ```math
  θ = δ⁴ / (δ⁴ + 100 κ⁴), \\quad δ = c₁ - c₀, \\quad κ = c₂ - 2c₁ + c₀,
  ```

  so that smooth data are extrapolated while jumps are mirrored, and every reconstruction that reads ghost values is
  bounded between the upwind and downwind cells and within twice the upwind gradient of the upwind cell, as in
  [SureshHuynh97](@citet).
"""
struct GhostCells{S, P}
    scheme :: S
    projection :: P
    bounded :: Bool
end

GhostCells(FT::DataType = Oceananigans.defaults.FloatType; bounded = true) = GhostCells(nothing, nothing, bounded)

Base.summary(scheme::GhostCells) = string("GhostCells(bounded=", scheme.bounded, ")")

const GhostCellWENO = WENO{<:Any, <:Any, <:Any, <:Any, <:Any, <:GhostCells}

Adapt.adapt_structure(to, scheme::GhostCells) =
    GhostCells(Adapt.adapt(to, scheme.scheme), Adapt.adapt(to, scheme.projection), scheme.bounded)

Oceananigans.Architectures.on_architecture(to, scheme::GhostCells) =
    GhostCells(on_architecture(to, scheme.scheme), on_architecture(to, scheme.projection), scheme.bounded)

# Lagrange extrapolation from the cells at distances 0, 1, 2 from the boundary to the ghost at depth `d`;
# `f₁` and `f₂` flag whether the second and third extrapolation nodes are active cells.
@inline function ghost_value(c₀, c₁, c₂, f₁, f₂, d)
    δ = ifelse(f₁, c₁ - c₀, zero(c₀))
    κ = ifelse(f₁ & f₂, c₂ - 2c₁ + c₀, zero(c₀))
    return c₀ - d * δ + (d * (d + 1) ÷ 2) * κ
end

# Blend of the mirror image of the active cells across the boundary with their Lagrange extrapolation, which turns to
# the mirror where the curvature is large compared with the gradient, so that jumps are not extrapolated
@inline function blended_ghost_value(mirror, c₀, c₁, c₂, f₁, f₂, d)
    δ = ifelse(f₁, c₁ - c₀, zero(c₀))
    κ = ifelse(f₁ & f₂, c₂ - 2c₁ + c₀, zero(c₀))
    δ⁴ = (δ * δ) * (δ * δ)
    Σ = δ⁴ + 100 * (κ * κ) * (κ * κ)
    θ = δ⁴ / ifelse(Σ > 0, Σ, one(Σ))
    return mirror + θ * (c₀ - d * δ + (d * (d + 1) ÷ 2) * κ - mirror)
end

@inline lagrange_ghost_values(c, f, ::Val{N}) where N = @inbounds ntuple(d -> ghost_value(c[1], c[2], c[3], f[1], f[2], d), Val(N - 1))
@inline blended_ghost_values(c, f, ::Val{N}) where N = @inbounds ntuple(d -> blended_ghost_value(c[d], c[1], c[2], c[3], f[1], f[2], d), Val(N - 1))

# The projection bounds the tracers that carry one; without it a bounded scheme mirrors the active cells across jumps
@inline ghost_values(scheme::GhostCells{<:Any, Nothing}, c, f, N) = ifelse(scheme.bounded, blended_ghost_values(c, f, N), lagrange_ghost_values(c, f, N))
@inline ghost_values(scheme::GhostCells, c, f, N) = lagrange_ghost_values(c, f, N)

@inline upwind_stencil_length(N) = max(2N - 1, N + 2)

function clamped_to_lower_end(N, q)
    if q < 1
        return clamped_to_lower_end(N, 1)
    elseif q ≥ N
        return :(S[$q])
    else
        return :(ifelse(Aᵃ[$(N - q)], S[$q], $(clamped_to_lower_end(N, q + 1))))
    end
end

function clamped_to_upper_end(N, q)
    if q > upwind_stencil_length(N)
        return clamped_to_upper_end(N, upwind_stencil_length(N))
    elseif q ≤ N
        return :(S[$q])
    else
        return :(ifelse(Aᵇ[$(q - N)], S[$q], $(clamped_to_upper_end(N, q - 1))))
    end
end

# Whether the stencil cell `q` is active, given the extent of the active segment on both sides of the upwind cell `N`
function active_position(N, q)
    if q == N
        return true
    elseif 1 ≤ q < N
        return :(Aᵃ[$(N - q)])
    elseif N < q ≤ upwind_stencil_length(N)
        return :(Aᵇ[$(q - N)])
    else
        return false
    end
end

function lower_extrapolation_node(N, k)
    node = clamped_to_upper_end(N, N + k)
    for extent in 1:N-2
        node = :(ifelse(Aᵃ[$extent], $(clamped_to_upper_end(N, N - extent + k)), $node))
    end
    return node
end

function lower_extrapolation_node_activity(N, k)
    activity = active_position(N, N + k)
    for extent in 1:N-2
        activity = :(ifelse(Aᵃ[$extent], $(active_position(N, N - extent + k)), $activity))
    end
    return activity
end

function upper_extrapolation_node(N, k)
    node = clamped_to_lower_end(N, N - k)
    for extent in 1:upwind_stencil_length(N)-N-1
        node = :(ifelse(Aᵇ[$extent], $(clamped_to_lower_end(N, N + extent - k)), $node))
    end
    return node
end

function upper_extrapolation_node_activity(N, k)
    activity = active_position(N, N - k)
    for extent in 1:upwind_stencil_length(N)-N-1
        activity = :(ifelse(Aᵇ[$extent], $(active_position(N, N + extent - k)), $activity))
    end
    return activity
end

function completed_cell(N, q)
    q == N && return :(S[$N])
    A, g, depth = if q < N
        (:Aᵃ, :gᵃ, N - q)
    else
        (:Aᵇ, :gᵇ, q - N)
    end
    ghost = :($g[$depth])
    for extent in 1:depth-1
        ghost = :(ifelse($A[$extent], $g[$(depth - extent)], $ghost))
    end
    return :(ifelse($A[$depth], S[$q], $ghost))
end

for N in advection_buffers[2:end]
    @eval begin
        @inline  lower_extrapolation_stencil(S, Aᵃ, Aᵇ, ::Val{$N}) = @inbounds $(Expr(:tuple, (lower_extrapolation_node(N, k) for k in 0:max(2, N - 2))...))
        @inline  upper_extrapolation_stencil(S, Aᵃ, Aᵇ, ::Val{$N}) = @inbounds $(Expr(:tuple, (upper_extrapolation_node(N, k) for k in 0:max(2, N - 2))...))
        @inline lower_extrapolation_activity(Aᵃ, Aᵇ, ::Val{$N})    = @inbounds $(Expr(:tuple, (lower_extrapolation_node_activity(N, k) for k in 1:2)...))
        @inline upper_extrapolation_activity(Aᵃ, Aᵇ, ::Val{$N})    = @inbounds $(Expr(:tuple, (upper_extrapolation_node_activity(N, k) for k in 1:2)...))
        @inline completed_stencil(S, Aᵃ, Aᵇ, gᵃ, gᵇ, ::Val{$N}) = @inbounds $(Expr(:tuple, (completed_cell(N, q) for q in 1:2N-1)...))
    end
end

@inline function ghost_cell_reconstruction(scheme::GhostCells{<:WENO{N}}, bias, S, Aᵃ, Aᵇ, δˢ) where N
    cᵃ = lower_extrapolation_stencil(S, Aᵃ, Aᵇ, Val(N)) # Active cells nearest the lower boundary of the stencil
    cᵇ = upper_extrapolation_stencil(S, Aᵃ, Aᵇ, Val(N)) # Active cells nearest the upper boundary of the stencil
    fᵃ = lower_extrapolation_activity(Aᵃ, Aᵇ, Val(N))
    fᵇ = upper_extrapolation_activity(Aᵃ, Aᵇ, Val(N))

    gᵃ = ghost_values(scheme, cᵃ, fᵃ, Val(N))
    gᵇ = ghost_values(scheme, cᵇ, fᵇ, Val(N))
    S̃  = completed_stencil(S, Aᵃ, Aᵇ, gᵃ, gᵇ, Val(N))

    ghosted = @inbounds !(Aᵃ[N - 1] & Aᵇ[N - 1])

    δ = weno_differences(scheme.scheme, S̃)
    ω = biased_weno_weights(smoothness(ghosted, δ, δˢ), nothing, scheme.scheme)
    ψ̂ = weno_reconstruction(scheme.scheme, weno_anchor(scheme.scheme, S̃), δ, ω)
    ψ̃ = @inbounds bounded_ghost_cell_value(scheme, ψ̂, S[N], S̃[N + 1], S̃[N - 1])

    return ifelse(ghosted, ψ̃, ψ̂)
end

# Between the upwind and the downwind cells, and within twice the upwind gradient of the upwind cell
@inline function monotone_bound(ψ̂, U, D, UU)
    Uᴸ = U + 2 * (U - UU)
    ψᵐᵃˣ = ifelse(D > Uᴸ, D, Uᴸ)
    ψᵐⁱⁿ = ifelse(D > Uᴸ, Uᴸ, D)
    return clamp(ψ̂, ifelse(U < ψᵐᵃˣ, U, ψᵐᵃˣ), ifelse(U > ψᵐⁱⁿ, U, ψᵐⁱⁿ))
end

# The projection bounds the tracers that carry one, so the reconstruction is bounded face by face only without it
@inline bounded_ghost_cell_value(scheme::GhostCells{<:Any, Nothing}, ψ̂, U, D, UU) = ifelse(scheme.bounded, monotone_bound(ψ̂, U, D, UU), ψ̂)
@inline bounded_ghost_cell_value(scheme::GhostCells, ψ̂, U, D, UU) = ψ̂

# Stencils with ghost points do not obey the `VelocityStencil` and `FunctionStencil`,
# they always treat the weno reconstruction as if there was a `DefaultStencil`
@inline smoothness(ghosted, δ, ::Nothing) = δ
@inline smoothness(ghosted, δ, δˢ::NTuple) = ifelse(ghosted, δ, δˢ)
@inline smoothness(ghosted, δ, δˢ::Tuple{Tuple, Tuple}) = ifelse(ghosted, (δ, δ), δˢ)

@inline active_node(i, j, k, grid, ℓx, ℓy, ℓz) = !inactive_node(i, j, k, grid, ℓx, ℓy, ℓz)

@inline function ghost_cell_interpolate(i, j, k, grid, scheme::GhostCells{<:WENO{N}}, bias, e, ℓx, ℓy, ℓz, δˢ, ψ, args...) where N
    S = upwind_stencil(i, j, k, grid, scheme.scheme, bias, e, ψ, args...)
    A = upwind_stencil(i, j, k, grid, scheme.scheme, bias, e, active_node, ℓx, ℓy, ℓz)

    Aᵃ = accumulate(&, ntuple(@inline(extent -> @inbounds(A[N - extent])), Val(N - 1)))
    Aᵇ = accumulate(&, ntuple(@inline(extent -> @inbounds(A[N + extent])), Val(upwind_stencil_length(N) - N)))

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
        @eval @inline $symmetric(i, j, k, grid, ::GhostCells{<:WENO{<:Any, FT}}, args...) where FT = $symmetric(i, j, k, grid, Centered{1, FT}(nothing, ExplicitTimeDiscretization()), args...)
    end
end
