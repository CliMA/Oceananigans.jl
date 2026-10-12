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
  the cell and its neighbors. The projection uses the stage time step; it is exact for forward-Euler and
  strong-stability-preserving Runge-Kutta stages and approximate for other time steppers.
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
        @inline  lower_extrapolation_stencil(S, Aᵃ, Aᵇ, ::Val{$N}) = @inbounds $(Expr(:tuple, (lower_extrapolation_node(N, k) for k in 0:2)...))
        @inline  upper_extrapolation_stencil(S, Aᵃ, Aᵇ, ::Val{$N}) = @inbounds $(Expr(:tuple, (upper_extrapolation_node(N, k) for k in 0:2)...))
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

    gᵃ = @inbounds ntuple(d -> ghost_value(cᵃ[1], cᵃ[2], cᵃ[3], fᵃ[1], fᵃ[2], d), Val(N - 1))
    gᵇ = @inbounds ntuple(d -> ghost_value(cᵇ[1], cᵇ[2], cᵇ[3], fᵇ[1], fᵇ[2], d), Val(N - 1))
    S̃  = completed_stencil(S, Aᵃ, Aᵇ, gᵃ, gᵇ, Val(N))

    ghosted = @inbounds !(Aᵃ[N - 1] & Aᵇ[N - 1])

    δ = weno_differences(scheme.scheme, S̃)
    ω = biased_weno_weights(smoothness(ghosted, δ, δˢ), nothing, scheme.scheme)

    return weno_reconstruction(scheme.scheme, weno_anchor(scheme.scheme, S̃), δ, ω)
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
