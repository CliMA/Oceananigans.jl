using Oceananigans.Grids: MultiEnvelopeGrid, ξnode, ηnode,
    static_column_depthᶜᶜᵃ, static_column_depthᶠᶜᵃ, static_column_depthᶜᶠᵃ, static_column_depthᶠᶠᵃ
import Oceananigans.Grids: znode, _node

#####
##### Multi-envelope metric: total scaling σ = static envelope Jacobian σᵉ × z-star scaling σ (σ at [i,j,1]).
##### More specific than the `grid::AMG` z-star methods, so they win by dispatch and the `Δz = Δr · σⁿ`
##### macros become depth-dependent automatically.
#####

@inline σⁿ(i, j, k, grid::MultiEnvelopeGrid, ::C, ::C, ℓz) = @inbounds grid.z.σᶜᶜᵉ[i, j, k] * grid.z.σᶜᶜⁿ[i, j, 1]
@inline σⁿ(i, j, k, grid::MultiEnvelopeGrid, ::F, ::C, ℓz) = @inbounds grid.z.σᶠᶜᵉ[i, j, k] * grid.z.σᶠᶜⁿ[i, j, 1]
@inline σⁿ(i, j, k, grid::MultiEnvelopeGrid, ::C, ::F, ℓz) = @inbounds grid.z.σᶜᶠᵉ[i, j, k] * grid.z.σᶜᶠⁿ[i, j, 1]
@inline σⁿ(i, j, k, grid::MultiEnvelopeGrid, ::F, ::F, ℓz) = @inbounds grid.z.σᶠᶠᵉ[i, j, k] * grid.z.σᶠᶠⁿ[i, j, 1]

@inline σ⁻(i, j, k, grid::MultiEnvelopeGrid, ::C, ::C, ℓz) = @inbounds grid.z.σᶜᶜᵉ[i, j, k] * grid.z.σᶜᶜ⁻[i, j, 1]

# σᵉ here is conservation-critical: the grid-motion velocity Δr_k·∂t_σ(i,j,k) in compute_w_from_continuity!
# must equal ∂t(Δz_k) = Δr_k σᵉ(k) ∂tσ; dropping σᵉ silently breaks the discrete GCL.
@inline ∂t_σ(i, j, k, grid::MultiEnvelopeGrid) = @inbounds grid.z.σᶜᶜᵉ[i, j, k] * grid.z.∂t_σ[i, j, 1]

#####
##### Node position from the cumulative Δz integral.
#####
##### The generic mutable-grid `znode = rnode·σ + η` is exact only when σ is k-independent (z-star, or a
##### single linear envelope). For a depth-dependent σᵉ the position is the integral z(k) = −H + Σ_{k′<k} Δz,
##### which already folds in σ_fs (via Δz) and η; it reduces *identically* to rnode·σ + η when σᵉ is constant.
#####

@inline function bottom_up_znode(i, j, k, grid, Δz, H)
    z = -H
    @inbounds for k′ in 1:k-1
        z += Δz(i, j, k′, grid)
    end
    return z
end

@inline znode(i, j, k, grid::MultiEnvelopeGrid, ::C, ::C, ::F) = bottom_up_znode(i, j, k, grid, Δzᶜᶜᶜ, static_column_depthᶜᶜᵃ(i, j, grid))
@inline znode(i, j, k, grid::MultiEnvelopeGrid, ::F, ::C, ::F) = bottom_up_znode(i, j, k, grid, Δzᶠᶜᶜ, static_column_depthᶠᶜᵃ(i, j, grid))
@inline znode(i, j, k, grid::MultiEnvelopeGrid, ::C, ::F, ::F) = bottom_up_znode(i, j, k, grid, Δzᶜᶠᶜ, static_column_depthᶜᶠᵃ(i, j, grid))
@inline znode(i, j, k, grid::MultiEnvelopeGrid, ::F, ::F, ::F) = bottom_up_znode(i, j, k, grid, Δzᶠᶠᶜ, static_column_depthᶠᶠᵃ(i, j, grid))

@inline znode(i, j, k, grid::MultiEnvelopeGrid, ::C, ::C, ::C) = znode(i, j, k, grid, C(), C(), F()) + Δzᶜᶜᶜ(i, j, k, grid) / 2
@inline znode(i, j, k, grid::MultiEnvelopeGrid, ::F, ::C, ::C) = znode(i, j, k, grid, F(), C(), F()) + Δzᶠᶜᶜ(i, j, k, grid) / 2
@inline znode(i, j, k, grid::MultiEnvelopeGrid, ::C, ::F, ::C) = znode(i, j, k, grid, C(), F(), F()) + Δzᶜᶠᶜ(i, j, k, grid) / 2
@inline znode(i, j, k, grid::MultiEnvelopeGrid, ::F, ::F, ::C) = znode(i, j, k, grid, F(), F(), F()) + Δzᶠᶠᶜ(i, j, k, grid) / 2

#####
##### `_node` override so `set!(field, f)` evaluates `f` at the physical depth (znode), not the reference r.
##### Stock `_node` uses `rnode` for the vertical entry; on an ME grid that is the computational coordinate,
##### not the physical depth, so depth-based initial conditions would be placed at the wrong level. Mirrors
##### Breeze's TerrainFollowingVerticalDiscretization handling. The four ℓz===nothing methods are identical
##### to stock and exist only to resolve dispatch ambiguity.
#####

@inline _node(i, j, k, grid::MultiEnvelopeGrid, ℓx, ℓy, ℓz) =
    (ξnode(i, j, k, grid, ℓx, ℓy, ℓz), ηnode(i, j, k, grid, ℓx, ℓy, ℓz), znode(i, j, k, grid, ℓx, ℓy, ℓz))
@inline _node(i, j, k, grid::MultiEnvelopeGrid, ℓx::Nothing, ℓy, ℓz) =
    (ηnode(i, j, k, grid, ℓx, ℓy, ℓz), znode(i, j, k, grid, ℓx, ℓy, ℓz))
@inline _node(i, j, k, grid::MultiEnvelopeGrid, ℓx, ℓy::Nothing, ℓz) =
    (ξnode(i, j, k, grid, ℓx, ℓy, ℓz), znode(i, j, k, grid, ℓx, ℓy, ℓz))
@inline _node(i, j, k, grid::MultiEnvelopeGrid, ℓx::Nothing, ℓy::Nothing, ℓz) =
    tuple(znode(i, j, k, grid, ℓx, ℓy, ℓz))
@inline _node(i, j, k, grid::MultiEnvelopeGrid, ℓx, ℓy, ℓz::Nothing) =
    (ξnode(i, j, k, grid, ℓx, ℓy, ℓz), ηnode(i, j, k, grid, ℓx, ℓy, ℓz))
@inline _node(i, j, k, grid::MultiEnvelopeGrid, ℓx, ℓy::Nothing, ℓz::Nothing) =
    tuple(ξnode(i, j, k, grid, ℓx, ℓy, ℓz))
@inline _node(i, j, k, grid::MultiEnvelopeGrid, ℓx::Nothing, ℓy, ℓz::Nothing) =
    tuple(ηnode(i, j, k, grid, ℓx, ℓy, ℓz))
@inline _node(i, j, k, grid::MultiEnvelopeGrid, ℓx::Nothing, ℓy::Nothing, ℓz::Nothing) =
    tuple()
