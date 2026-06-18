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

# At a z-FACE the envelope Jacobian must be the centre-to-centre value ½(σᵉ[k-1]+σᵉ[k]), so that Δzᶜᶜᶠ equals
# the distance between the adjacent cell centres and stays consistent with the cumulative znode (Δzᶜᶜᶜ). Using
# the centre-k σᵉ (the generic ℓz method) mis-sizes Δzᶜᶜᶠ at a σᵉ jump (zone interface), making ∂xᵣ(pHY)
# inconsistent with ∂x_z there → a resting spurious pressure gradient. For uniform σᵉ (sigma) this is a no-op.
@inline σⁿ(i, j, k, grid::MultiEnvelopeGrid, ::C, ::C, ::F) = @inbounds (grid.z.σᶜᶜᵉ[i, j, k-1] + grid.z.σᶜᶜᵉ[i, j, k]) / 2 * grid.z.σᶜᶜⁿ[i, j, 1]
@inline σⁿ(i, j, k, grid::MultiEnvelopeGrid, ::F, ::C, ::F) = @inbounds (grid.z.σᶠᶜᵉ[i, j, k-1] + grid.z.σᶠᶜᵉ[i, j, k]) / 2 * grid.z.σᶠᶜⁿ[i, j, 1]
@inline σⁿ(i, j, k, grid::MultiEnvelopeGrid, ::C, ::F, ::F) = @inbounds (grid.z.σᶜᶠᵉ[i, j, k-1] + grid.z.σᶜᶠᵉ[i, j, k]) / 2 * grid.z.σᶜᶠⁿ[i, j, 1]
@inline σⁿ(i, j, k, grid::MultiEnvelopeGrid, ::F, ::F, ::F) = @inbounds (grid.z.σᶠᶠᵉ[i, j, k-1] + grid.z.σᶠᶠᵉ[i, j, k]) / 2 * grid.z.σᶠᶠⁿ[i, j, 1]

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

# The centre column is O(1): z-star maps the static resting depth as z = η + σ_fs·ẑ_rest, and the precomputed
# `zᶜᶜᶜᵉ` IS ẑ_rest at the centre. This is the znode the pressure-gradient `∂x_z`/`∂y_z` differences, so it is
# the hot path; the off-centre staggers (used in diagnostics, not the PGF) keep the O(Nz) cumulative.
@inline znode(i, j, k, grid::MultiEnvelopeGrid, ::C, ::C, ::C) =
    @inbounds grid.z.ηⁿ[i, j, 1] + grid.z.σᶜᶜⁿ[i, j, 1] * grid.z.zᶜᶜᶜᵉ[i, j, k]
@inline znode(i, j, k, grid::MultiEnvelopeGrid, ::C, ::C, ::F) = znode(i, j, k, grid, C(), C(), C()) - Δzᶜᶜᶜ(i, j, k, grid) / 2

@inline znode(i, j, k, grid::MultiEnvelopeGrid, ::F, ::C, ::F) = bottom_up_znode(i, j, k, grid, Δzᶠᶜᶜ, static_column_depthᶠᶜᵃ(i, j, grid))
@inline znode(i, j, k, grid::MultiEnvelopeGrid, ::C, ::F, ::F) = bottom_up_znode(i, j, k, grid, Δzᶜᶠᶜ, static_column_depthᶜᶠᵃ(i, j, grid))
@inline znode(i, j, k, grid::MultiEnvelopeGrid, ::F, ::F, ::F) = bottom_up_znode(i, j, k, grid, Δzᶠᶠᶜ, static_column_depthᶠᶠᵃ(i, j, grid))

@inline znode(i, j, k, grid::MultiEnvelopeGrid, ::F, ::C, ::C) = znode(i, j, k, grid, F(), C(), F()) + Δzᶠᶜᶜ(i, j, k, grid) / 2
@inline znode(i, j, k, grid::MultiEnvelopeGrid, ::F, ::C, ::C) = znode(i, j, k, grid, F(), C(), F()) + Δzᶠᶜᶜ(i, j, k, grid) / 2
@inline znode(i, j, k, grid::MultiEnvelopeGrid, ::C, ::F, ::C) = znode(i, j, k, grid, C(), F(), F()) + Δzᶜᶠᶜ(i, j, k, grid) / 2
@inline znode(i, j, k, grid::MultiEnvelopeGrid, ::F, ::F, ::C) = znode(i, j, k, grid, F(), F(), F()) + Δzᶠᶠᶜ(i, j, k, grid) / 2

# Flat dimensions carry a reduced (Nothing) location with no stagger; reuse the Center column there. Without
# these, `set!`/`_node` on a Flat-y (or Flat-x) grid fall back to the GENERIC reference-coordinate znode and
# place tracers at the reference depth r instead of the physical ẑ on terrain-following columns — seeding a
# spurious resting buoyancy gradient (and hence a spurious baroclinic flow).
@inline znode(i, j, k, grid::MultiEnvelopeGrid, ℓx, ℓy::Nothing, ℓz) = znode(i, j, k, grid, ℓx, C(), ℓz)
@inline znode(i, j, k, grid::MultiEnvelopeGrid, ℓx::Nothing, ℓy, ℓz) = znode(i, j, k, grid, C(), ℓy, ℓz)
@inline znode(i, j, k, grid::MultiEnvelopeGrid, ℓx::Nothing, ℓy::Nothing, ℓz) = znode(i, j, k, grid, C(), C(), ℓz)

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
