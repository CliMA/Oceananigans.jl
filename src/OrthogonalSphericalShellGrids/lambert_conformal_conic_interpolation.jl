using Oceananigans.Fields: FractionalIndices, fractional_z_index
import Oceananigans.Fields: _fractional_indices

# A LambertConformalConicGrid is exactly uniform in its projected (x, y) meters
# plane, so the fractional index is a closed-form O(1) lookup: forward-project the
# target (λ, φ) to projected meters and divide by the (constant) projected spacing.
# The horizontal index is *joint* in (λ, φ) — unlike the separable LatitudeLongitude
# axes — because each projected coordinate depends on both λ and φ, so the shortcut
# installs at the `_fractional_indices` layer where the full horizontal node is in scope.

const CenterOrFace = Union{Center, Face}

@inline function lcc_fractional_indices(λ, φ, grid::LambertConformalConicGrid, ℓx, ℓy)
    map  = grid.conformal_mapping
    x, y = lcc_forward(map, λ, φ)
    x₀   = lcc_xnode(1, ℓx, map)
    y₀   = lcc_ynode(1, ℓy, map)
    ii   = (x - x₀) / map.Δx + 1
    jj   = (y - y₀) / map.Δy + 1
    return ii, jj
end

@inline function _fractional_indices((λ, φ, z)::NTuple{3, Any},
                                     grid::LambertConformalConicGrid,
                                     ℓx::CenterOrFace, ℓy::CenterOrFace, ℓz::CenterOrFace)
    ii, jj = lcc_fractional_indices(λ, φ, grid, ℓx, ℓy)
    kk     = fractional_z_index(z, (ℓx, ℓy, ℓz), grid)
    return FractionalIndices(ii, jj, kk)
end

@inline function _fractional_indices((λ, φ)::NTuple{2, Any},
                                     grid::LambertConformalConicGrid,
                                     ℓx::CenterOrFace, ℓy::CenterOrFace, ::Nothing)
    ii, jj = lcc_fractional_indices(λ, φ, grid, ℓx, ℓy)
    return FractionalIndices(ii, jj, nothing)
end
