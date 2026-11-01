using Oceananigans.Operators
using Oceananigans.Operators: MRG, MLLG, MOSG, superscript_location, north_fold_index

using Oceananigans.Grids: AbstractUnderlyingGrid, Bounded, Closed, FaceFolded, Periodic, TwoSided, MutableVerticalDiscretization

import Oceananigans.Grids: column_depthᶜᶜᵃ,
                           column_depthᶜᶠᵃ,
                           column_depthᶠᶜᵃ,
                           column_depthᶠᶠᵃ

import Oceananigans.Operators: σⁿ, σ⁻, σ⁻¹ⁿ, ∂t_σ

const UnderlyingMutableGrid{FT, TX, TY} = AbstractUnderlyingGrid{FT, TX, TY, <:Bounded, <:MutableVerticalDiscretization}
const MutableImmersedGrid{FT, TX, TY}   = ImmersedBoundaryGrid{FT, TX, TY, <:Bounded, <:UnderlyingMutableGrid}
const MutableGridOfSomeKind{FT, TX, TY} = Union{MutableImmersedGrid{FT, TX, TY}, UnderlyingMutableGrid{FT, TX, TY}}

@inline mutable_column_depth(h, η) = ifelse(h == 0, zero(h), h + η)

@inline column_depthᶜᶜᵃ(i, j, k, grid::MutableGridOfSomeKind, η) = mutable_column_depth(static_column_depthᶜᶜᵃ(i, j, grid), @inbounds η[i, j, k])
@inline column_depthᶠᶜᵃ(i, j, k, grid::MutableGridOfSomeKind, η) = mutable_column_depth(static_column_depthᶠᶜᵃ(i, j, grid),  ℑxᶠᵃᵃ(i, j, k, grid, η))
@inline column_depthᶜᶠᵃ(i, j, k, grid::MutableGridOfSomeKind, η) = mutable_column_depth(static_column_depthᶜᶠᵃ(i, j, grid),  ℑyᵃᶠᵃ(i, j, k, grid, η))
@inline column_depthᶠᶠᵃ(i, j, k, grid::MutableGridOfSomeKind, η) = mutable_column_depth(static_column_depthᶠᶠᵃ(i, j, grid), ℑxyᶠᶠᵃ(i, j, k, grid, η))

# The moving depth of a face with static depth `h` between columns (i⁻, j⁻) and (i, j)
@inline moving_face_depth(h, i⁻, j⁻, i, j, k, grid, η) = @inbounds mutable_column_depth(h, (η[i⁻, j⁻, k] + η[i, j, k]) / 2)

const MutableImmersedTopGrid = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Bounded, <:UnderlyingMutableGrid,
                                                    <:Union{GridFittedWithTop, PartialCellWithTop}}

@inline function column_stretching(i, j, k, grid, η)
    h = static_column_depthᶜᶜᵃ(i, j, grid)
    return @inbounds ifelse(h == 0, one(grid), (h + η[i, j, k]) / ifelse(h == 0, one(grid), h))
end

# Beneath a top, adjacent columns can differ in depth by far more than their face, which then stretches with its columns
@inline moving_face_depth(h, i⁻, j⁻, i, j, k, grid::MutableImmersedTopGrid, η) =
    h * (column_stretching(i⁻, j⁻, k, grid, η) + column_stretching(i, j, k, grid, η)) / 2

@inline column_depthᶠᶜᵃ(i, j, k, grid::MutableImmersedTopGrid, η) = static_column_depthᶠᶜᵃ(i, j, grid) * ℑxᶠᵃᵃ(i, j, k, grid, column_stretching, η)
@inline column_depthᶜᶠᵃ(i, j, k, grid::MutableImmersedTopGrid, η) = static_column_depthᶜᶠᵃ(i, j, grid) * ℑyᵃᶠᵃ(i, j, k, grid, column_stretching, η)
@inline column_depthᶠᶠᵃ(i, j, k, grid::MutableImmersedTopGrid, η) = static_column_depthᶠᶠᵃ(i, j, grid) * ℑxyᶠᶠᵃ(i, j, k, grid, column_stretching, η)

# Convenience methods
@inline column_depthᶜᶜᵃ(i, j, grid) = static_column_depthᶜᶜᵃ(i, j, grid)
@inline column_depthᶜᶠᵃ(i, j, grid) = static_column_depthᶜᶠᵃ(i, j, grid)
@inline column_depthᶠᶜᵃ(i, j, grid) = static_column_depthᶠᶜᵃ(i, j, grid)
@inline column_depthᶠᶠᵃ(i, j, grid) = static_column_depthᶠᶠᵃ(i, j, grid)

@inline column_depthᶜᶜᵃ(i, j, grid::MutableGridOfSomeKind) = column_depthᶜᶜᵃ(i, j, 1, grid, grid.z.ηⁿ)
@inline column_depthᶜᶠᵃ(i, j, grid::MutableGridOfSomeKind) = column_depthᶜᶠᵃ(i, j, 1, grid, grid.z.ηⁿ)
@inline column_depthᶠᶜᵃ(i, j, grid::MutableGridOfSomeKind) = column_depthᶠᶜᵃ(i, j, 1, grid, grid.z.ηⁿ)
@inline column_depthᶠᶠᵃ(i, j, grid::MutableGridOfSomeKind) = column_depthᶠᶠᵃ(i, j, 1, grid, grid.z.ηⁿ)

# Three dimensional column_depth methods for use in `KernelOperations`
@inline column_depthᶜᶜᵃ(i, j, k, grid) = static_column_depthᶜᶜᵃ(i, j, grid)
@inline column_depthᶜᶠᵃ(i, j, k, grid) = static_column_depthᶜᶠᵃ(i, j, grid)
@inline column_depthᶠᶜᵃ(i, j, k, grid) = static_column_depthᶠᶜᵃ(i, j, grid)
@inline column_depthᶠᶠᵃ(i, j, k, grid) = static_column_depthᶠᶠᵃ(i, j, grid)

@inline column_depthᶜᶜᵃ(i, j, k, grid::MutableGridOfSomeKind) = column_depthᶜᶜᵃ(i, j, 1, grid, grid.z.ηⁿ)
@inline column_depthᶜᶠᵃ(i, j, k, grid::MutableGridOfSomeKind) = column_depthᶜᶠᵃ(i, j, 1, grid, grid.z.ηⁿ)
@inline column_depthᶠᶜᵃ(i, j, k, grid::MutableGridOfSomeKind) = column_depthᶠᶜᵃ(i, j, 1, grid, grid.z.ηⁿ)
@inline column_depthᶠᶠᵃ(i, j, k, grid::MutableGridOfSomeKind) = column_depthᶠᶠᵃ(i, j, 1, grid, grid.z.ηⁿ)

# Topology - aware column height (used for the SplitExplicitFreeSurface)

@inline column_depthTᶠᶜᵃ(i, j, k, grid::AbstractGrid, η) = column_depthᶠᶜᵃ(i, j, k, grid, η)
@inline column_depthTᶜᶠᵃ(i, j, k, grid::AbstractGrid, η) = column_depthᶜᶠᵃ(i, j, k, grid, η)

const AMGXP = MutableGridOfSomeKind{<:Any, Periodic}
const AMGXB = MutableGridOfSomeKind{<:Any, <:TwoSided{Closed}}

const AMGYP = MutableGridOfSomeKind{<:Any, <:Any, Periodic}
const AMGYS = MutableGridOfSomeKind{<:Any, <:Any, TwoSided{L, R}} where {L, R}
const AMGYF = MutableGridOfSomeKind{<:Any, <:Any, TwoSided{L, FaceFolded}} where L

# Enforce Periodic conditions for column depth
@inline column_depthTᶠᶜᵃ(i, j, k, grid::AMGXP, η) = moving_face_depth(static_column_depthᶠᶜᵃ(i, j, grid), ifelse(i == 1, grid.Nx, i - 1), j, i, j, k, grid, η)
@inline column_depthTᶜᶠᵃ(i, j, k, grid::AMGYP, η) = moving_face_depth(static_column_depthᶜᶠᵃ(i, j, grid), i, ifelse(j == 1, grid.Ny, j - 1), i, j, k, grid, η)

# Enforce boundary conditions on a left wall
@inline column_depthTᶠᶜᵃ(i, j, k, grid::AMGXB, η) = moving_face_depth(static_column_depthᶠᶜᵃ(i, j, grid), ifelse(i == 1, i, i - 1), j, i, j, k, grid, η)
@inline column_depthTᶜᶠᵃ(i, j, k, grid::AMGYS{L}, η) where L = moving_face_depth(static_column_depthᶜᶠᵃ(i, j, grid), i, ifelse((L === Closed) & (j == 1), j, j - 1), i, j, k, grid, η)

@inline function column_depthTᶜᶠᵃ(i, j, k, grid::AMGYF{L}, η) where L
    fold = j == grid.Ny + 1
    i⁺ = ifelse(fold, north_fold_index(i, grid), i)
    j⁺ = ifelse(fold, grid.Ny, j)
    return moving_face_depth(static_column_depthᶜᶠᵃ(i, j, grid), i, ifelse((L === Closed) & (j == 1), j, j - 1), i⁺, j⁺, k, grid, η)
end

# Fallbacks
@inline σⁿ(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = σⁿ(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)
@inline σ⁻(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = σ⁻(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)
@inline σ⁻¹ⁿ(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = σ⁻¹ⁿ(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)

@inline ∂t_σ(i, j, k, ibg::IBG) = ∂t_σ(i, j, k, ibg.underlying_grid)

# Extend the 3D vertical spacing operators on an Immersed Mutable grid
const IMRG  = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:MRG}
const IMLLG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:MLLG}
const IMOSG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:MOSG}

for LX in (:ᶠ, :ᶜ), LY in (:ᶠ, :ᶜ), LZ in (:ᶠ, :ᶜ)
    zspacing = Symbol(:Δz, LX, LY, LZ)
    rspacing = Symbol(:Δr, LX, LY, LZ)

    rcp_zspacing = Symbol(:Δz⁻¹, LX, LY, LZ)
    rcp_rspacing = Symbol(:Δr⁻¹, LX, LY, LZ)

    ℓx = superscript_location(LX)
    ℓy = superscript_location(LY)
    ℓz = superscript_location(LZ)

    @eval begin
        using Oceananigans.Operators: $rspacing, $rcp_rspacing
        import Oceananigans.Operators: $zspacing, $rcp_zspacing

        @inline $zspacing(i, j, k, grid::IMRG)  = $rspacing(i, j, k, grid) * σⁿ(i, j, k, grid, $ℓx(), $ℓy(), $ℓz())
        @inline $zspacing(i, j, k, grid::IMLLG) = $rspacing(i, j, k, grid) * σⁿ(i, j, k, grid, $ℓx(), $ℓy(), $ℓz())
        @inline $zspacing(i, j, k, grid::IMOSG) = $rspacing(i, j, k, grid) * σⁿ(i, j, k, grid, $ℓx(), $ℓy(), $ℓz())

        @inline $rcp_zspacing(i, j, k, grid::IMRG)  = $rcp_rspacing(i, j, k, grid) * σ⁻¹ⁿ(i, j, k, grid, $ℓx(), $ℓy(), $ℓz())
        @inline $rcp_zspacing(i, j, k, grid::IMLLG) = $rcp_rspacing(i, j, k, grid) * σ⁻¹ⁿ(i, j, k, grid, $ℓx(), $ℓy(), $ℓz())
        @inline $rcp_zspacing(i, j, k, grid::IMOSG) = $rcp_rspacing(i, j, k, grid) * σ⁻¹ⁿ(i, j, k, grid, $ℓx(), $ℓy(), $ℓz())
    end
end

#####
##### Chain-rule-correct horizontal derivatives for MutableImmersedGrid
#####
##### Forward to underlying grid which has the actual chain-rule implementation.
#####

import Oceananigans.Operators: ∂xᶠᶜᶜ, ∂xᶜᶜᶜ, ∂xᶠᶜᶠ, ∂xᶜᶠᶜ, ∂xᶠᶠᶜ
import Oceananigans.Operators: ∂yᶜᶠᶜ, ∂yᶜᶜᶜ, ∂yᶜᶠᶠ, ∂yᶠᶜᶜ, ∂yᶠᶠᶜ
import Oceananigans.Operators: ∂x_zᶠᶜᶜ, ∂x_zᶜᶜᶜ, ∂x_zᶠᶜᶠ, ∂x_zᶜᶠᶜ, ∂x_zᶠᶠᶜ, ∂x_zᶜᶜᶠ, ∂x_zᶜᶠᶠ
import Oceananigans.Operators: ∂y_zᶜᶠᶜ, ∂y_zᶜᶜᶜ, ∂y_zᶜᶠᶠ, ∂y_zᶠᶜᶜ, ∂y_zᶠᶠᶜ, ∂y_zᶜᶜᶠ, ∂y_zᶠᶜᶠ

using Oceananigans.Operators: Δx⁻¹ᶜᶜᶜ, Δx⁻¹ᶜᶜᶠ, Δx⁻¹ᶜᶠᶜ, Δx⁻¹ᶜᶠᶠ, Δx⁻¹ᶠᶜᶜ, Δx⁻¹ᶠᶜᶠ, Δx⁻¹ᶠᶠᶜ
using Oceananigans.Operators: Δy⁻¹ᶜᶜᶜ, Δy⁻¹ᶜᶜᶠ, Δy⁻¹ᶜᶠᶜ, Δy⁻¹ᶜᶠᶠ, Δy⁻¹ᶠᶜᶜ, Δy⁻¹ᶠᶜᶠ, Δy⁻¹ᶠᶠᶜ
using Oceananigans.Operators: δxᶜᶜᶜ, δxᶜᶜᶠ, δxᶜᶠᶜ, δxᶜᶠᶠ, δxᶠᶜᶜ, δxᶠᶜᶠ, δxᶠᶠᶜ, δyᶜᶜᶜ, δyᶜᶜᶠ, δyᶜᶠᶜ, δyᶜᶠᶠ, δyᶠᶜᶜ, δyᶠᶜᶠ, δyᶠᶠᶜ
using Oceananigans.Operators: ℑxzᶜᵃᶜ, ℑxzᶠᵃᶜ, ℑxzᶠᵃᶠ, ℑyzᵃᶜᶜ, ℑyzᵃᶠᶜ, ℑyzᵃᶠᶠ, active_weighted_ℑxzᶠᶜᶜ, active_weighted_ℑyzᶜᶠᶜ
using Oceananigans.Operators: ∂zᶜᶜᶜ, ∂zᶜᶜᶠ, ∂zᶜᶠᶠ, ∂zᶠᶜᶠ, ∂zᶠᶠᶠ

#####
##### Generalized coordinate derivatives for mutable vertical grids
#####
##### For z-star coordinates where z(ξ, η, r, t) = η_fs + σ·r, derivatives transform as:
#####
##### Horizontal derivatives (chain rule):
#####   ∂ϕ/∂x|_z = ∂ϕ/∂x|_r - (∂z/∂x|_r)(∂ϕ/∂z)
#####   ∂ϕ/∂y|_z = ∂ϕ/∂y|_r - (∂z/∂y|_r)(∂ϕ/∂z)
#####
##### Vertical derivatives (stretching):
#####   ∂ϕ/∂z = (1/σ)(∂ϕ/∂r)
#####
##### Note: Vertical derivatives are already correct because Δz = σ·Δr is used
##### in the spacing operators for mutable grids (see time_variable_grid_operators.jl).
#####
##### The grid slopes ∂z/∂x|_r and ∂z/∂y|_r are computed using difference operators
##### (not derivatives) to avoid recursion.
#####

using Oceananigans.Grids: znode, Center, Face

const AMG = MutableGridOfSomeKind
const C = Center
const F = Face

const MutableImmersedTopGrid = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Bounded, <:UnderlyingMutableGrid,
                                                    <:Union{GridFittedWithTop, PartialCellWithTop}}

# Stretch about the immersed top `zᵗ` so the bottom stays put and the top moves with `η`
@inline top_stretched_znode(r, σ, η, zᵗ) = zᵗ + η + σ * (r - zᵗ)

@inline top_heightᶜᶜᵃ(i, j, k, grid) = top_height(i, j, grid)

@inline Grids.znode(i, j, k, grid::MutableImmersedTopGrid, ::C, ::C, ℓz) =
    top_stretched_znode(rnode(i, j, k, grid, C(), C(), ℓz), σⁿ(i, j, k, grid, C(), C(), ℓz),
                        @inbounds(grid.z.ηⁿ[i, j, 1]), top_height(i, j, grid))

@inline Grids.znode(i, j, k, grid::MutableImmersedTopGrid, ::F, ::C, ℓz) =
    top_stretched_znode(rnode(i, j, k, grid, F(), C(), ℓz), σⁿ(i, j, k, grid, F(), C(), ℓz),
                        ℑxᶠᵃᵃ(i, j, 1, grid, grid.z.ηⁿ), ℑxᶠᵃᵃ(i, j, 1, grid, top_heightᶜᶜᵃ))

@inline Grids.znode(i, j, k, grid::MutableImmersedTopGrid, ::C, ::F, ℓz) =
    top_stretched_znode(rnode(i, j, k, grid, C(), F(), ℓz), σⁿ(i, j, k, grid, C(), F(), ℓz),
                        ℑyᵃᶠᵃ(i, j, 1, grid, grid.z.ηⁿ), ℑyᵃᶠᵃ(i, j, 1, grid, top_heightᶜᶜᵃ))

@inline Grids.znode(i, j, k, grid::MutableImmersedTopGrid, ::F, ::F, ℓz) =
    top_stretched_znode(rnode(i, j, k, grid, F(), F(), ℓz), σⁿ(i, j, k, grid, F(), F(), ℓz),
                        ℑxyᶠᶠᵃ(i, j, 1, grid, grid.z.ηⁿ), ℑxyᶠᶠᵃ(i, j, 1, grid, top_heightᶜᶜᵃ))

#####
##### Grid slope functions: ∂z/∂x|_r and ∂z/∂y|_r at various staggerings
#####
##### We use difference operators (δx, δy) instead of derivative operators (∂x, ∂y)
##### to avoid infinite recursion, since we're overriding ∂x/∂y.
#####

# x-direction slopes at different staggerings
@inline ∂x_zᶠᶜᶜ(i, j, k, grid::AMG) = δxᶠᶜᶜ(i, j, k, grid, znode, C(), C(), C()) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
@inline ∂x_zᶜᶜᶜ(i, j, k, grid::AMG) = δxᶜᶜᶜ(i, j, k, grid, znode, F(), C(), C()) * Δx⁻¹ᶜᶜᶜ(i, j, k, grid)
@inline ∂x_zᶠᶜᶠ(i, j, k, grid::AMG) = δxᶠᶜᶠ(i, j, k, grid, znode, C(), C(), F()) * Δx⁻¹ᶠᶜᶠ(i, j, k, grid)
@inline ∂x_zᶜᶠᶜ(i, j, k, grid::AMG) = δxᶜᶠᶜ(i, j, k, grid, znode, F(), F(), C()) * Δx⁻¹ᶜᶠᶜ(i, j, k, grid)
@inline ∂x_zᶠᶠᶜ(i, j, k, grid::AMG) = δxᶠᶠᶜ(i, j, k, grid, znode, C(), F(), C()) * Δx⁻¹ᶠᶠᶜ(i, j, k, grid)
@inline ∂x_zᶜᶜᶠ(i, j, k, grid::AMG) = δxᶜᶜᶠ(i, j, k, grid, znode, F(), C(), F()) * Δx⁻¹ᶜᶜᶠ(i, j, k, grid)
@inline ∂x_zᶜᶠᶠ(i, j, k, grid::AMG) = δxᶜᶠᶠ(i, j, k, grid, znode, F(), F(), F()) * Δx⁻¹ᶜᶠᶠ(i, j, k, grid)

# y-direction slopes at different staggerings
@inline ∂y_zᶜᶠᶜ(i, j, k, grid::AMG) = δyᶜᶠᶜ(i, j, k, grid, znode, C(), C(), C()) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
@inline ∂y_zᶜᶜᶜ(i, j, k, grid::AMG) = δyᶜᶜᶜ(i, j, k, grid, znode, C(), F(), C()) * Δy⁻¹ᶜᶜᶜ(i, j, k, grid)
@inline ∂y_zᶜᶠᶠ(i, j, k, grid::AMG) = δyᶜᶠᶠ(i, j, k, grid, znode, C(), C(), F()) * Δy⁻¹ᶜᶠᶠ(i, j, k, grid)
@inline ∂y_zᶠᶜᶜ(i, j, k, grid::AMG) = δyᶠᶜᶜ(i, j, k, grid, znode, F(), F(), C()) * Δy⁻¹ᶠᶜᶜ(i, j, k, grid)
@inline ∂y_zᶠᶠᶜ(i, j, k, grid::AMG) = δyᶠᶠᶜ(i, j, k, grid, znode, F(), C(), C()) * Δy⁻¹ᶠᶠᶜ(i, j, k, grid)
@inline ∂y_zᶜᶜᶠ(i, j, k, grid::AMG) = δyᶜᶜᶠ(i, j, k, grid, znode, C(), F(), F()) * Δy⁻¹ᶜᶜᶠ(i, j, k, grid)
@inline ∂y_zᶠᶜᶠ(i, j, k, grid::AMG) = δyᶠᶜᶠ(i, j, k, grid, znode, F(), F(), F()) * Δy⁻¹ᶠᶜᶠ(i, j, k, grid)

#####
##### Disambiguation for Number arguments (derivative of a constant is zero)
#####

@inline ∂xᶠᶜᶜ(i, j, k, grid::AMG, c::Number) = zero(grid)
@inline ∂xᶜᶜᶜ(i, j, k, grid::AMG, c::Number) = zero(grid)
@inline ∂xᶠᶜᶠ(i, j, k, grid::AMG, c::Number) = zero(grid)
@inline ∂xᶜᶠᶜ(i, j, k, grid::AMG, c::Number) = zero(grid)
@inline ∂xᶠᶠᶜ(i, j, k, grid::AMG, c::Number) = zero(grid)

@inline ∂yᶜᶠᶜ(i, j, k, grid::AMG, c::Number) = zero(grid)
@inline ∂yᶜᶜᶜ(i, j, k, grid::AMG, c::Number) = zero(grid)
@inline ∂yᶜᶠᶠ(i, j, k, grid::AMG, c::Number) = zero(grid)
@inline ∂yᶠᶜᶜ(i, j, k, grid::AMG, c::Number) = zero(grid)
@inline ∂yᶠᶠᶜ(i, j, k, grid::AMG, c::Number) = zero(grid)

#####
##### Chain-rule-correct x-derivatives: ∂ϕ/∂x|_z = ∂ϕ/∂x|_r - (∂z/∂x|_r)(∂ϕ/∂z)
#####

@inline vertical_derivativeᶠᶜᶜ(i, j, k, grid, args...) = ℑxzᶠᵃᶜ(i, j, k, grid, ∂zᶜᶜᶠ, args...)
@inline vertical_derivativeᶜᶠᶜ(i, j, k, grid, args...) = ℑyzᵃᶠᶜ(i, j, k, grid, ∂zᶜᶜᶠ, args...)

# Pressure lives at the underlying cell centres, so partial top cells must not shorten the spacing
@inline active_∂zᶜᶜᶠ(i, j, k, grid, args...) = ifelse(peripheral_node(i, j, k, grid, Center(), Center(), Face()), zero(grid), ∂zᶜᶜᶠ(i, j, k, grid.underlying_grid, args...))

@inline vertical_derivativeᶠᶜᶜ(i, j, k, grid::MutableImmersedTopGrid, args...) = active_weighted_ℑxzᶠᶜᶜ(i, j, k, grid, active_∂zᶜᶜᶠ, args...)
@inline vertical_derivativeᶜᶠᶜ(i, j, k, grid::MutableImmersedTopGrid, args...) = active_weighted_ℑyzᶜᶠᶜ(i, j, k, grid, active_∂zᶜᶜᶠ, args...)

# ∂xᶠᶜᶜ: tracer/buoyancy/pressure x-derivatives (most common)
@inline function ∂xᶠᶜᶜ(i, j, k, grid::AMG, ϕ)
    ∂x_at_r = δxᶠᶜᶜ(i, j, k, grid, ϕ) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    ∂z_ϕ = vertical_derivativeᶠᶜᶜ(i, j, k, grid, ϕ)
    ∂x_z = ∂x_zᶠᶜᶜ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

@inline function ∂xᶠᶜᶜ(i, j, k, grid::AMG, f::Function, args...)
    ∂x_at_r = δxᶠᶜᶜ(i, j, k, grid, f, args...) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    ∂z_ϕ = vertical_derivativeᶠᶜᶜ(i, j, k, grid, f, args...)
    ∂x_z = ∂x_zᶠᶜᶜ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

# ∂xᶜᶜᶜ: filtered velocity derivatives (Smagorinsky)
@inline function ∂xᶜᶜᶜ(i, j, k, grid::AMG, ϕ)
    ∂x_at_r = δxᶜᶜᶜ(i, j, k, grid, ϕ) * Δx⁻¹ᶜᶜᶜ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶜᵃᶜ(i, j, k, grid, ∂zᶠᶜᶠ, ϕ)
    ∂x_z = ∂x_zᶜᶜᶜ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

@inline function ∂xᶜᶜᶜ(i, j, k, grid::AMG, f::Function, args...)
    ∂x_at_r = δxᶜᶜᶜ(i, j, k, grid, f, args...) * Δx⁻¹ᶜᶜᶜ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶜᵃᶜ(i, j, k, grid, ∂zᶠᶜᶠ, f, args...)
    ∂x_z = ∂x_zᶜᶜᶜ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

# ∂xᶠᶜᶠ: w x-derivative
@inline function ∂xᶠᶜᶠ(i, j, k, grid::AMG, ϕ)
    ∂x_at_r = δxᶠᶜᶠ(i, j, k, grid, ϕ) * Δx⁻¹ᶠᶜᶠ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶠᵃᶠ(i, j, k, grid, ∂zᶜᶜᶜ, ϕ)
    ∂x_z = ∂x_zᶠᶜᶠ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

@inline function ∂xᶠᶜᶠ(i, j, k, grid::AMG, f::Function, args...)
    ∂x_at_r = δxᶠᶜᶠ(i, j, k, grid, f, args...) * Δx⁻¹ᶠᶜᶠ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶠᵃᶠ(i, j, k, grid, ∂zᶜᶜᶜ, f, args...)
    ∂x_z = ∂x_zᶠᶜᶠ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

# ∂xᶜᶠᶜ: vorticity x-derivative (Leith)
@inline function ∂xᶜᶠᶜ(i, j, k, grid::AMG, ϕ)
    ∂x_at_r = δxᶜᶠᶜ(i, j, k, grid, ϕ) * Δx⁻¹ᶜᶠᶜ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶜᵃᶜ(i, j, k, grid, ∂zᶠᶠᶠ, ϕ)
    ∂x_z = ∂x_zᶜᶠᶜ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

@inline function ∂xᶜᶠᶜ(i, j, k, grid::AMG, f::Function, args...)
    ∂x_at_r = δxᶜᶠᶜ(i, j, k, grid, f, args...) * Δx⁻¹ᶜᶠᶜ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶜᵃᶜ(i, j, k, grid, ∂zᶠᶠᶠ, f, args...)
    ∂x_z = ∂x_zᶜᶠᶜ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

# ∂xᶠᶠᶜ: filtered v x-derivative
@inline function ∂xᶠᶠᶜ(i, j, k, grid::AMG, ϕ)
    ∂x_at_r = δxᶠᶠᶜ(i, j, k, grid, ϕ) * Δx⁻¹ᶠᶠᶜ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶠᵃᶜ(i, j, k, grid, ∂zᶜᶠᶠ, ϕ)
    ∂x_z = ∂x_zᶠᶠᶜ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

@inline function ∂xᶠᶠᶜ(i, j, k, grid::AMG, f::Function, args...)
    ∂x_at_r = δxᶠᶠᶜ(i, j, k, grid, f, args...) * Δx⁻¹ᶠᶠᶜ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶠᵃᶜ(i, j, k, grid, ∂zᶜᶠᶠ, f, args...)
    ∂x_z = ∂x_zᶠᶠᶜ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

#####
##### Chain-rule-correct y-derivatives: ∂ϕ/∂y|_z = ∂ϕ/∂y|_r - (∂z/∂y|_r)(∂ϕ/∂z)
#####

# ∂yᶜᶠᶜ: tracer/buoyancy/pressure y-derivatives (most common)
@inline function ∂yᶜᶠᶜ(i, j, k, grid::AMG, ϕ)
    ∂y_at_r = δyᶜᶠᶜ(i, j, k, grid, ϕ) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    ∂z_ϕ = vertical_derivativeᶜᶠᶜ(i, j, k, grid, ϕ)
    ∂y_z = ∂y_zᶜᶠᶜ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

@inline function ∂yᶜᶠᶜ(i, j, k, grid::AMG, f::Function, args...)
    ∂y_at_r = δyᶜᶠᶜ(i, j, k, grid, f, args...) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    ∂z_ϕ = vertical_derivativeᶜᶠᶜ(i, j, k, grid, f, args...)
    ∂y_z = ∂y_zᶜᶠᶜ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

# ∂yᶜᶜᶜ: filtered velocity derivatives
@inline function ∂yᶜᶜᶜ(i, j, k, grid::AMG, ϕ)
    ∂y_at_r = δyᶜᶜᶜ(i, j, k, grid, ϕ) * Δy⁻¹ᶜᶜᶜ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶜᶜ(i, j, k, grid, ∂zᶜᶠᶠ, ϕ)
    ∂y_z = ∂y_zᶜᶜᶜ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

@inline function ∂yᶜᶜᶜ(i, j, k, grid::AMG, f::Function, args...)
    ∂y_at_r = δyᶜᶜᶜ(i, j, k, grid, f, args...) * Δy⁻¹ᶜᶜᶜ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶜᶜ(i, j, k, grid, ∂zᶜᶠᶠ, f, args...)
    ∂y_z = ∂y_zᶜᶜᶜ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

# ∂yᶜᶠᶠ: w y-derivative
@inline function ∂yᶜᶠᶠ(i, j, k, grid::AMG, ϕ)
    ∂y_at_r = δyᶜᶠᶠ(i, j, k, grid, ϕ) * Δy⁻¹ᶜᶠᶠ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶠᶠ(i, j, k, grid, ∂zᶜᶜᶜ, ϕ)
    ∂y_z = ∂y_zᶜᶠᶠ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

@inline function ∂yᶜᶠᶠ(i, j, k, grid::AMG, f::Function, args...)
    ∂y_at_r = δyᶜᶠᶠ(i, j, k, grid, f, args...) * Δy⁻¹ᶜᶠᶠ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶠᶠ(i, j, k, grid, ∂zᶜᶜᶜ, f, args...)
    ∂y_z = ∂y_zᶜᶠᶠ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

# ∂yᶠᶜᶜ: vorticity y-derivative
@inline function ∂yᶠᶜᶜ(i, j, k, grid::AMG, ϕ)
    ∂y_at_r = δyᶠᶜᶜ(i, j, k, grid, ϕ) * Δy⁻¹ᶠᶜᶜ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶜᶜ(i, j, k, grid, ∂zᶠᶠᶠ, ϕ)
    ∂y_z = ∂y_zᶠᶜᶜ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

@inline function ∂yᶠᶜᶜ(i, j, k, grid::AMG, f::Function, args...)
    ∂y_at_r = δyᶠᶜᶜ(i, j, k, grid, f, args...) * Δy⁻¹ᶠᶜᶜ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶜᶜ(i, j, k, grid, ∂zᶠᶠᶠ, f, args...)
    ∂y_z = ∂y_zᶠᶜᶜ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

# ∂yᶠᶠᶜ: filtered u y-derivative
@inline function ∂yᶠᶠᶜ(i, j, k, grid::AMG, ϕ)
    ∂y_at_r = δyᶠᶠᶜ(i, j, k, grid, ϕ) * Δy⁻¹ᶠᶠᶜ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶠᶜ(i, j, k, grid, ∂zᶠᶜᶠ, ϕ)
    ∂y_z = ∂y_zᶠᶠᶜ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

@inline function ∂yᶠᶠᶜ(i, j, k, grid::AMG, f::Function, args...)
    ∂y_at_r = δyᶠᶠᶜ(i, j, k, grid, f, args...) * Δy⁻¹ᶠᶠᶜ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶠᶜ(i, j, k, grid, ∂zᶠᶜᶠ, f, args...)
    ∂y_z = ∂y_zᶠᶠᶜ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

# Note: For z-reduced fields (fields with Nothing as z-location), the chain-rule
# correction term (∂z/∂x|_r)(∂ϕ/∂z) is automatically zero since ∂ϕ/∂z = 0 for such fields.
# Therefore, the general implementations above correctly return ∂ϕ/∂x|_z = ∂ϕ/∂x|_r.
