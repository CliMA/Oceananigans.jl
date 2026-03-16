using Oceananigans.Operators
using Oceananigans.Operators: MRG, MLLG, MOSG, superscript_location

using Oceananigans.Grids: AbstractUnderlyingGrid,
                          Bounded,
                          LeftConnected,
                          Periodic,
                          RightConnected,
                          RightCenterFolded,
                          RightFaceFolded,
                          MutableVerticalDiscretization

import Oceananigans.Grids: column_depthᶜᶜᵃ,
                           column_depthᶜᶠᵃ,
                           column_depthᶠᶜᵃ,
                           column_depthᶠᶠᵃ

import Oceananigans.Operators: σⁿ, σ⁻, ∂t_σ

const UnderlyingMutableGrid{FT, TX, TY} = AbstractUnderlyingGrid{FT, TX, TY, <:Bounded, <:MutableVerticalDiscretization}
const MutableImmersedGrid{FT, TX, TY}   = ImmersedBoundaryGrid{FT, TX, TY, <:Bounded, <:UnderlyingMutableGrid}
const MutableGridOfSomeKind{FT, TX, TY} = Union{MutableImmersedGrid{FT, TX, TY}, UnderlyingMutableGrid{FT, TX, TY}}

@inline column_depthᶜᶜᵃ(i, j, k, grid::MutableGridOfSomeKind, η) = static_column_depthᶜᶜᵃ(i, j, grid) +  @inbounds η[i, j, k]
@inline column_depthᶠᶜᵃ(i, j, k, grid::MutableGridOfSomeKind, η) = static_column_depthᶠᶜᵃ(i, j, grid) +  ℑxᶠᵃᵃ(i, j, k, grid, η)
@inline column_depthᶜᶠᵃ(i, j, k, grid::MutableGridOfSomeKind, η) = static_column_depthᶜᶠᵃ(i, j, grid) +  ℑyᵃᶠᵃ(i, j, k, grid, η)
@inline column_depthᶠᶠᵃ(i, j, k, grid::MutableGridOfSomeKind, η) = static_column_depthᶠᶠᵃ(i, j, grid) + ℑxyᶠᶠᵃ(i, j, k, grid, η)

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

const AMGXB = MutableGridOfSomeKind{<:Any, Bounded}
const AMGXP = MutableGridOfSomeKind{<:Any, Periodic}
const AMGXR = MutableGridOfSomeKind{<:Any, <:Union{RightConnected, RightCenterFolded, RightFaceFolded}}
const AMGXL = MutableGridOfSomeKind{<:Any, LeftConnected}

const AMGYB = MutableGridOfSomeKind{<:Any, <:Any, Bounded}
const AMGYP = MutableGridOfSomeKind{<:Any, <:Any, Periodic}
const AMGYR = MutableGridOfSomeKind{<:Any, <:Any, <:Union{RightConnected, RightCenterFolded, RightFaceFolded}}
const AMGYL = MutableGridOfSomeKind{<:Any, <:Any, LeftConnected}

# Enforce Periodic conditions for column depth
@inline function column_depthTᶠᶜᵃ(i, j, k, grid::AMGXP, η)
    Hᶠᶜᵃ = column_depthᶠᶜᵃ(i, j, k, grid, η)
    hᶠᶜᵃ = static_column_depthᶠᶜᵃ(i, j, grid)
    ηᶠᶜᵃ = @inbounds (η[grid.Nx, j, k] + η[1, j, k]) / 2
    return ifelse(i == 1, hᶠᶜᵃ + ηᶠᶜᵃ, Hᶠᶜᵃ)
end

@inline function column_depthTᶜᶠᵃ(i, j, k, grid::AMGYP, η)
    Hᶜᶠᵃ = column_depthᶜᶠᵃ(i, j, k, grid, η)
    hᶜᶠᵃ = static_column_depthᶜᶠᵃ(i, j, grid)
    ηᶜᶠᵃ = @inbounds (η[i, grid.Ny, k] + η[i, 1, k]) / 2
    return ifelse(j == 1, hᶜᶠᵃ + ηᶜᶠᵃ, Hᶜᶠᵃ)
end

# Enforce boundary conditions for Bounded topologies
@inline function column_depthTᶠᶜᵃ(i, j, k, grid::AMGXB, η)
    Hᶠᶜᵃ = column_depthᶠᶜᵃ(i, j, k, grid, η)
    hᶠᶜᵃ = static_column_depthᶠᶜᵃ(i, j, grid)
    η₁ = @inbounds η[i, j, k]
    return ifelse(i == 1, hᶠᶜᵃ + η₁, Hᶠᶜᵃ)
end

@inline function column_depthTᶜᶠᵃ(i, j, k, grid::AMGYB, η)
    Hᶜᶠᵃ = column_depthᶜᶠᵃ(i, j, k, grid, η)
    hᶜᶠᵃ = static_column_depthᶜᶠᵃ(i, j, grid)
    η₁ = @inbounds η[i, j, k]
    return ifelse(j == 1, hᶜᶠᵃ + η₁, Hᶜᶠᵃ)
end

# Enforce boundary conditions for RightConnected/RightFolded topologies
@inline function column_depthTᶠᶜᵃ(i, j, k, grid::AMGXR, η)
    Hᶠᶜᵃ = column_depthᶠᶜᵃ(i, j, k, grid, η)
    hᶠᶜᵃ = static_column_depthᶠᶜᵃ(i, j, grid)
    η₁ = @inbounds η[1, j, k]
    return ifelse(i == 1, hᶠᶜᵃ + η₁,  Hᶠᶜᵃ)
end

@inline function column_depthTᶜᶠᵃ(i, j, k, grid::AMGYR, η)
    Hᶜᶠᵃ = column_depthᶜᶠᵃ(i, j, k, grid, η)
    hᶜᶠᵃ = static_column_depthᶜᶠᵃ(i, j, grid)
    η₁ = @inbounds η[i, j, k]
    return ifelse(j == 1, hᶜᶠᵃ + η₁, Hᶜᶠᵃ)
end

# Enforce boundary conditions for LeftConnected topologies
@inline function column_depthTᶠᶜᵃ(i, j, k, grid::AMGXL, η)
    Hᶠᶜᵃ = column_depthᶠᶜᵃ(i, j, k, grid, η)
    hᶠᶜᵃ = static_column_depthᶠᶜᵃ(i, j, grid)
    ηₑ = @inbounds η[grid.Nx, j, k]
    return ifelse(i == grid.Nx + 1, hᶠᶜᵃ + ηₑ, Hᶠᶜᵃ)
end

@inline function column_depthTᶜᶠᵃ(i, j, k, grid::AMGYL, η)
    Hᶜᶠᵃ = column_depthᶜᶠᵃ(i, j, k, grid, η)
    hᶜᶠᵃ = static_column_depthᶜᶠᵃ(i, j, grid)
    ηₑ = @inbounds η[i, grid.Ny, k]
    return ifelse(j == grid.Ny + 1, hᶜᶠᵃ + ηₑ, Hᶜᶠᵃ)
end

# Fallbacks
@inline σⁿ(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = σⁿ(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)
@inline σ⁻(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = σ⁻(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)

@inline ∂t_σ(i, j, k, ibg::IBG) = ∂t_σ(i, j, k, ibg.underlying_grid)

# Extend the 3D vertical spacing operators on an Immersed Mutable grid
const IMRG  = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:MRG}
const IMLLG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:MLLG}
const IMOSG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:MOSG}

for LX in (:ᶠ, :ᶜ), LY in (:ᶠ, :ᶜ), LZ in (:ᶠ, :ᶜ)
    zspacing = Symbol(:Δz, LX, LY, LZ)
    rspacing = Symbol(:Δr, LX, LY, LZ)

    ℓx = superscript_location(LX)
    ℓy = superscript_location(LY)
    ℓz = superscript_location(LZ)

    @eval begin
        using Oceananigans.Operators: $rspacing
        import Oceananigans.Operators: $zspacing

        @inline $zspacing(i, j, k, grid::IMRG)  = $rspacing(i, j, k, grid) * σⁿ(i, j, k, grid, $ℓx(), $ℓy(), $ℓz())
        @inline $zspacing(i, j, k, grid::IMLLG) = $rspacing(i, j, k, grid) * σⁿ(i, j, k, grid, $ℓx(), $ℓy(), $ℓz())
        @inline $zspacing(i, j, k, grid::IMOSG) = $rspacing(i, j, k, grid) * σⁿ(i, j, k, grid, $ℓx(), $ℓy(), $ℓz())
    end
end

#####
##### Chain-rule-correct horizontal derivatives for MutableImmersedGrid
#####
##### Forward to underlying grid which has the actual chain-rule implementation.
#####

import Oceananigans.Operators: ∂xᶠᶜᶜ, ∂xᶜᶜᶜ, ∂xᶠᶜᶠ, ∂xᶜᶠᶜ, ∂xᶠᶠᶜ
import Oceananigans.Operators: ∂yᶜᶠᶜ, ∂yᶜᶜᶜ, ∂yᶜᶠᶠ, ∂yᶠᶜᶜ, ∂yᶠᶠᶜ
import Oceananigans.Operators: ∂x_zᶠᶜᶜ, ∂x_zᶜᶜᶜ, ∂x_zᶠᶜᶠ, ∂x_zᶜᶠᶜ, ∂x_zᶠᶠᶜ, ∂x_zᶜᶜᶠ
import Oceananigans.Operators: ∂y_zᶜᶠᶜ, ∂y_zᶜᶜᶜ, ∂y_zᶜᶠᶠ, ∂y_zᶠᶜᶜ, ∂y_zᶠᶠᶜ, ∂y_zᶜᶜᶠ

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

# y-direction slopes at different staggerings
@inline ∂y_zᶜᶠᶜ(i, j, k, grid::AMG) = δyᶜᶠᶜ(i, j, k, grid, znode, C(), C(), C()) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
@inline ∂y_zᶜᶜᶜ(i, j, k, grid::AMG) = δyᶜᶜᶜ(i, j, k, grid, znode, C(), F(), C()) * Δy⁻¹ᶜᶜᶜ(i, j, k, grid)
@inline ∂y_zᶜᶠᶠ(i, j, k, grid::AMG) = δyᶜᶠᶠ(i, j, k, grid, znode, C(), C(), F()) * Δy⁻¹ᶜᶠᶠ(i, j, k, grid)
@inline ∂y_zᶠᶜᶜ(i, j, k, grid::AMG) = δyᶠᶜᶜ(i, j, k, grid, znode, F(), F(), C()) * Δy⁻¹ᶠᶜᶜ(i, j, k, grid)
@inline ∂y_zᶠᶠᶜ(i, j, k, grid::AMG) = δyᶠᶠᶜ(i, j, k, grid, znode, F(), C(), C()) * Δy⁻¹ᶠᶠᶜ(i, j, k, grid)
@inline ∂y_zᶜᶜᶠ(i, j, k, grid::AMG) = δyᶜᶜᶠ(i, j, k, grid, znode, C(), F(), F()) * Δy⁻¹ᶜᶜᶠ(i, j, k, grid)

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

# ∂xᶠᶜᶜ: tracer/buoyancy/pressure x-derivatives (most common)
@inline function ∂xᶠᶜᶜ(i, j, k, grid::AMG, ϕ)
    ∂x_at_r = δxᶠᶜᶜ(i, j, k, grid, ϕ) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶠᵃᶜ(i, j, k, grid, ∂zᶜᶜᶠ, ϕ)
    ∂x_z = ∂x_zᶠᶜᶜ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

@inline function ∂xᶠᶜᶜ(i, j, k, grid::AMG, f::Function, args...)
    ∂x_at_r = δxᶠᶜᶜ(i, j, k, grid, f, args...) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶠᵃᶜ(i, j, k, grid, ∂zᶜᶜᶠ, f, args...)
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
    ∂z_ϕ = ℑyzᵃᶠᶜ(i, j, k, grid, ∂zᶜᶜᶠ, ϕ)
    ∂y_z = ∂y_zᶜᶠᶜ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

@inline function ∂yᶜᶠᶜ(i, j, k, grid::AMG, f::Function, args...)
    ∂y_at_r = δyᶜᶠᶜ(i, j, k, grid, f, args...) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶠᶜ(i, j, k, grid, ∂zᶜᶜᶠ, f, args...)
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
