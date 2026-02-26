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

# x-derivatives
@inline ∂xᶠᶜᶜ(i, j, k, ibg::MutableImmersedGrid, ϕ) = ∂xᶠᶜᶜ(i, j, k, ibg.underlying_grid, ϕ)
@inline ∂xᶠᶜᶜ(i, j, k, ibg::MutableImmersedGrid, f::Function, args...) = ∂xᶠᶜᶜ(i, j, k, ibg.underlying_grid, f, args...)
@inline ∂xᶠᶜᶜ(i, j, k, ibg::MutableImmersedGrid, c::Number) = zero(ibg)

@inline ∂xᶜᶜᶜ(i, j, k, ibg::MutableImmersedGrid, ϕ) = ∂xᶜᶜᶜ(i, j, k, ibg.underlying_grid, ϕ)
@inline ∂xᶜᶜᶜ(i, j, k, ibg::MutableImmersedGrid, f::Function, args...) = ∂xᶜᶜᶜ(i, j, k, ibg.underlying_grid, f, args...)
@inline ∂xᶜᶜᶜ(i, j, k, ibg::MutableImmersedGrid, c::Number) = zero(ibg)

@inline ∂xᶠᶜᶠ(i, j, k, ibg::MutableImmersedGrid, ϕ) = ∂xᶠᶜᶠ(i, j, k, ibg.underlying_grid, ϕ)
@inline ∂xᶠᶜᶠ(i, j, k, ibg::MutableImmersedGrid, f::Function, args...) = ∂xᶠᶜᶠ(i, j, k, ibg.underlying_grid, f, args...)
@inline ∂xᶠᶜᶠ(i, j, k, ibg::MutableImmersedGrid, c::Number) = zero(ibg)

@inline ∂xᶜᶠᶜ(i, j, k, ibg::MutableImmersedGrid, ϕ) = ∂xᶜᶠᶜ(i, j, k, ibg.underlying_grid, ϕ)
@inline ∂xᶜᶠᶜ(i, j, k, ibg::MutableImmersedGrid, f::Function, args...) = ∂xᶜᶠᶜ(i, j, k, ibg.underlying_grid, f, args...)
@inline ∂xᶜᶠᶜ(i, j, k, ibg::MutableImmersedGrid, c::Number) = zero(ibg)

@inline ∂xᶠᶠᶜ(i, j, k, ibg::MutableImmersedGrid, ϕ) = ∂xᶠᶠᶜ(i, j, k, ibg.underlying_grid, ϕ)
@inline ∂xᶠᶠᶜ(i, j, k, ibg::MutableImmersedGrid, f::Function, args...) = ∂xᶠᶠᶜ(i, j, k, ibg.underlying_grid, f, args...)
@inline ∂xᶠᶠᶜ(i, j, k, ibg::MutableImmersedGrid, c::Number) = zero(ibg)

# y-derivatives
@inline ∂yᶜᶠᶜ(i, j, k, ibg::MutableImmersedGrid, ϕ) = ∂yᶜᶠᶜ(i, j, k, ibg.underlying_grid, ϕ)
@inline ∂yᶜᶠᶜ(i, j, k, ibg::MutableImmersedGrid, f::Function, args...) = ∂yᶜᶠᶜ(i, j, k, ibg.underlying_grid, f, args...)
@inline ∂yᶜᶠᶜ(i, j, k, ibg::MutableImmersedGrid, c::Number) = zero(ibg)

@inline ∂yᶜᶜᶜ(i, j, k, ibg::MutableImmersedGrid, ϕ) = ∂yᶜᶜᶜ(i, j, k, ibg.underlying_grid, ϕ)
@inline ∂yᶜᶜᶜ(i, j, k, ibg::MutableImmersedGrid, f::Function, args...) = ∂yᶜᶜᶜ(i, j, k, ibg.underlying_grid, f, args...)
@inline ∂yᶜᶜᶜ(i, j, k, ibg::MutableImmersedGrid, c::Number) = zero(ibg)

@inline ∂yᶜᶠᶠ(i, j, k, ibg::MutableImmersedGrid, ϕ) = ∂yᶜᶠᶠ(i, j, k, ibg.underlying_grid, ϕ)
@inline ∂yᶜᶠᶠ(i, j, k, ibg::MutableImmersedGrid, f::Function, args...) = ∂yᶜᶠᶠ(i, j, k, ibg.underlying_grid, f, args...)
@inline ∂yᶜᶠᶠ(i, j, k, ibg::MutableImmersedGrid, c::Number) = zero(ibg)

@inline ∂yᶠᶜᶜ(i, j, k, ibg::MutableImmersedGrid, ϕ) = ∂yᶠᶜᶜ(i, j, k, ibg.underlying_grid, ϕ)
@inline ∂yᶠᶜᶜ(i, j, k, ibg::MutableImmersedGrid, f::Function, args...) = ∂yᶠᶜᶜ(i, j, k, ibg.underlying_grid, f, args...)
@inline ∂yᶠᶜᶜ(i, j, k, ibg::MutableImmersedGrid, c::Number) = zero(ibg)

@inline ∂yᶠᶠᶜ(i, j, k, ibg::MutableImmersedGrid, ϕ) = ∂yᶠᶠᶜ(i, j, k, ibg.underlying_grid, ϕ)
@inline ∂yᶠᶠᶜ(i, j, k, ibg::MutableImmersedGrid, f::Function, args...) = ∂yᶠᶠᶜ(i, j, k, ibg.underlying_grid, f, args...)
@inline ∂yᶠᶠᶜ(i, j, k, ibg::MutableImmersedGrid, c::Number) = zero(ibg)
