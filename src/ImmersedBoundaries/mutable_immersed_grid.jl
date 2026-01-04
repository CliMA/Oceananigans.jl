using Oceananigans.Grids: AbstractGeneralizedVerticalGrid, AbstractMutableGrid
using Oceananigans.Operators
using Oceananigans.Operators: GRG, GLLG, GOSG, superscript_location

import Oceananigans.Grids: column_depthᶜᶜᵃ,
                           column_depthᶜᶠᵃ,
                           column_depthᶠᶜᵃ,
                           column_depthᶠᶠᵃ

import Oceananigans.Operators: σⁿ, σ⁻, ∂t_σ

# Type aliases for grids with generalized vertical coordinates
const GeneralizedImmersedGrid   = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:AbstractGeneralizedVerticalGrid}
const GeneralizedGridOfSomeKind = Union{GeneralizedImmersedGrid, AbstractGeneralizedVerticalGrid}

# Backward compatibility aliases
const MutableImmersedGrid   = GeneralizedImmersedGrid
const MutableGridOfSomeKind = GeneralizedGridOfSomeKind

@inline column_depthᶜᶜᵃ(i, j, k, grid::GeneralizedGridOfSomeKind, η) = static_column_depthᶜᶜᵃ(i, j, grid) +  @inbounds η[i, j, k]
@inline column_depthᶠᶜᵃ(i, j, k, grid::GeneralizedGridOfSomeKind, η) = static_column_depthᶠᶜᵃ(i, j, grid) +  ℑxᶠᵃᵃ(i, j, k, grid, η)
@inline column_depthᶜᶠᵃ(i, j, k, grid::GeneralizedGridOfSomeKind, η) = static_column_depthᶜᶠᵃ(i, j, grid) +  ℑyᵃᶠᵃ(i, j, k, grid, η)
@inline column_depthᶠᶠᵃ(i, j, k, grid::GeneralizedGridOfSomeKind, η) = static_column_depthᶠᶠᵃ(i, j, grid) + ℑxyᶠᶠᵃ(i, j, k, grid, η)

# Convenience methods
@inline column_depthᶜᶜᵃ(i, j, grid) = static_column_depthᶜᶜᵃ(i, j, grid)
@inline column_depthᶜᶠᵃ(i, j, grid) = static_column_depthᶜᶠᵃ(i, j, grid)
@inline column_depthᶠᶜᵃ(i, j, grid) = static_column_depthᶠᶜᵃ(i, j, grid)
@inline column_depthᶠᶠᵃ(i, j, grid) = static_column_depthᶠᶠᵃ(i, j, grid)

@inline column_depthᶜᶜᵃ(i, j, grid::GeneralizedGridOfSomeKind) = column_depthᶜᶜᵃ(i, j, 1, grid, grid.z.ηⁿ)
@inline column_depthᶜᶠᵃ(i, j, grid::GeneralizedGridOfSomeKind) = column_depthᶜᶠᵃ(i, j, 1, grid, grid.z.ηⁿ)
@inline column_depthᶠᶜᵃ(i, j, grid::GeneralizedGridOfSomeKind) = column_depthᶠᶜᵃ(i, j, 1, grid, grid.z.ηⁿ)
@inline column_depthᶠᶠᵃ(i, j, grid::GeneralizedGridOfSomeKind) = column_depthᶠᶠᵃ(i, j, 1, grid, grid.z.ηⁿ)

# Three dimensional column_depth methods for use in `KernelOperations`
@inline column_depthᶜᶜᵃ(i, j, k, grid) = static_column_depthᶜᶜᵃ(i, j, grid)
@inline column_depthᶜᶠᵃ(i, j, k, grid) = static_column_depthᶜᶠᵃ(i, j, grid)
@inline column_depthᶠᶜᵃ(i, j, k, grid) = static_column_depthᶠᶜᵃ(i, j, grid)
@inline column_depthᶠᶠᵃ(i, j, k, grid) = static_column_depthᶠᶠᵃ(i, j, grid)

@inline column_depthᶜᶜᵃ(i, j, k, grid::GeneralizedGridOfSomeKind) = column_depthᶜᶜᵃ(i, j, 1, grid, grid.z.ηⁿ)
@inline column_depthᶜᶠᵃ(i, j, k, grid::GeneralizedGridOfSomeKind) = column_depthᶜᶠᵃ(i, j, 1, grid, grid.z.ηⁿ)
@inline column_depthᶠᶜᵃ(i, j, k, grid::GeneralizedGridOfSomeKind) = column_depthᶠᶜᵃ(i, j, 1, grid, grid.z.ηⁿ)
@inline column_depthᶠᶠᵃ(i, j, k, grid::GeneralizedGridOfSomeKind) = column_depthᶠᶠᵃ(i, j, 1, grid, grid.z.ηⁿ)

# Fallbacks
@inline σⁿ(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = σⁿ(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)
@inline σ⁻(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = σ⁻(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)

@inline ∂t_σ(i, j, k, ibg::IBG) = ∂t_σ(i, j, k, ibg.underlying_grid)

# Extend the 3D vertical spacing operators on an Immersed Mutable grid
const IGRG  = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:GRG}
const IGLLG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:GLLG}
const IGOSG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:GOSG}

for LX in (:ᶠ, :ᶜ), LY in (:ᶠ, :ᶜ), LZ in (:ᶠ, :ᶜ)
    zspacing = Symbol(:Δz, LX, LY, LZ)
    rspacing = Symbol(:Δr, LX, LY, LZ)

    ℓx = superscript_location(LX)
    ℓy = superscript_location(LY)
    ℓz = superscript_location(LZ)

    @eval begin
        using Oceananigans.Operators: $rspacing
        import Oceananigans.Operators: $zspacing

        @inline $zspacing(i, j, k, grid::IGRG)  = $rspacing(i, j, k, grid) * σⁿ(i, j, k, grid, $ℓx(), $ℓy(), $ℓz())
        @inline $zspacing(i, j, k, grid::IGLLG) = $rspacing(i, j, k, grid) * σⁿ(i, j, k, grid, $ℓx(), $ℓy(), $ℓz())
        @inline $zspacing(i, j, k, grid::IGOSG) = $rspacing(i, j, k, grid) * σⁿ(i, j, k, grid, $ℓx(), $ℓy(), $ℓz())
    end
end

#####
##### Chain-rule-correct horizontal derivatives for GeneralizedImmersedGrid
#####
##### These forward to the underlying grid which has the actual implementation.
#####

import Oceananigans.Operators: ∂xᶠᶜᶜ, ∂xᶜᶜᶜ, ∂xᶠᶠᶜ, ∂yᶜᶠᶜ, ∂yᶜᶜᶜ, ∂yᶠᶠᶜ

# x-derivatives
@inline ∂xᶠᶜᶜ(i, j, k, ibg::GeneralizedImmersedGrid, c) = ∂xᶠᶜᶜ(i, j, k, ibg.underlying_grid, c)
@inline ∂xᶠᶜᶜ(i, j, k, ibg::GeneralizedImmersedGrid, f::Function, args...) = ∂xᶠᶜᶜ(i, j, k, ibg.underlying_grid, f, args...)

@inline ∂xᶜᶜᶜ(i, j, k, ibg::GeneralizedImmersedGrid, c) = ∂xᶜᶜᶜ(i, j, k, ibg.underlying_grid, c)
@inline ∂xᶜᶜᶜ(i, j, k, ibg::GeneralizedImmersedGrid, f::Function, args...) = ∂xᶜᶜᶜ(i, j, k, ibg.underlying_grid, f, args...)

@inline ∂xᶠᶠᶜ(i, j, k, ibg::GeneralizedImmersedGrid, c) = ∂xᶠᶠᶜ(i, j, k, ibg.underlying_grid, c)
@inline ∂xᶠᶠᶜ(i, j, k, ibg::GeneralizedImmersedGrid, f::Function, args...) = ∂xᶠᶠᶜ(i, j, k, ibg.underlying_grid, f, args...)

# y-derivatives
@inline ∂yᶜᶠᶜ(i, j, k, ibg::GeneralizedImmersedGrid, c) = ∂yᶜᶠᶜ(i, j, k, ibg.underlying_grid, c)
@inline ∂yᶜᶠᶜ(i, j, k, ibg::GeneralizedImmersedGrid, f::Function, args...) = ∂yᶜᶠᶜ(i, j, k, ibg.underlying_grid, f, args...)

@inline ∂yᶜᶜᶜ(i, j, k, ibg::GeneralizedImmersedGrid, c) = ∂yᶜᶜᶜ(i, j, k, ibg.underlying_grid, c)
@inline ∂yᶜᶜᶜ(i, j, k, ibg::GeneralizedImmersedGrid, f::Function, args...) = ∂yᶜᶜᶜ(i, j, k, ibg.underlying_grid, f, args...)

@inline ∂yᶠᶠᶜ(i, j, k, ibg::GeneralizedImmersedGrid, c) = ∂yᶠᶠᶜ(i, j, k, ibg.underlying_grid, c)
@inline ∂yᶠᶠᶜ(i, j, k, ibg::GeneralizedImmersedGrid, f::Function, args...) = ∂yᶠᶠᶜ(i, j, k, ibg.underlying_grid, f, args...)
