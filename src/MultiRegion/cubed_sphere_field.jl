using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Fields: AbstractField, FunctionField

# Flavors of CubedSphereField
const CubedSphereField{LX, LY, LZ} =
    Union{Field{LX, LY, LZ, <:Any, <:ConformalCubedSphereGrid},
          Field{LX, LY, LZ, <:Any, <:ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:ConformalCubedSphereGrid}}}

const CubedSphereFunctionField{LX, LY, LZ} =
    Union{FunctionField{LX, LY, LZ, <:Any, <:Any, <:Any, <:ConformalCubedSphereGrid},
          FunctionField{LX, LY, LZ, <:Any, <:Any, <:Any, <:ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:ConformalCubedSphereGrid}}}

const CubedSphereAbstractField{LX, LY, LZ} =
    Union{AbstractField{LX, LY, LZ, <:ConformalCubedSphereGrid},
          AbstractField{LX, LY, LZ, <:ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:ConformalCubedSphereGrid}}}

const AbstractCubedSphereField{LX, LY, LZ} =
    Union{CubedSphereAbstractField{LX, LY, LZ},
                  CubedSphereField{LX, LY, LZ}}

Base.summary(::AbstractCubedSphereField{LX, LY, LZ}) where {LX, LY, LZ} =
    "CubedSphereField{$LX, $LY, $LZ}"
