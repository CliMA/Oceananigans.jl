using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Fields: AbstractField, FunctionField

# Flavors of CubedSphereField
const CubedSphereField{LX, LY, LZ} =
    Union{Field{LX, LY, LZ, <:Nothing, <:ConformalCubedSphereGrid},
          Field{LX, LY, LZ, <:AbstractOperation, <:ConformalCubedSphereGrid},
          Field{LX, LY, LZ, <:Nothing, <:ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:ConformalCubedSphereGrid}},
          Field{LX, LY, LZ, <:AbstractOperation, <:ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:ConformalCubedSphereGrid}}}

const CubedSphereFunctionField{LX, LY, LZ} =
    FunctionField{LX, LY, LZ, <:Any, <:Any, <:Any, <:ConformalCubedSphereGrid}

const CubedSphereAbstractField{LX, LY, LZ} =
    AbstractField{LX, LY, LZ, <:ConformalCubedSphereGrid}

const AbstractCubedSphereField{LX, LY, LZ} =
    Union{CubedSphereAbstractField{LX, LY, LZ},
                  CubedSphereField{LX, LY, LZ}}

Field(f::CubedSphereField; args...) = f

Base.summary(::AbstractCubedSphereField{LX, LY, LZ}) where {LX, LY, LZ} =
    "CubedSphereField{$LX, $LY, $LZ}"
