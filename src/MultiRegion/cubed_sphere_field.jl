using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Fields: AbstractField, FunctionField

# Flavors of CubedSphereField
const CubedSphereField{LX, LY, LZ} =
    Union{Field{LX, LY, LZ, <:Nothing, <:ConformalCubedSphereGrid},
        Field{LX, LY, LZ, <:AbstractOperation, <:ConformalCubedSphereGrid}}

const CubedSphereFunctionField{LX, LY, LZ} =
    FunctionField{LX, LY, LZ, <:Any, <:Any, <:Any, <:ConformalCubedSphereGrid}

const CubedSphereAbstractField{LX, LY, LZ} =
    AbstractField{LX, LY, LZ, <:ConformalCubedSphereGrid}

const AbstractCubedSphereField{LX, LY, LZ} =
    Union{CubedSphereAbstractField{LX, LY, LZ},
                  CubedSphereField{LX, LY, LZ}}

function Field(f::CubedSphereField; indices = (:, :, :))
    view_of_f = Field(location(f), f.grid, indices = (:, :, indices[3]))
    return view_of_f
end

Base.summary(::AbstractCubedSphereField{LX, LY, LZ}) where {LX, LY, LZ} =
    "CubedSphereField{$LX, $LY, $LZ}"
