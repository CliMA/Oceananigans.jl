using Oceananigans: fields
using Oceananigans.BoundaryConditions: getbc
using Oceananigans.Fields: Field, location
using Oceananigans.AbstractOperations: KernelFunctionOperation

struct BoundaryConditionKernelFunction{Side, BC}
    bc :: BC
end

const BottomTopFunc  = Union{BoundaryConditionKernelFunction{:bottom},   BoundaryConditionKernelFunction{:top}}
const WestEastFunc   = Union{BoundaryConditionKernelFunction{:west},  BoundaryConditionKernelFunction{:east}}
const SouthNorthFunc = Union{BoundaryConditionKernelFunction{:south}, BoundaryConditionKernelFunction{:north}}
const ImmersedFunc   = BoundaryConditionKernelFunction{:immersed}

@inline  (kf::BottomTopFunc)(i, j, k, grid, args...) = getbc(kf.bc, i, j, grid, args...)
@inline   (kf::WestEastFunc)(i, j, k, grid, args...) = getbc(kf.bc, j, k, grid, args...)
@inline (kf::SouthNorthFunc)(i, j, k, grid, args...) = getbc(kf.bc, i, k, grid, args...)
@inline   (kf::ImmersedFunc)(i, j, k, grid, args...) = getbc(kf.bc, i, j, k, grid, args...)

boundary_condition_args(model::Union{NonHydrostaticModel, HydrostaticFreeSurfaceModel}) = (model.clock, fields(model))

function boundary_condition_location(side, LX, LY, LZ)
    if side == :top || side == :bottom
        return LX, LY, Nothing
    elseif side == :west || side == :east
        return Nothing, LY, LZ
    elseif side == :south || side == :north
        return LX, Nothing, LZ
    elseif side == :immersed
        return LX, LY, LZ
    end
end

const BoundaryConditionOperation{LX, LY, LZ} =
    KernelFunctionOperation{LX, LY, LZ, <:Any, <:Any, <:BoundaryConditionKernelFunction} where {LX, LY, LZ}

const BoundaryConditionField{LX, LY, LZ} =
    Field{LX, LY, LZ, <:BoundaryConditionOperation} where {LX, LY, LZ}
 
function BoundaryConditionOperation(field::Field, side::Symbol, model::AbstractModel)
    grid = field.grid
    args = boundary_condition_args(model)
    LX, LY, LZ = boundary_condition_location(side, location(field)...)
    bc = getproperty(field.boundary_conditions, side)
    kernel_func = BoundaryConditionKernelFunction{side, typeof(bc)}(bc)
    return KernelFunctionOperation{LX, LY, LZ}(kernel_func, grid, args...)
end

function BoundaryConditionField(field::Field, side::Symbol, model::AbstractModel)
    op = BoundaryConditionOperation(field, side, model)
    return Field(op)
end
