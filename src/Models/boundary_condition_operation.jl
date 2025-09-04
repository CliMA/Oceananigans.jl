using Oceananigans: fields
using Oceananigans.BoundaryConditions: getbc, bc_str
using Oceananigans.Fields: Field, location
using Oceananigans.AbstractOperations: KernelFunctionOperation

import Oceananigans.Utils: prettysummary

struct BoundaryConditionKernelFunction{Side, BC}
    bc :: BC
end

function prettysummary(kf::BoundaryConditionKernelFunction{Side}) where Side
    return string("BoundaryConditionKernelFunction{", Side, "}(", bc_str(kf.bc), ")")
end

const BottomTopFunc  = Union{BoundaryConditionKernelFunction{:bottom}, BoundaryConditionKernelFunction{:top}}
const WestEastFunc   = Union{BoundaryConditionKernelFunction{:west},   BoundaryConditionKernelFunction{:east}}
const SouthNorthFunc = Union{BoundaryConditionKernelFunction{:south},  BoundaryConditionKernelFunction{:north}}
const ImmersedFunc   = BoundaryConditionKernelFunction{:immersed}

@inline  (kf::BottomTopFunc)(i, j, k, grid, args...) = getbc(kf.bc, i, j, grid, args...)
@inline   (kf::WestEastFunc)(i, j, k, grid, args...) = getbc(kf.bc, j, k, grid, args...)
@inline (kf::SouthNorthFunc)(i, j, k, grid, args...) = getbc(kf.bc, i, k, grid, args...)
@inline   (kf::ImmersedFunc)(i, j, k, grid, args...) = getbc(kf.bc, i, j, k, grid, args...)

boundary_condition_args(model::Union{NonhydrostaticModel, HydrostaticFreeSurfaceModel}) = (model.clock, fields(model))

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
 
"""
    BoundaryConditionOperation(field::Field, side::Symbol, model::AbstractModel)

Returns a `KernelFunctionOperation` that evaluates a `field`'s boundary condition
on the specified `side` using the properties of `model`.

Example
=======

Build a `BoundaryConditionOperation` for a top flux boundary condition:

```jldoctest bc_op
using Oceananigans
using Oceananigans.Models: BoundaryConditionOperation

grid = RectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1))

c_flux(x, y, t) = sin(2π * x)
c_top_bc = FluxBoundaryCondition(c_flux)
c_bcs = FieldBoundaryConditions(top=c_top_bc)
model = NonhydrostaticModel(; grid, tracers=:c, boundary_conditions=(; c=c_bcs))

c_flux_op = BoundaryConditionOperation(model.tracers.c, :top, model)

# output
KernelFunctionOperation at (Center, Center, ⋅)
├── grid: 16×16×16 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: BoundaryConditionKernelFunction{top}(Flux)
└── arguments: ("Clock", "NamedTuple")
```

Next, we build a `BoundaryConditionField` for the top flux, and compute it:

```jldoctest bc_op
using Oceananigans.Models: BoundaryConditionField
c_flux_field = BoundaryConditionField(model.tracers.c, :top, model)
compute!(c_flux_field)

# output
16×16×1 Field{Center, Center, Nothing} reduced over dims = (3,) on RectilinearGrid on CPU
├── grid: 16×16×16 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: Nothing, top: Nothing, immersed: Nothing
├── operand: KernelFunctionOperation at (Center, Center, ⋅)
├── status: time=0.0
└── data: 22×22×1 OffsetArray(::Array{Float64, 3}, -2:19, -2:19, 1:1) with eltype Float64 with indices -2:19×-2:19×1:1
    └── max=0.980785, min=-0.980785, mean=1.0842e-19
```
"""
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
