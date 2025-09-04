using Adapt

import Oceananigans.Utils: prettysummary

struct ForcingKernelFunction{F}
    forcing :: F
end

Adapt.adapt_structure(to, fkf::ForcingKernelFunction) =
    ForcingKernelFunction(adapt(to, fkf.forcing))

prettysummary(kf::ForcingKernelFunction) = "ForcingKernelFunction"

@inline function (kf::ForcingKernelFunction)(i, j, k, grid, args...)
    return kf.forcing(i, j, k, grid, args...)
end

const ForcingOperation{LX, LY, LZ} =
    KernelFunctionOperation{LX, LY, LZ, <:Any, <:ForcingKernelFunction} where {LX, LY, LZ}

"""
    ForcingOperation(name::Symbol, model::AbstractModel)

Create a `KernelFunctionOperation` that evaluates the `model.forcing` for
prognostic variable `name`.

Example
=======

```jldoctest forcing_op
using Oceananigans
using Oceananigans.Models: ForcingOperation

grid = RectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1))

damping(x, y, z, t, c, τ) = - c / τ
c_forcing = Forcing(damping, field_dependencies=:c, parameters=60)
model = NonhydrostaticModel(; grid, tracers=:c, forcing=(; c=c_forcing))

c_forcing_op = ForcingOperation(:c, model)

# output
KernelFunctionOperation at (Center, Center, Center)
├── grid: 16×16×16 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: ForcingKernelFunction
└── arguments: ("Clock", "NamedTuple")
```

Next, we build a `ForcingField` for the damping, and compute it:

```jldoctest forcing_op
using Oceananigans.Models: ForcingField
set!(model, c=1)
c_forcing_field = ForcingField(:c, model)
compute!(c_forcing_field)

# output
16×16×16 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 16×16×16 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
├── operand: KernelFunctionOperation at (Center, Center, Center)
├── status: time=0.0
└── data: 22×22×22 OffsetArray(::Array{Float64, 3}, -2:19, -2:19, -2:19) with eltype Float64 with indices -2:19×-2:19×-2:19
    └── max=-0.0166667, min=-0.0166667, mean=-0.0166667
```
"""
function ForcingOperation(name::Symbol, model::AbstractModel)
    model_fields = fields(model)
    LX, LY, LZ = location(model_fields[name])
    forcing = getproperty(model.forcing, name)
    grid = model.grid
    args = (model.clock, model_fields)
    kernel_func = ForcingKernelFunction(forcing)
    return KernelFunctionOperation{LX, LY, LZ}(kernel_func, grid, args...)
end

const ForcingField{LX, LY, LZ} =
    Field{LX, LY, LZ, <:ForcingOperation} where {LX, LY, LZ}

function ForcingField(name::Symbol, model::AbstractModel)
    op = ForcingOperation(name, model)
    return Field(op)
end
