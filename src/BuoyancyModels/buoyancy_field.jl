using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: Field, ZeroField

buoyancy(model) = buoyancy(model.buoyancy, model)

buoyancy(::Nothing, model) = ZeroField()
buoyancy(::BuoyancyTracer, model) = model.tracers.b
buoyancy(b, model) = KernelFunctionOperation{Center, Center, Center}(buoyancy_perturbation, model.grid, computed_dependencies=(b.model, model.tracers))
 
BuoyancyField(model) = Field(buoyancy(model))
