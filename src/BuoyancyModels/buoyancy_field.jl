using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: Field, ZeroField


buoyancy(::Nothing, args...) = ZeroField()
buoyancy(::BuoyancyTracer, grid, tracers) = tracers.b

buoyancy(model) = buoyancy(model.buoyancy, model.grid, model.tracers)

buoyancy(b, grid, tracers) = KernelFunctionOperation{Center, Center, Center}(buoyancy_perturbation,
                                                                             grid,
                                                                             computed_dependencies=(b.model, tracers))

BuoyancyField(model) = Field(buoyancy(model))
