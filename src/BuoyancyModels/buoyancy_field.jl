using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: Field, ZeroField

buoyancy(::Nothing, args...) = ZeroField()
buoyancy(::BuoyancyTracer, grid, tracers) = tracers.b

# TODO: move to Models
buoyancy(model) = buoyancy(model.buoyancy, model.grid, model.tracers)
buoyancy(b, grid, tracers) = KernelFunctionOperation{Center, Center, Center}(buoyancy_perturbationᶜᶜᶜ, grid, b.model, tracers)
BuoyancyField(model) = Field(buoyancy(model))

buoyancy_frequency(b::Buoyancy, grid, tracers) = KernelFunctionOperation{Center, Center, Face}(∂z_b, grid, b.model, tracers)
buoyancy_frequency(b, grid, tracers)           = KernelFunctionOperation{Center, Center, Face}(∂z_b, grid, b, tracers)
buoyancy_frequency(model) = buoyancy_frequency(model.buoyancy, model.grid, model.tracers)

