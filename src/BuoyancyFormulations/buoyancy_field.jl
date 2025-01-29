using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: Field, ZeroField

buoyancy(::Nothing, args...) = ZeroField()
buoyancy(::BuoyancyTracer, grid, tracers) = tracers.b

# TODO: move to Models
buoyancy(model) = buoyancy(model.buoyancy, model.grid, model.tracers)
buoyancy(bf::BuoyancyForce, grid, tracers) = buoyancy(bf.formulation, grid, tracers)

buoyancy(bm::AbstractBuoyancyFormulation, grid, tracers) =
    KernelFunctionOperation{Center, Center, Center}(buoyancy_perturbationᶜᶜᶜ, grid, bm, tracers)

BuoyancyField(model) = Field(buoyancy(model))

buoyancy_frequency(model) = buoyancy_frequency(model.buoyancy, model.grid, model.tracers)
buoyancy_frequency(b::BuoyancyForce, grid, tracers) = buoyancy_frequency(b.formulation, grid, tracers)
buoyancy_frequency(bm::AbstractBuoyancyFormulation, grid, tracers) =
    KernelFunctionOperation{Center, Center, Face}(∂z_b, grid, bm, tracers)

