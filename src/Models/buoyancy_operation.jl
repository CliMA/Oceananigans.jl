using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: Field, ZeroField
using Oceananigans.BuoyancyFormulations:
    BuoyancyForce,
    BuoyancyTracer,
    AbstractBuoyancyFormulation,
    ∂z_b,
    buoyancy_perturbationᶜᶜᶜ

"""
    buoyancy_operation(model)

Return a `KernelFunctionOperation` that computes the buoyancy perturbation,
`Field` corresponding to the buoyancy perturbation directly, or `ZeroField` if buoyancy is `nothing`.
"""
buoyancy_operation(model) = buoyancy_operation(model.buoyancy, model.grid, model.tracers)
buoyancy_operation(bf::BuoyancyForce, grid, tracers) = buoyancy_operation(bf.formulation, grid, tracers)

buoyancy_operation(::Nothing, args...) = ZeroField()
buoyancy_operation(::BuoyancyTracer, grid, tracers) = tracers.b

buoyancy_operation(bm::AbstractBuoyancyFormulation, grid, tracers) =
    KernelFunctionOperation{Center, Center, Center}(buoyancy_perturbationᶜᶜᶜ, grid, bm, tracers)

"""
    buoyancy_field(model)

Return a `Field` that can `compute!` and store the buoyancy perturbation.
"""
buoyancy_field(model) = Field(buoyancy_operation(model))

"""
    buoyancy_frequency(model)

Returns a `KernelFunctionOperation` that computes the vertical derivative of buoyancy,
which is also known as the square of the buoyancy frequency.
"""
buoyancy_frequency(model) = buoyancy_frequency(model.buoyancy, model.grid, model.tracers)
buoyancy_frequency(b::BuoyancyForce, grid, tracers) = buoyancy_frequency(b.formulation, grid, tracers)
buoyancy_frequency(bm::AbstractBuoyancyFormulation, grid, tracers) =
    KernelFunctionOperation{Center, Center, Face}(∂z_b, grid, bm, tracers)
