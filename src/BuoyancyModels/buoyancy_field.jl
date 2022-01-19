using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: Field, ZeroField

function BuoyancyField(model)
    isnothing(model.buoyancy) && return ZeroField()

    grid = model.grid
    buoyancy = model.buoyancy
    tracers = model.tracers

    op = KernelFunctionOperation{Center, Center, Center}(buoyancy_perturbation, grid,
                                                         computed_dependencies=(buoyancy.model, tracers))

    return Field(op)
end

