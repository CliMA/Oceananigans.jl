using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: Field
using Adapt
using KernelAbstractions

function buoyancy_operation(model)
    buoyancy_model = model.buoyancy.model
    tracers = model.tracers
    return buoyancy_operation(buoyancy_model, model.grid, tracers)
end

buoyancy_operation(buoyancy_model, grid, tracers) =
    KernelFunctionOperation{Center, Center, Center}(buoyancy_perturbation, grid, computed_dependencies=(buoyancy_model, tracers))

buoyancy_operation(::Nothing, grid, tracers) = nothing

function BuoyancyField(model)
    op = buoyancy_operation(model)
    return Field(op)
end

