module ShallowWaterModels

export ShallowWaterModel, ShallowWaterScalarDiffusivity,
       ConservativeFormulation, VectorInvariantFormulation

using KernelAbstractions: @index, @kernel

using Adapt
using Oceananigans.Utils: launch!

import Oceananigans: fields, prognostic_fields
import Oceananigans.Simulations: timestepper
import Oceananigans.Models: ForcingOperation, boundary_condition_args
import Oceananigans: initialize!
import Oceananigans.TimeSteppers: reset!

#####
##### ShallowWaterModel definition
#####

include("shallow_water_model.jl")
include("set_shallow_water_model.jl")
include("show_shallow_water_model.jl")

#####
##### Time-stepping ShallowWaterModels
#####

"""
    fields(model::ShallowWaterModel)

Return a flattened `NamedTuple` of the fields in `model.solution` and `model.tracers` for
a `ShallowWaterModel` model.
"""
fields(model::ShallowWaterModel) = merge(model.solution, model.tracers)

"""
    prognostic_fields(model::HydrostaticFreeSurfaceModel)

Return a flattened `NamedTuple` of the prognostic fields associated with `ShallowWaterModel`.
"""
prognostic_fields(model::ShallowWaterModel) = fields(model)

function ForcingOperation(name::Symbol, model::ShallowWaterModel)
    model_fields = shallow_water_fields(model.velocities, model.solution, model.tracers, model.formulation)
    LX, LY, LZ = Oceananigans.Fields.location(model_fields[name])
    forcing = getproperty(model.forcing, name)
    grid = model.grid
    args = (model.clock, model_fields)
    kernel_func = Oceananigans.Models.ForcingKernelFunction(forcing)
    return Oceananigans.AbstractOperations.KernelFunctionOperation{LX, LY, LZ}(kernel_func, grid, args...)
end

boundary_condition_args(model::ShallowWaterModel) =
    (model.clock,
     shallow_water_fields(model.velocities, model.solution, model.tracers, model.formulation))

include("solution_and_tracer_tendencies.jl")
include("compute_shallow_water_tendencies.jl")
include("shallow_water_rk3_substep.jl")
include("shallow_water_ab2_step.jl")
include("update_shallow_water_state.jl")

function initialize!(model::ShallowWaterModel)
    update_state!(model)
    Oceananigans.TurbulenceClosures.initialize_closure_fields!(model.closure_fields, model.closure, model)
    return nothing
end

function reset!(model::ShallowWaterModel)
    for field in fields(model)
        fill!(field, 0)
    end

    for field in model.timestepper.G⁻
        fill!(field, 0)
    end

    for field in model.timestepper.Gⁿ
        fill!(field, 0)
    end

    update_state!(model)
    Oceananigans.TurbulenceClosures.initialize_closure_fields!(model.closure_fields, model.closure, model)
    return nothing
end

include("cache_shallow_water_tendencies.jl")
include("shallow_water_advection_operators.jl")
include("shallow_water_diffusion_operators.jl")
include("shallow_water_cell_advection_timescale.jl")

end # module
