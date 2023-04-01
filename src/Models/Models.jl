module Models

export
    NonhydrostaticModel,
    ShallowWaterModel, ConservativeFormulation, VectorInvariantFormulation,
    SingleLayerModel, MultiLayerModel,
    HydrostaticFreeSurfaceModel,
    ExplicitFreeSurface, ImplicitFreeSurface, SplitExplicitFreeSurface,
    PrescribedVelocityFields, PressureField

using Oceananigans: AbstractModel
using Oceananigans.Grids: halo_size, inflate_halo_size

import Oceananigans: initialize!
import Oceananigans.Architectures: device_event, architecture

device_event(model::AbstractModel) = device_event(model.architecture)
architecture(model::AbstractModel) = model.architecture

initialize!(model::AbstractModel) = nothing

using Oceananigans: fields
import Oceananigans.TimeSteppers: reset!

function reset!(model::AbstractModel)

    for field in fields(model)
        fill!(field, 0.0)
    end

    for field in model.timestepper.G⁻
        fill!(field, 0.0)
    end

    for field in model.timestepper.Gⁿ
        fill!(field, 0.0)
    end
    
    return nothing
end


abstract type AbstractNonhydrostaticModel{TS} <: AbstractModel{TS} end

function validate_model_halo(grid, momentum_advection, tracer_advection, closure)
    user_halo = halo_size(grid)
    required_halo = inflate_halo_size(1, 1, 1, grid,
                                      momentum_advection,
                                      tracer_advection,
                                      closure)

    any(user_halo .< required_halo) &&
        throw(ArgumentError("The grid halo $user_halo must be at least equal to $required_halo. Note that an ImmersedBoundaryGrid requires an extra halo point in all non-flat directions compared to a non-immersed boundary grid."))
end

include("NonhydrostaticModels/NonhydrostaticModels.jl")
include("HydrostaticFreeSurfaceModels/HydrostaticFreeSurfaceModels.jl")
include("ShallowWaterModels/ShallowWaterModels.jl")

using .NonhydrostaticModels: NonhydrostaticModel, PressureField

using .HydrostaticFreeSurfaceModels:
    HydrostaticFreeSurfaceModel,
    ExplicitFreeSurface, ImplicitFreeSurface, SplitExplicitFreeSurface,
    PrescribedVelocityFields

using .ShallowWaterModels: ShallowWaterModel, 
    ConservativeFormulation, VectorInvariantFormulation, 
    SingleLayerModel, MultiLayerModel

end # module
