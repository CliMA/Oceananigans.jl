module Models

export
    NonhydrostaticModel,
    ShallowWaterModel, ConservativeFormulation, VectorInvariantFormulation,
    HydrostaticFreeSurfaceModel,
    ExplicitFreeSurface, ImplicitFreeSurface, SplitExplicitFreeSurface,
    PrescribedVelocityFields, PressureField

using Oceananigans: AbstractModel

import Oceananigans.Architectures: architecture

architecture(model::AbstractModel) = model.architecture

initialize_model!(model::AbstractModel) = nothing

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

include("NonhydrostaticModels/NonhydrostaticModels.jl")
include("HydrostaticFreeSurfaceModels/HydrostaticFreeSurfaceModels.jl")
include("ShallowWaterModels/ShallowWaterModels.jl")

using .NonhydrostaticModels: NonhydrostaticModel, PressureField

using .HydrostaticFreeSurfaceModels:
    HydrostaticFreeSurfaceModel,
    ExplicitFreeSurface, ImplicitFreeSurface, SplitExplicitFreeSurface,
    PrescribedVelocityFields

using .ShallowWaterModels: ShallowWaterModel, ConservativeFormulation, VectorInvariantFormulation

end # module
