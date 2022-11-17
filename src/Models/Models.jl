module Models

export
    NonhydrostaticModel,
    ShallowWaterModel, ConservativeFormulation, VectorInvariantFormulation,
    HydrostaticFreeSurfaceModel,
    ExplicitFreeSurface, ImplicitFreeSurface, SplitExplicitFreeSurface,
    PrescribedVelocityFields, PressureField

using Oceananigans.Grids: halo_size, inflate_halo_size, with_halo
using Oceananigans: AbstractModel

import Oceananigans.Architectures: device_event, architecture

device_event(model::AbstractModel) = device_event(model.architecture)
architecture(model::AbstractModel) = model.architecture

abstract type AbstractNonhydrostaticModel{TS} <: AbstractModel{TS} end

#####
##### Halo validation for models
#####

function validate_halo(grid, tendency_terms...)
    user_halo = halo_size(grid)
    required_halo = inflate_halo_size(1, 1, 1, grid, tendency_terms...)

    any(user_halo .< required_halo) &&
      throw(ArgumentError("The grid halo $user_halo must be at least equal to $required_halo. " *
                          "Note that an ImmersedBoundaryGrid requires an extra halo point."))
end

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
