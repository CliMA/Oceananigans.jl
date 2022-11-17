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

#####
##### extract_boundary_conditions:
#####
##### Recursive util for building NamedTuples of boundary conditions from NamedTuples of fields.
#####
##### Note: ignores tuples, including tuples of Symbols (tracer names) and
##### tuples of DiffusivityFields (which occur for tupled closures)
#####

extract_boundary_conditions(::Nothing) = NamedTuple()
extract_boundary_conditions(::Tuple) = NamedTuple()

function extract_boundary_conditions(field_tuple::NamedTuple)
    names = propertynames(field_tuple)
    bcs = Tuple(extract_boundary_conditions(field) for field in field_tuple)
    return NamedTuple{names}(bcs)
end

extract_boundary_conditions(field::Field) = field.boundary_conditions

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
