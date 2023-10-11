module Models

export
    NonhydrostaticModel,
    ShallowWaterModel, ConservativeFormulation, VectorInvariantFormulation,
    HydrostaticFreeSurfaceModel,
    ExplicitFreeSurface, ImplicitFreeSurface, SplitExplicitFreeSurface,
    PrescribedVelocityFields, PressureField,
    LagrangianParticles

using Oceananigans.Utils
using Oceananigans: AbstractModel
using Oceananigans.Advection: AbstractAdvectionScheme, CenteredSecondOrder, VectorInvariant
using Oceananigans.Grids: halo_size, inflate_halo_size
using Oceananigans.Fields: Field
using Oceananigans: fields, prognostic_fields

import Oceananigans: initialize!
import Oceananigans.Architectures: architecture
import Oceananigans.TimeSteppers: reset!

# A prototype interface for AbstractModel.
# 
# TODO: decide if we like this.
#
# We assume that model has some properties, eg:
#   - model.clock::Clock
#   - model.architecture.
#   - model.timestepper with timestepper.G⁻ and timestepper.Gⁿ :spiral_eyes:
#
# Perhaps this is a little unclean.

iteration(model::AbstractModel) = model.clock.iteration
Base.time(model::AbstractModel) = model.clock.time
architecture(model::AbstractModel) = model.architecture
initialize!(model::AbstractModel) = nothing
total_velocities(model::AbstractModel) = nothing
timestepper(model::AbstractModel) = model.timestepper

function reset!(model::AbstractModel)

    for field in fields(model)
        fill!(field, 0)
    end

    # TODO: abstract this better to other time-steppers
    for field in model.timestepper.G⁻
        fill!(field, 0)
    end

    for field in model.timestepper.Gⁿ
        fill!(field, 0)
    end
    
    return nothing
end

#####
##### Model-building utilities
#####

function validate_model_halo(grid, momentum_advection, tracer_advection, closure)
    user_halo = halo_size(grid)
    required_halo = inflate_halo_size(1, 1, 1, grid,
                                      momentum_advection,
                                      tracer_advection,
                                      closure)

    any(user_halo .< required_halo) &&
        throw(ArgumentError("The grid halo $user_halo must be at least equal to $required_halo. \
                            Note that an ImmersedBoundaryGrid requires an extra halo point in all \
                            non-flat directions compared to a non-immersed boundary grid."))
end

#
# Recursive util for building NamedTuples of boundary conditions from NamedTuples of fields
#
# Note: ignores tuples, including tuples of Symbols (tracer names) and
# tuples of DiffusivityFields (which occur for tupled closures)

extract_boundary_conditions(::Nothing) = NamedTuple()
extract_boundary_conditions(::Tuple) = NamedTuple()

function extract_boundary_conditions(field_tuple::NamedTuple)
    names = propertynames(field_tuple)
    bcs = Tuple(extract_boundary_conditions(field) for field in field_tuple)
    return NamedTuple{names}(bcs)
end

extract_boundary_conditions(field::Field) = field.boundary_conditions

# Util for validation tracer advection schemes

""" Returns a default_tracer_advection, tracer_advection `tuple`. """
validate_tracer_advection(invalid_tracer_advection, grid) = error("$invalid_tracer_advection is invalid tracer_advection!")
validate_tracer_advection(tracer_advection_tuple::NamedTuple, grid) = CenteredSecondOrder(), tracer_advection_tuple
validate_tracer_advection(tracer_advection::AbstractAdvectionScheme, grid) = tracer_advection, NamedTuple()
validate_tracer_advection(tracer_advection::Nothing, grid) = nothing, NamedTuple()

# Util for checking whether the model's prognostic state has NaN'd
include("nan_checker.jl")

# Communication - Computation overlap in distributed models
include("interleave_communication_and_computation.jl")

#####
##### All the code
#####

include("NonhydrostaticModels/NonhydrostaticModels.jl")
include("HydrostaticFreeSurfaceModels/HydrostaticFreeSurfaceModels.jl")
include("ShallowWaterModels/ShallowWaterModels.jl")
include("LagrangianParticleTracking/LagrangianParticleTracking.jl")

using .NonhydrostaticModels: NonhydrostaticModel, PressureField

using .HydrostaticFreeSurfaceModels:
    HydrostaticFreeSurfaceModel,
    ExplicitFreeSurface, ImplicitFreeSurface, SplitExplicitFreeSurface,
    PrescribedVelocityFields

using .ShallowWaterModels: ShallowWaterModel, ConservativeFormulation, VectorInvariantFormulation

using .LagrangianParticleTracking: LagrangianParticles

#####
##### Stuff common to all "Oceananigans models" (eg the models implemented here) goes below
#####

const OceananigansModels = Union{NonhydrostaticModel,
                                 HydrostaticFreeSurfaceModel,
                                 ShallowWaterModel} 

# Check for NaNs in the first prognostic field (generalizes to prescribed velocitries).
function default_nan_checker(model::OceananigansModels)
    model_fields = prognostic_fields(model)
    first_name = first(keys(model_fields))
    field_to_check_nans = NamedTuple{tuple(first_name)}(model_fields)
    nan_checker = NaNChecker(field_to_check_nans)
    return nan_checker
end

end # module

