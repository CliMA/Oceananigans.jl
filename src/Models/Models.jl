module Models

export
    NonhydrostaticModel, BackgroundField, BackgroundFields,
    ShallowWaterModel, ConservativeFormulation, VectorInvariantFormulation,
    HydrostaticFreeSurfaceModel, ZStar, ZCoordinate,
    ExplicitFreeSurface, ImplicitFreeSurface, SplitExplicitFreeSurface,
    PrescribedVelocityFields, PressureField,
    LagrangianParticles,
    seawater_density

using Oceananigans: AbstractModel, fields, prognostic_fields
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Advection: AbstractAdvectionScheme, Centered, VectorInvariant
using Oceananigans.Fields: AbstractField, Field, flattened_unique_values, boundary_conditions
using Oceananigans.Grids: AbstractGrid, halo_size, inflate_halo_size
using Oceananigans.OutputReaders: update_field_time_series!, extract_field_time_series
using Oceananigans.TimeSteppers: AbstractTimeStepper, Clock, update_state!
using Oceananigans.Utils: Time

import Oceananigans: initialize!
import Oceananigans.Architectures: architecture
import Oceananigans.TimeSteppers: reset!
import Oceananigans.Solvers: iteration

# A prototype interface for AbstractModel.
#
# TODO: decide if we like this.
#
# We assume that model has some properties, eg:
#   - model.clock::Clock
#   - model.architecture.
#   - model.timestepper with timestepper.G⁻ and timestepper.Gⁿ :spiral_eyes:

iteration(model::AbstractModel) = model.clock.iteration
Base.time(model::AbstractModel) = model.clock.time
Base.eltype(model::AbstractModel) = eltype(model.grid)
architecture(model::AbstractModel) = model.grid.architecture
initialize!(model::AbstractModel) = nothing
total_velocities(model::AbstractModel) = nothing
timestepper(model::AbstractModel) = model.timestepper
initialization_update_state!(model::AbstractModel; kw...) = update_state!(model; kw...) # fallback

# Fallback for any abstract model that does not contain `FieldTimeSeries`es
update_model_field_time_series!(model::AbstractModel, clock::Clock) = nothing

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
        throw(ArgumentError("The grid halo $user_halo must be at least equal to $required_halo. \n \
                            Note that an ImmersedBoundaryGrid requires an extra halo point in all \n \
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
validate_tracer_advection(tracer_advection_tuple::NamedTuple, grid) = Centered(), tracer_advection_tuple
validate_tracer_advection(tracer_advection::AbstractAdvectionScheme, grid) = tracer_advection, NamedTuple()
validate_tracer_advection(tracer_advection::Nothing, grid) = nothing, NamedTuple()

# Communication - Computation overlap in distributed models
include("interleave_communication_and_computation.jl")

#####
##### All the code
#####

include("NonhydrostaticModels/NonhydrostaticModels.jl")
include("HydrostaticFreeSurfaceModels/HydrostaticFreeSurfaceModels.jl")
include("ShallowWaterModels/ShallowWaterModels.jl")
include("LagrangianParticleTracking/LagrangianParticleTracking.jl")

using .NonhydrostaticModels: NonhydrostaticModel, PressureField, BackgroundField, BackgroundFields

using .HydrostaticFreeSurfaceModels:
    HydrostaticFreeSurfaceModel,
    ExplicitFreeSurface, ImplicitFreeSurface, SplitExplicitFreeSurface,
    PrescribedVelocityFields, ZStar, ZCoordinate

using .ShallowWaterModels: ShallowWaterModel, ConservativeFormulation, VectorInvariantFormulation

using .LagrangianParticleTracking: LagrangianParticles

const OceananigansModels = Union{HydrostaticFreeSurfaceModel,
                                 NonhydrostaticModel,
                                 ShallowWaterModel}

"""
    possible_field_time_series(model::HydrostaticFreeSurfaceModel)

Return a `Tuple` containing properties of and `OceananigansModel` that could contain `FieldTimeSeries`.
"""
function possible_field_time_series(model::OceananigansModels)
    forcing = model.forcing
    model_fields = fields(model)
    # Note: we may need to include other objects in the tuple below,
    # such as model.diffusivity_fields
    return tuple(model_fields, forcing)
end

# Update _all_ `FieldTimeSeries`es in an `OceananigansModel`.
# Extract `FieldTimeSeries` from all property names that might contain a `FieldTimeSeries`
# Flatten the resulting tuple by extracting unique values and set! them to the
# correct time range by looping over them
function update_model_field_time_series!(model::OceananigansModels, clock::Clock)
    time = Time(clock.time)

    possible_fts = possible_field_time_series(model)
    time_series_tuple = extract_field_time_series(possible_fts)
    time_series_tuple = flattened_unique_values(time_series_tuple)

    for fts in time_series_tuple
        update_field_time_series!(fts, time)
    end

    return nothing
end

import Oceananigans.TimeSteppers: reset!

function reset!(model::OceananigansModels)

    for field in fields(model)
        fill!(field, 0)
    end

    for field in model.timestepper.G⁻
        fill!(field, 0)
    end

    for field in model.timestepper.Gⁿ
        fill!(field, 0)
    end

    reset!(timestepper(model))

    return nothing
end

using Oceananigans.Diagnostics: NaNChecker
import Oceananigans.Diagnostics: default_nan_checker

# Check for NaNs in the first prognostic field (generalizes to prescribed velocities).
function default_nan_checker(model::OceananigansModels)
    model_fields = prognostic_fields(model)

    if isempty(model_fields)
        return nothing
    end

    first_name = first(keys(model_fields))
    field_to_check_nans = NamedTuple{tuple(first_name)}(model_fields)
    nan_checker = NaNChecker(field_to_check_nans)
    return nan_checker
end

using Oceananigans.Models.HydrostaticFreeSurfaceModels: OnlyParticleTrackingModel

# Particle tracking models with prescribed velocities (and no tracers)
# have no prognostic fields and no chance to producing a NaN.
default_nan_checker(::OnlyParticleTrackingModel) = nothing

# Extend output writer functionality to Ocenanaigans' models
import Oceananigans.OutputWriters: default_included_properties, checkpointer_address

default_included_properties(::NonhydrostaticModel) = [:grid, :coriolis, :buoyancy, :closure]
default_included_properties(::ShallowWaterModel) = [:grid, :coriolis, :closure]
default_included_properties(::HydrostaticFreeSurfaceModel) = [:grid, :coriolis, :buoyancy, :closure]

checkpointer_address(::ShallowWaterModel) = "ShallowWaterModel"
checkpointer_address(::NonhydrostaticModel) = "NonhydrostaticModel"
checkpointer_address(::HydrostaticFreeSurfaceModel) = "HydrostaticFreeSurfaceModel"

# Implementation of a `seawater_density` `KernelFunctionOperation
# applicable to both `NonhydrostaticModel` and  `HydrostaticFreeSurfaceModel`
include("seawater_density.jl")
include("boundary_mean.jl")
include("boundary_condition_operation.jl")

end # module
