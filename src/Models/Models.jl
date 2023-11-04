module Models

export
    NonhydrostaticModel,
    ShallowWaterModel, ConservativeFormulation, VectorInvariantFormulation,
    HydrostaticFreeSurfaceModel,
    ExplicitFreeSurface, ImplicitFreeSurface, SplitExplicitFreeSurface,
    PrescribedVelocityFields, PressureField,
    LagrangianParticles,
    seawater_density

using Oceananigans: AbstractModel, fields, prognostic_fields
using Oceananigans.Grids: AbstractGrid, halo_size, inflate_halo_size
using Oceananigans.TimeSteppers: AbstractTimeStepper, Clock
using Oceananigans.Utils: Time
using Oceananigans.Fields: AbstractField, Field, flattened_unique_values
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Advection: AbstractAdvectionScheme, CenteredSecondOrder, VectorInvariant

import Oceananigans: initialize!
import Oceananigans.Architectures: architecture
import Oceananigans.TimeSteppers: reset!

using Oceananigans.OutputReaders: update_field_time_series!, extract_field_timeseries

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
architecture(model::AbstractModel) = model.architecture
initialize!(model::AbstractModel) = nothing
total_velocities(model::AbstractModel) = nothing
timestepper(model::AbstractModel) = model.timestepper

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

const OceananigansModels = Union{HydrostaticFreeSurfaceModel, 
                                 NonhydrostaticModel, 
                                 ShallowWaterModel}

# Update _all_ `FieldTimeSeries`es in an `OceananigansModel`. 
# Extract `FieldTimeSeries` from all property names that might contain a `FieldTimeSeries`
# Flatten the resulting tuple by extracting unique values and set! them to the 
# correct time range by looping over them
function update_model_field_time_series!(model::OceananigansModels, clock::Clock)
    time = Time(clock.time)

    possible_fts = possible_field_time_series(model)

    time_series_tuple = extract_field_timeseries(possible_fts)
    time_series_tuple = flattened_unique_values(time_series_tuple)

    for fts in time_series_tuple
        update_field_time_series!(fts, time)
    end

    return nothing
end

"""
    possible_field_time_series(model::HydrostaticFreeSurfaceModel)

Return a `Tuple` containing properties of and `OceananigansModel` that could contain `FieldTimeSeries`.
"""
possible_field_time_series(model::OceananigansModels) = tuple(fields(model), model.forcing, model.diffusivity_fields)
                
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
    
    return nothing
end

# Check for NaNs in the first prognostic field (generalizes to prescribed velocities).
function default_nan_checker(model::OceananigansModels)
    model_fields = prognostic_fields(model)

    if isempty(model_fields) 
        @warn "The NaNChecker is disabled because this model does not evolve prognostic variables."
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

# General seawater density calculation applicable to both `NonhydrostaticModel` and  `HydrostaticFreeSurfaceModel`
include("seawater_density.jl")

end # module
