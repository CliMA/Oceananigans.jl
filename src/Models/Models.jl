module Models

export
    NonhydrostaticModel, BackgroundField, BackgroundFields,
    ShallowWaterModel, ConservativeFormulation, VectorInvariantFormulation,
    HydrostaticFreeSurfaceModel, ZStarCoordinate, ZCoordinate,
    ExplicitFreeSurface, ImplicitFreeSurface, SplitExplicitFreeSurface,
    PrescribedVelocityFields, PressureField,
    LagrangianParticles, DroguedParticleDynamics,
    BoundaryConditionOperation, ForcingOperation,
    seawater_density,
    BulkDrag, BulkDragFunction, BulkDragBoundaryCondition,
    XDirectionBulkDragFunction, YDirectionBulkDragFunction, ZDirectionBulkDragFunction,
    LinearFormulation, QuadraticFormulation

using Oceananigans: AbstractModel, fields, prognostic_fields
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Advection: AbstractAdvectionScheme, Centered
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: ConstantField, Field, OneField, ZeroField, flattened_unique_values, quadfolded_companion_field
using Oceananigans.Forcings: AdvectiveForcing, MultipleForcings
using Oceananigans.Biogeochemistry: biogeochemical_drift_velocity
using Oceananigans.TurbulenceClosures: closure_auxiliary_velocity
using Oceananigans.Grids: Face, Center, halo_size, inflate_halo_size
using Oceananigans.OutputReaders: update_field_time_series!, extract_field_time_series
using Oceananigans.TimeSteppers: Clock
using Oceananigans.Units: Time

import Oceananigans: initialize!
import Oceananigans.Architectures: architecture
import Oceananigans.Grids: grid
import Oceananigans.Fields: set!
import Oceananigans.Solvers: iteration
import Oceananigans.OutputWriters: default_included_properties
import Oceananigans.TimeSteppers: reset!

# A prototype interface for AbstractModel.
#
# We assume that model has some properties:
#   - model.clock::Clock
#
# Models with grids should override `architecture` and `eltype`.

iteration(model::AbstractModel) = model.clock.iteration
Base.time(model::AbstractModel) = model.clock.time
Base.eltype(model::AbstractModel) = Float64
grid(model::AbstractModel) = model.grid
architecture(model::AbstractModel) = nothing
initialize!(model::AbstractModel) = nothing
total_velocities(model::AbstractModel) = nothing

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

refresh_vertical_advective_velocity_halos!(::Union{ZeroField, OneField, ConstantField}) = nothing

function refresh_vertical_advective_velocity_halos!(w)
    fill_halo_regions!(w)
    return nothing
end

function refresh_horizontal_advective_velocity_halos!(u, v)
    fill_halo_regions!(u)
    fill_halo_regions!(v)
    return nothing
end

refresh_horizontal_advective_velocity_halos!(::Union{ZeroField, OneField, ConstantField},
                                             ::Union{ZeroField, OneField, ConstantField}) = nothing

refresh_horizontal_advective_velocity_halos!(u::Field{Face, Center, LZ},
                                             v::Field{Center, Face, LZ}) where LZ =
    fill_halo_regions!((u, v))

function refresh_horizontal_advective_velocity_halos!(u::Field{Face, Center, LZ},
                                                      v::Union{ZeroField, OneField, ConstantField}) where LZ
    companion_v = quadfolded_companion_field(u)
    set!(companion_v, v)
    fill_halo_regions!((u, companion_v))
    return nothing
end

function refresh_horizontal_advective_velocity_halos!(u::Union{ZeroField, OneField, ConstantField},
                                                      v::Field{Center, Face, LZ}) where LZ
    companion_u = quadfolded_companion_field(v)
    set!(companion_u, u)
    fill_halo_regions!((companion_u, v))
    return nothing
end

refresh_tracer_auxiliary_velocity_halos!(::Nothing) = nothing

function refresh_tracer_auxiliary_velocity_halos!(velocities)
    refresh_horizontal_advective_velocity_halos!(velocities.u, velocities.v)
    refresh_vertical_advective_velocity_halos!(velocities.w)
    return nothing
end

refresh_tracer_advective_forcing_halos!(forcing) = nothing

refresh_tracer_advective_forcing_halos!(forcing::Tuple) =
    foreach(refresh_tracer_advective_forcing_halos!, forcing)

refresh_tracer_advective_forcing_halos!(forcing::MultipleForcings) =
    refresh_tracer_advective_forcing_halos!(forcing.forcings)

function refresh_tracer_advective_forcing_halos!(forcing::AdvectiveForcing)
    refresh_horizontal_advective_velocity_halos!(forcing.u, forcing.v)
    refresh_vertical_advective_velocity_halos!(forcing.w)
    return nothing
end

function refresh_all_tracer_auxiliary_halos!(model)
    for tracer_name in keys(model.tracers)
        tracer_name_val = Val(tracer_name)
        @inbounds forcing = model.forcing[tracer_name]

        biogeochemical_velocities = biogeochemical_drift_velocity(model.biogeochemistry, tracer_name_val)
        closure_velocities = closure_auxiliary_velocity(model.closure, model.closure_fields, tracer_name_val)

        refresh_tracer_auxiliary_velocity_halos!(biogeochemical_velocities)
        refresh_tracer_auxiliary_velocity_halos!(closure_velocities)
        refresh_tracer_advective_forcing_halos!(forcing)
    end

    return nothing
end

# Used in both NonhydrostaticModels and HydrostaticFreeSurfaceModels
function materialize_free_surface end

# Forward-declared so submodules (ShallowWaterModels, etc.) can `import` and add methods
# before the defining files are include()d further down.
function ForcingOperation end
function boundary_condition_args end

# Communication - Computation overlap in distributed models
include("interleave_communication_and_computation.jl")

# Shared open-boundary transport building blocks: per-boundary integrals,
# initialization, and correction helpers. NonhydrostaticModel composes these
# into `enforce_net_zero_transport!` to enforce the incompressible-pressure
# solvability condition; external anelastic models can compose their own
# variant for density-weighted momentum (ρu, ρv, ρw).
include("boundary_transport.jl")

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
    PrescribedVelocityFields, ZStarCoordinate, ZCoordinate,
    contravariant_velocities, kinetic_energy, relative_vorticity, vertical_vorticity

using .ShallowWaterModels: ShallowWaterModel, ConservativeFormulation, VectorInvariantFormulation
using .LagrangianParticleTracking: LagrangianParticles, DroguedParticleDynamics

# BulkDrag for quadratic drag boundary conditions
include("BulkDragBoundaryConditions.jl")
using .BulkDragBoundaryConditions: BulkDrag, BulkDragFunction, BulkDragBoundaryCondition,
                                  XDirectionBulkDragFunction, YDirectionBulkDragFunction, ZDirectionBulkDragFunction,
                                  LinearFormulation, QuadraticFormulation

const OceananigansModels = Union{HydrostaticFreeSurfaceModel,
                                 NonhydrostaticModel,
                                 ShallowWaterModel}

# OceananigansModels have grids, so we can use grid-based implementations
Base.eltype(model::OceananigansModels) = eltype(model.grid)
architecture(model::OceananigansModels) = model.grid.architecture

set!(model::OceananigansModels, new_clock::Clock) = set!(model.clock, new_clock)

"""
    possible_field_time_series(model::OceananigansModels)

Return a `Tuple` containing properties of and `OceananigansModel` that could contain `FieldTimeSeries`.
"""
function possible_field_time_series(model::OceananigansModels)
    forcing = model.forcing
    model_fields = fields(model)
    # Note: we may need to include other objects in the tuple below,
    # such as model.closure_fields
    return tuple(model_fields, forcing)
end

function possible_field_time_series(model::NonhydrostaticModel)
    forcing = model.forcing
    model_fields = fields(model)
    background_velocities = model.background_fields.velocities
    background_tracers = model.background_fields.tracers
    return tuple(model_fields, forcing, background_velocities, background_tracers)
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

    refresh_restored_model_state!(model)
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

# Extend output writer functionality to custom Oceananigans.Models
import Oceananigans.OutputWriters: default_included_properties,
                                   checkpointer_address

default_included_properties(::NonhydrostaticModel) = [:coriolis, :buoyancy, :closure]
default_included_properties(::HydrostaticFreeSurfaceModel) = [:coriolis, :buoyancy, :closure]
default_included_properties(::ShallowWaterModel) = [:coriolis, :closure]

checkpointer_address(::ShallowWaterModel) = "ShallowWaterModel"
checkpointer_address(::NonhydrostaticModel) = "NonhydrostaticModel"
checkpointer_address(::HydrostaticFreeSurfaceModel) = "HydrostaticFreeSurfaceModel"

default_included_properties(::OceananigansModels) = Symbol[]

refresh_restored_model_state!(model) = nothing

function refresh_restored_model_state!(model::HydrostaticFreeSurfaceModel)
    Oceananigans.TimeSteppers.reconcile_state!(model)
    Oceananigans.TurbulenceClosures.initialize_closure_fields!(model.closure_fields, model.closure, model)
    return nothing
end

function refresh_restored_model_state!(model::NonhydrostaticModel)
    NonhydrostaticModels.refresh_restored_nonhydrostatic_model_state!(model)
    Oceananigans.TurbulenceClosures.initialize_closure_fields!(model.closure_fields, model.closure, model)
    return nothing
end

function refresh_restored_model_state!(model::ShallowWaterModel)
    Oceananigans.TimeSteppers.update_state!(model)
    Oceananigans.TurbulenceClosures.initialize_closure_fields!(model.closure_fields, model.closure, model)
    return nothing
end

# Specialized output attributes for velocity and tracer fields
include("output_attributes.jl")

# Implementation of diagnostics applicable to both `NonhydrostaticModel` and `HydrostaticFreeSurfaceModel`
include("seawater_density.jl")
include("buoyancy_operation.jl")
include("boundary_mean.jl")
include("boundary_condition_operation.jl")
include("forcing_operation.jl")
include("set_model.jl")

# Implementation of the diagnostic for computing the dissipation rate
include("VarianceDissipationComputations/VarianceDissipationComputations.jl")

using .VarianceDissipationComputations

end # module
