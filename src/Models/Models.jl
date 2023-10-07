module Models

export
    NonhydrostaticModel,
    ShallowWaterModel, ConservativeFormulation, VectorInvariantFormulation,
    HydrostaticFreeSurfaceModel,
    ExplicitFreeSurface, ImplicitFreeSurface, SplitExplicitFreeSurface,
    PrescribedVelocityFields, PressureField,
    LagrangianParticles

using Oceananigans: AbstractModel, fields
using Oceananigans.Grids: AbstractGrid, halo_size, inflate_halo_size
using Oceananigans.TimeSteppers: AbstractTimeStepper, Clock
using Oceananigans.Utils: Time
using Oceananigans.Fields: AbstractField, flattened_unique_values
using Oceananigans.AbstractOperations: AbstractOperation

import Oceananigans: initialize!
import Oceananigans.Architectures: architecture

using Oceananigans.OutputReaders: update_field_time_series!, extract_field_timeseries

architecture(model::AbstractModel) = model.architecture
initialize!(model::AbstractModel) = nothing

total_velocities() = nothing
possible_field_time_series(model) = ()

import Oceananigans.TimeSteppers: reset!

# Update _all_ `FieldTimeSeries`es in an `AbstractModel`. 
# Loop over all propery names and extract any of them which is a `FieldTimeSeries`.
# Flatten the resulting tuple by extracting unique values and set! them to the 
# correct time range by looping over them
function update_model_field_time_series!(model::AbstractModel, clock::Clock)
    time = Time(clock.time)

    possible_field_time_series = possible_field_time_series(model)

    time_series_tuple = extract_field_timeseries(possible_field_time_series)
    time_series_tuple = flattened_unique_values(time_series_tuple)

    for fts in time_series_tuple
        update_field_time_series!(fts, time)
    end

    return nothing
end

function reset!(model::AbstractModel)

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

end # module
