module Models

export
    NonhydrostaticModel,
    ShallowWaterModel, ConservativeFormulation, VectorInvariantFormulation,
    HydrostaticFreeSurfaceModel,
    ExplicitFreeSurface, ImplicitFreeSurface, SplitExplicitFreeSurface,
    PrescribedVelocityFields, PressureField,
    LagrangianParticles

using Oceananigans: AbstractModel
using Oceananigans.Grids: halo_size, inflate_halo_size
using Oceananigans: fields
using Oceananigans.TimeSteppers: AbstractTimeStepper
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Fields: AbstractField, flattened_unique_values
using Oceananigans.AbstractOperations: AbstractOperation

import Oceananigans: initialize!
import Oceananigans.Architectures: architecture

architecture(model::AbstractModel) = model.architecture
initialize!(model::AbstractModel) = nothing

total_velocities() = nothing

import Oceananigans.TimeSteppers: reset!

# Update _all_ `FieldTimeSeries`es in an `AbstractModel`. 
# Loop over all propery names and extract any of them which is a `FieldTimeSeries`.
# Flatten the resulting tuple by extracting unique values and set! them to the 
# correct time range by looping over them
function update_time_series!(model::AbstractModel, clock::Clock)

    time = Time(clock.time)
    time_series_tuple = extract_field_timeseries(model)
    time_series_tuple = flattened_unique_values(time_series_tuple)

    for fts in time_series_tuple
        update_time_series!(fts, time)
    end

    return nothing
end

update_time_series!(::Nothing, time) = nothing

# Recursion for all properties 
function extract_field_timeseries(t) 
    prop = propertynames(t)
    if isempty(prop)
        return nothing
    end

    return Tuple(extract_field_timeseries(getproperty(t, p)) for p in prop)
end

# For types that do not contain `FieldTimeSeries`, halt the recursion
NonFTS = [:Number, :AbstractArray, :AbstractTimeStepper, :AbstractGrid]

for NonFTSType in NonFTS
    @eval extract_field_timeseries(::$NonFTSType) = nothing
end

# Special recursion rules for `Tuple` and `Field` types
extract_field_timeseries(t::AbstractField)     = Tuple(extract_field_timeseries(getproperty(t, p)) for p in propertynames(t))
extract_field_timeseries(t::AbstractOperation) = Tuple(extract_field_timeseries(getproperty(t, p)) for p in propertynames(t))
extract_field_timeseries(t::Tuple)             = Tuple(extract_field_timeseries(n) for n in t)
extract_field_timeseries(t::NamedTuple)        = Tuple(extract_field_timeseries(n) for n in t)

# Termination
extract_field_timeseries(f::FieldTimeSeries)   = f

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
