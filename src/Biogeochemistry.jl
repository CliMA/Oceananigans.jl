module Biogeochemistry
using Oceananigans.Grids: Center
using Oceananigans.Forcings: model_forcing, maybe_constant_field
using Oceananigans.Advection: div_Uc, UpwindBiasedFifthOrder

import Oceananigans.Fields: location
#####
##### Generic fallbacks for biogeochemistry
#####

nothing_function(args...) = nothing

"""
Update tracer tendencies.

Called at the end of calculate_tendencies!
"""
update_tendencies!(bgc, model) = nothing

"""
Update tracer tendencies.

Called at the end of calculate_tendencies!
"""
update_biogeochemical_state!(bgc, model) = nothing

"""Return the biogeochemical forcing for `val_tracer_name` when model is called."""
abstract type AbstractBiogeochemistry end

@inline (::AbstractBiogeochemistry)(i, j, k, grid, val_tracer_name, clock, fields) = zero(grid)

struct NoBiogeochemistry <: AbstractBiogeochemistry end

tracernames(tracers) = keys(tracers)
tracernames(tracers::Tuple) = tracers

"""Ensure that `tracers` contains biogeochemical tracers and `auxiliary_fields` contains biogeochemical auxiliary fields (e.g. PAR)."""
@inline function validate_biogeochemistry!(tracers, auxiliary_fields, bgc)
    req_tracers = required_biogeochemical_tracers(bgc)
    
    all(tracer ∈ tracernames(tracers) for tracer in req_tracers) ||
        error("$(req_tracers) must be among the list of tracers to use $(typeof(bgc).name.wrapper)")

    req_auxiliary_fields = required_biogeochemical_auxiliary_fields(bgc)
    
    all(field ∈ tracernames(auxiliary_fields) for field in req_auxiliary_fields) ||
        error("$(req_auxiliary_fields) must be among the list of auxiliary fields to use $(typeof(bgc).name.wrapper)")
    
    return nothing
end

required_biogeochemical_tracers(::NoBiogeochemistry) = ()
required_biogeochemical_auxiliary_fields(::NoBiogeochemistry) = ()

"""Sets up a tracer based biogeochemical model in a similar way to SeawaterBouyancy"""
struct TracerBasedBiogeochemistry <: AbstractBiogeochemistry
    biogeochemical_tracers
    reactions
    advection_scheme
    sinking_velocities
    auxiliary_fields
end

function regularize_sinking_velocities(sinking_speeds)
    sinking_velocities = []
    for w in values(sinking_speeds)
        u, v, w = maybe_constant_field.((0.0, 0.0, - w))
        push!(sinking_velocities, (; u, v, w))
    end

    return NamedTuple{keys(sinking_speeds)}(sinking_velocities)
end

# `model_forcing` expects to be passed a NamedTuple of fields and checks the location for evaluating continuous forcing functions so it can force velocities
# here we will only ever be passing tracer fields so they will always be center fields
struct FillerTracerField end
location(::FillerTracerField) =  (Center, Center, Center)

@inline function TracerBasedBiogeochemistry(tracers, reactions; advection_scheme=UpwindBiasedFifthOrder, sinking_velocities=NamedTuple(), auxiliary_fields=())
    reactions = model_forcing(NamedTuple{tracers}(repeat([FillerTracerField()], length(tracers))); reactions...)
    sinking_velocities = regularize_sinking_velocities(sinking_velocities)
    return TracerBasedBiogeochemistry(tracers, reactions, advection_scheme, sinking_velocities, auxiliary_fields)
end

@inline function (bgc::TracerBasedBiogeochemistry)(i, j, k, grid, val_tracer_name::Val{tracer_name}, clock, fields) where tracer_name
    # there is probably a cleaner way todo this with multiple dispathc
    reaction = bgc.reactions[tracer_name](i, j, k, grid, clock, fields)

    if tracer_name in keys(bgc.sinking_velocities)
        sinking = - div_Uc(i, j, k, grid, bgc.adv_scheme, bgc.sinking_velocities[tracer_name], fields[tracer_name])
        return reaction + sinking
    else
        return reaction
    end
end

required_biogeochemical_tracers(bgc::TracerBasedBiogeochemistry) = bgc.biogeochemical_tracers
required_biogeochemical_auxiliary_fields(bgc::TracerBasedBiogeochemistry) = bgc.auxiliary_fields

end # module
