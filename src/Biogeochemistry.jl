module Biogeochemistry
using Oceananigans.Grids: Center, xnode, ynode, znode
using Oceananigans.Forcings: maybe_constant_field, DiscreteForcing
using Oceananigans.Advection: div_Uc, UpwindBiasedFifthOrder
using Oceananigans.Operators: identity1

import Oceananigans.Fields: location, CenterField
import Oceananigans.Forcings: regularize_forcing
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

@inline function all_fields_present(fields::NamedTuple, required_fields, grid)
    field_names = keys(fields)
    field_values = values(fields)

    for field_name in required_fields
        if field_name not in field_names
            push!(field_names, field_name)
            push!(field_values, CenterField(grid))
        end
    end

    return NamedTuple{field_names}(field_values)
end

@inline all_fields_present(fields::Tuple, required_fields, grid) = (fields..., required_fields...)

"""Ensure that `tracers` contains biogeochemical tracers and `auxiliary_fields` contains biogeochemical auxiliary fields (e.g. PAR)."""
@inline function validate_biogeochemistry!(tracers, auxiliary_fields, bgc, grid)
    req_tracers = required_biogeochemical_tracers(bgc)
    tracers = all_fields_present(tracers, req_tracers, grid)

    req_auxiliary_fields = required_biogeochemical_auxiliary_fields(bgc)
    auxiliary_fields = all_fields_present(auxiliary_fields, req_auxiliary_fields, grid)
    
    return nothing
end

required_biogeochemical_tracers(::NoBiogeochemistry) = ()
required_biogeochemical_auxiliary_fields(bgc::AbstractBiogeochemistry) = ()

"""Sets up a tracer based biogeochemical model in a similar way to SeawaterBouyancy"""
struct BiogeochemicalModel <: AbstractBiogeochemistry
    biogeochemical_tracers::NTuple{N, Symbol} where N
    reactions::NamedTuple
    advection_scheme::NamedTuple
    sinking_velocities::NamedTuple
    auxiliary_fields::NTuple{M, Symbol} where M
end

function regularize_sinking_velocities(sinking_speeds)
    sinking_velocities = []
    for w in values(sinking_speeds)
        u, v, w = maybe_constant_field.((0.0, 0.0, - w))
        push!(sinking_velocities, (; u, v, w))
    end

    return NamedTuple{keys(sinking_speeds)}(sinking_velocities)
end

# we can't use the standard `ContinuousForcing` regularisation here because it requires all the tracers to be inplace to have the correct indices
struct ContinuousBiogeochemicalForcing
    func::Function
    parameters::NamedTuple
    field_dependencies::NTuple{N, Symbol} where N
end

DiscreteBiogeochemicalForcing = DiscreteForcing

ContinuousBiogeochemicalForcing(func; parameters=nothing, field_dependencies=()) = ContinuousBiogeochemicalForcing(func, parameters, field_dependencies)

function BiogeochemicalForcing(func; parameters=nothing, discrete_form=false, field_dependencies=())
    if discrete_form
        return DiscreteBiogeochemicalForcing(func, parameters)
    else
        return ContinuousBiogeochemicalForcing(func, parameters, field_dependencies)
    end
end

function regularize_biogeochemical_forcing(forcing::Function)
    return ContinuousBiogeochemicalForcing(forcing)
end

regularize_biogeochemical_forcing(forcing) = forcing

@inline getargs(fields, field_dependencies, i, j, k, grid, params::Nothing) = @inbounds identity1.(i, j, k, grid, fields[field_dependencies])
@inline getargs(fields, field_dependencies, i, j, k, grid, params) = @inbounds tuple(identity1.(i, j, k, grid, fields[field_dependencies])..., params)

@inline function (forcing::ContinuousBiogeochemicalForcing)(i, j, k, grid, clock, fields)
    args = getargs(fields, forcing.field_dependencies, i, j, k, grid, forcing.parameters)

    x = xnode(Center(), Center(), Center(), i, j, k, grid)
    y = ynode(Center(), Center(), Center(), i, j, k, grid)
    z = znode(Center(), Center(), Center(), i, j, k, grid)

    return forcing.func(x, y, z, clock.time, args...)
end

@inline function BiogeochemicalModel(tracers, reactions; advection_scheme=UpwindBiasedFifthOrder, sinking_velocities=NamedTuple(), auxiliary_fields=())
    reactions = NamedTuple{keys(reactions)}([regularize_biogeochemical_forcing(reaction) for reaction in values(reactions)]) 
    sinking_velocities = regularize_sinking_velocities(sinking_velocities)
    return BiogeochemicalModel(tracers, reactions, advection_scheme, sinking_velocities, auxiliary_fields)
end

@inline function (bgc::BiogeochemicalModel)(i, j, k, grid, val_tracer_name::Val{tracer_name}, clock, fields) where tracer_name
    # there is probably a cleaner way todo this with multiple dispathc
    reaction = bgc.reactions[tracer_name](i, j, k, grid, clock, fields)

    if tracer_name in keys(bgc.sinking_velocities)
        sinking = - div_Uc(i, j, k, grid, bgc.adv_scheme, bgc.sinking_velocities[tracer_name], fields[tracer_name])
        return reaction + sinking
    else
        return reaction
    end
end

required_biogeochemical_tracers(bgc::BiogeochemicalModel) = bgc.biogeochemical_tracers
required_biogeochemical_auxiliary_fields(bgc::BiogeochemicalModel) = bgc.auxiliary_fields

end # module
