module Biogeochemistry

using Oceananigans.Grids: Center, xnode, ynode, znode
using Oceananigans.Advection: div_Uc, CenteredSecondOrder
using Oceananigans.Architectures: device, architecture

import Oceananigans.Fields: CenterField

#####
##### Generic fallbacks for biogeochemistry
#####

"""
    update_tendencies!(bgc, model)

Called at the end of calculate_tendencies! 
"""
update_tendencies!(bgc, model) = nothing

"""
    update_biogeochemical_state!(bgc, model)

Called at the end of update_state!
"""
update_biogeochemical_state!(bgc, model) = nothing

@inline biogeochemical_drift_velocity(bgc, val_tracer_name) = nothing
@inline biogeochemical_advection_scheme(bgc, val_tracer_name) = nothing
@inline biogeochemical_auxiliary_fields(bgc) = NamedTuple()

"""
    AbstractBiogeochemistry

Abstract type for biogeochemical models. To define a biogeochemcial relaionship
the following functions must have methods defined where `BiogeochemicalModel`
is a subtype of `AbstractBioeochemistry`:

    - `(bgc::BiogeochemicalModel)(i, j, k, grid, ::Val{:tracer_name}, clock, fields)` which 
       returns the biogeochemical reaction for for each tracer.

    - `required_biogeochemical_tracers(::BiogeochemicalModel)` which returns a tuple of
       required `tracer_names`.

    - `required_biogeochemical_auxiliary_fields(::BiogeochemicalModel)` which returns 
       a tuple of required auxiliary fields.

    - `biogeochemical_auxiliary_fields(bgc::BiogeochemicalModel)` which returns a `NamedTuple`
       of the models auxiliary fields.

    - `biogeochemical_drift_velocity(bgc::BiogeochemicalModel, ::Val{:tracer_name})` which 
       returns a velocity fields (i.e. a `NamedTuple` of fields with keys `u`, `v` & `w`)
       for each tracer.

    - `biogeochemical_advection_scheme(bgc::BiogeochemicalModel, ::Val{:tracer_name})` which
       returns an advection scheme for each tracer.

    - `update_biogeochemical_state!(bgc::BiogeochemicalModel, model)` (optional) to update the
        model state.
"""
abstract type AbstractBiogeochemistry end

@inline biogeochemistry_rhs(i, j, k, grid, ::Nothing, val_tracer_name, clock, fields) = zero(grid)

@inline function biogeochemistry_rhs(i, j, k, grid, bgc, val_tracer_name::Val{tracer_name}, clock, fields) where tracer_name
    U_drift = biogeochemical_drift_velocity(bgc, val_tracer_name)
    scheme = biogeochemical_advection_scheme(bgc, val_tracer_name)

    # gets the biogeochemical reaction forcing (including transforming form for continuous form)
    src = biogeochemical_transition(i, j, k, grid, bgc, val_tracer_name, clock, fields)
    
    c = @inbounds fields[tracer_name]
        
    return src - div_Uc(i, j, k, grid, scheme, U_drift, c)
end

# Returns the forcing for discrete form models
@inline biogeochemical_transition(i, j, k, grid, bgc, val_tracer_name, clock, fields) =
    bgc(i, j, k, grid, val_tracer_name, clock, fields)


# Required for when a model is defined but not for all tracers
@inline (bgc::AbstractBiogeochemistry)(i, j, k, grid, val_tracer_name, clock, fields) = zero(grid)

"""
    AbstractContinuousFormBiogeochemistry

Abstract type for biogeochemical models with continuous form biogeochemical reaction 
functions. To define a biogeochemcial relaionship the following functions must have methods 
defined where `BiogeochemicalModel` is a subtype of `AbstractContinuousFormBiogeochemistry`:

 - `(bgc::BiogeochemicalModel)(::Val{:tracer_name}, x, y, z, t, tracers..., auxiliary_fields...)` 
    which returns the biogeochemical reaction for for each tracer.

 - `required_biogeochemical_tracers(::BiogeochemicalModel)` which returns a tuple of
    required tracer names.

 - `required_biogeochemical_auxiliary_fields(::BiogeochemicalModel)` which returns 
    a tuple of required auxiliary fields.

 - `biogeochemical_auxiliary_fields(bgc::BiogeochemicalModel)` which returns a `NamedTuple`
    of the models auxiliary fields

 - `biogeochemical_drift_velocity(bgc::BiogeochemicalModel, ::Val{:tracer_name})` which 
    returns "additional" velocity fields modeling, for example, sinking particles

 - `biogeochemical_advection_scheme(bgc::BiogeochemicalModel, ::Val{:tracer_name})` which
    returns an advection scheme for each tracer.

 - `update_biogeochemical_state!(bgc::BiogeochemicalModel, model)` (optional) to update the
    model state
"""
abstract type AbstractContinuousFormBiogeochemistry <: AbstractBiogeochemistry end

@inline extract_biogeochemical_fields(i, j, k, grid, fields, names::NTuple{1}) =
    @inbounds tuple(fields[names[1]][i, j, k])

@inline extract_biogeochemical_fields(i, j, k, grid, fields, names::NTuple{2}) =
    @inbounds (fields[names[1]][i, j, k],
               fields[names[2]][i, j, k])

@inline extract_biogeochemical_fields(i, j, k, grid, fields, names::NTuple{N}) where N =
    @inbounds ntuple(n -> fields[names[n]][i, j, k], Val(N))

"""Return the biogeochemical forcing for `val_tracer_name` for continuous form when model is called."""
@inline function biogeochemical_transition(i, j, k, grid, bgc::AbstractContinuousFormBiogeochemistry,
                                           val_tracer_name, clock, fields)

    names_to_extract = tuple(required_biogeochemical_tracers(bgc)...,
                             required_biogeochemical_auxiliary_fields(bgc)...)

    fields_ijk = extract_biogeochemical_fields(i, j, k, grid, fields, names_to_extract)

    x = xnode(Center(), Center(), Center(), i, j, k, grid)
    y = ynode(Center(), Center(), Center(), i, j, k, grid)
    z = znode(Center(), Center(), Center(), i, j, k, grid)

    return bgc(val_tracer_name, x, y, z, clock.time, fields_ijk...)
end

@inline (bgc::AbstractContinuousFormBiogeochemistry)(val_tracer_name, x, y, z, t, fields...) = zero(x)

tracernames(tracers) = keys(tracers)
tracernames(tracers::Tuple) = tracers

add_biogeochemical_tracer(tracers::Tuple, name, grid) = tuple(tracers..., name)
add_biogeochemical_tracer(tracers::NamedTuple, name, grid) = merge(tracers, NamedTuple(name => CenterField(grid)))

@inline function has_biogeochemical_tracers(fields, required_fields, grid)
    user_specified_tracers = [name in tracernames(fields) for name in required_fields]

    if !all(user_specified_tracers) && any(user_specified_tracers)

        throw(ArgumentError("The biogeochemical model you have selected requires $required_fields.\n" *
                            "You have specified some but not all of these as tracers so may be attempting\n" *
                            "to use them for a different purpose. Please either specify all of the required\n" *
                            "fields, or none and allow them to be automatically added."))
                            
    elseif !any(user_specified_tracers)
        for field_name in required_fields
            fields = add_biogeochemical_tracer(fields, field_name, grid)
        end
    else
        fields = fields
    end

    return fields
end

"""
    validate_biogeochemistry(tracers, auxiliary_fields, bgc, grid, clock)

Ensure that `tracers` contains biogeochemical tracers and `auxiliary_fields`
contains biogeochemical auxiliary fields.
"""
@inline function validate_biogeochemistry(tracers, auxiliary_fields, bgc, grid, clock)
    req_tracers = required_biogeochemical_tracers(bgc)
    tracers = has_biogeochemical_tracers(tracers, req_tracers, grid)
    req_auxiliary_fields = required_biogeochemical_auxiliary_fields(bgc)

    all(field ∈ tracernames(auxiliary_fields) for field in req_auxiliary_fields) ||
        error("$(req_auxiliary_fields) must be among the list of auxiliary fields to use $(typeof(bgc).name.wrapper)")

    # Return tracers and aux fields so that users may overload and
    # define their own special auxiliary fields
    return tracers, auxiliary_fields 
end

const AbstractBGCOrNothing = Union{Nothing, AbstractBiogeochemistry}

required_biogeochemical_tracers(::AbstractBGCOrNothing) = ()
required_biogeochemical_auxiliary_fields(::AbstractBGCOrNothing) = ()

end # module
