module Biogeochemistry

using Oceananigans.Grids: Center, xnode, ynode, znode
using Oceananigans.Advection: div_Uc, CenteredSecondOrder
using Oceananigans.Utils: tupleit
using Oceananigans.Fields: Field
using Oceananigans.Architectures: device, architecture
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel

import Oceananigans.Fields: location, CenterField
import Oceananigans.Forcings: regularize_forcing, maybe_constant_field

#####
##### Generic fallbacks for biogeochemistry
#####

"""
Update tendencies.

Called at the end of calculate_tendencies!
"""
update_tendencies!(bgc, model) = nothing

"""
Update tracer tendencies.

Called at the end of calculate_tendencies!
"""
update_biogeochemical_state!(bgc, model) = nothing

@inline biogeochemical_drift_velocity(bgc, val_tracer_name) = nothing
@inline biogeochemical_advection_scheme(bgc, val_tracer_name) = nothing
@inline biogeochemical_auxiliary_fields(bgc) = NamedTuple()

#####
##### Default (discrete form) biogeochemical source
#####

abstract type AbstractBiogeochemistry end

@inline function biogeochemistry_rhs(i, j, k, grid, bgc, val_tracer_name::Val{tracer_name}, clock, fields) where tracer_name
    U_drift = biogeochemical_drift_velocity(bgc, val_tracer_name)
    scheme = biogeochemical_advection_scheme(bgc, val_tracer_name)
    src = biogeochemical_transition(i, j, k, grid, bgc, val_tracer_name, clock, fields)
    c = @inbounds fields[tracer_name]
        
    return src - div_Uc(i, j, k, grid, scheme, U_drift, c)
end

@inline biogeochemical_transition(i, j, k, grid, bgc, val_tracer_name, clock, fields) =
    bgc(i, j, k, grid, val_tracer_name, clock, fields)

@inline (bgc::AbstractBiogeochemistry)(i, j, k, grid, val_tracer_name, clock, fields) = zero(grid)

#####
##### Continuous form biogeochemical source
#####
 
"""Return the biogeochemical forcing for `val_tracer_name` when model is called."""
abstract type AbstractContinuousFormBiogeochemistry <: AbstractBiogeochemistry end

@inline extract_biogeochemical_fields(i, j, k, grid, fields, names::NTuple{1}) =
    @inbounds (fields[names[1]][i, j, k],)

@inline extract_biogeochemical_fields(i, j, k, grid, fields, names::NTuple{2}) =
    @inbounds (fields[names[1]][i, j, k],
               fields[names[2]][i, j, k])

@inline extract_biogeochemical_fields(i, j, k, grid, fields, names::NTuple{N}) where N =
    @inbounds ntuple(n -> fields[names[n]][i, j, k], Val(N))

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

struct NoBiogeochemistry <: AbstractBiogeochemistry end

tracernames(tracers) = keys(tracers)
tracernames(tracers::Tuple) = tracers

add_biogeochemical_tracer(tracers::Tuple, name, grid) = tuple(tracers..., name)
add_biogeochemical_tracer(tracers::NamedTuple, name, grid) = merge(tracers, NamedTuple(name => CenterField(grid)))

@inline function has_biogeochemical_tracers(fields, required_fields, grid)
    user_specified_tracers = [name in tracernames(fields) for name in required_fields]

    if !all(user_specified_tracers) && any(user_specified_tracers)
        throw(ArgumentError("The biogeochemical model you have selected requires $required_fields. 
    You have specified some but not all of these as tracers so may be attempting to use them for a different purpose.
    Please either specify all of the required fields, or none and allow them to be automatically added."))
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
contains biogeochemical auxiliary fields (e.g. PAR).
"""
@inline function validate_biogeochemistry(tracers, auxiliary_fields, bgc, grid, clock)
    req_tracers = required_biogeochemical_tracers(bgc)
    tracers = has_biogeochemical_tracers(tracers, req_tracers, grid)
    req_auxiliary_fields = required_biogeochemical_auxiliary_fields(bgc)

    all(field ∈ tracernames(auxiliary_fields) for field in req_auxiliary_fields) ||
        error("$(req_auxiliary_fields) must be among the list of auxiliary fields to use $(typeof(bgc).name.wrapper)")

    # Return tracers and aux fields so that users may overload and
    # define their own special auxiliary fields (e.g. PAR in test)
    return tracers, auxiliary_fields 
end

required_biogeochemical_tracers(::NoBiogeochemistry) = ()
required_biogeochemical_auxiliary_fields(bgc::AbstractBiogeochemistry) = ()

end # module
