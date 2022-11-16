module Biogeochemistry

using Oceananigans.Grids: Center, xnode, ynode, znode
using Oceananigans.Advection: div_Uc
using Oceananigans.Utils: tupleit

import Oceananigans.Fields: location, CenterField
import Oceananigans.Forcings: regularize_forcing

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

#####
##### Default (discrete form) biogeochemical source
#####

abstract type AbstractBiogeochemistry end

@inline function biogeochemistry_rhs(i, j, k, grid, bgc, val_tracer_name::Val{tracer_name}, clock, fields) where tracer_name
    U_drift = biogeochemical_drift_velocity(bgc, val_tracer_name)
    scheme = biogeochemical_advection_scheme(bgc, val_tracer_name)
    src = biogeochemical_transition(i, j, k, grid, bgc, val_tracer_name, clock, fields)
    c = fields[tracer_name]
        
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

@inline function all_fields_present(fields::NamedTuple, required_fields, grid)

    for field_name in required_fields
        if !(field_name in keys(fields))
            fields = merge(fields, NamedTuple{(field_name, )}((CenterField(grid), )))
        end
    end

    return fields
end

@inline all_fields_present(fields::Tuple, required_fields, grid) = (fields..., required_fields...)

"""Ensure that `tracers` contains biogeochemical tracers and `auxiliary_fields` contains biogeochemical auxiliary fields (e.g. PAR)."""
@inline function validate_biogeochemistry(tracers, auxiliary_fields, bgc, grid)
    req_tracers = required_biogeochemical_tracers(bgc)
    tracers = all_fields_present(tracers, req_tracers, grid)

    req_auxiliary_fields = required_biogeochemical_auxiliary_fields(bgc)
    auxiliary_fields = all_fields_present(auxiliary_fields, req_auxiliary_fields, grid)
    
    return tracers, auxiliary_fields
end

required_biogeochemical_tracers(::NoBiogeochemistry) = ()
required_biogeochemical_auxiliary_fields(bgc::AbstractBiogeochemistry) = ()

"""
    SomethingBiogeochemistry <: AbstractBiogeochemistry

Sets up a tracer based biogeochemical model in a similar way to SeawaterBuoyancy.

Example
=======

@inline growth(x, y, z, t, P, μ₀, λ, m) = (μ₀ * exp(z / λ) - m) * P 

biogeochemistry = Biogeochemistry(tracers = :P, transitions = (; P=growth))
"""
struct SomethingBiogeochemistry{T, S, U, A, P, SU} <: AbstractContinuousFormBiogeochemistry
    biogeochemical_tracers :: NTuple{N, Symbol} where N
    transitions :: T
    advection_schemes :: S
    drift_velocities :: U
    auxiliary_fields :: A
    parameters :: P
    state_updates :: SU
end

@inline required_biogeochemical_tracers(bgc::SomethingBiogeochemistry) = bgc.biogeochemical_tracers
@inline required_biogeochemical_auxiliary_fields(bgc::SomethingBiogeochemistry) = bgc.auxiliary_fields

@inline biogeochemical_drift_velocity(bgc::SomethingBiogeochemistry, val_tracer_name) = bgc.drift_velocities[val_tracer_name]
@inline biogeochemical_advection_scheme(bgc::SomethingBiogeochemistry, val_tracer_name) = bgc.advection_schemes[val_tracer_name]

@inline update_biogeochemical_state!(bgc::SomethingBiogeochemistry, model) = bgc.state_updates(bgc, model)

@inline (bgc::SomethingBiogeochemistry)(::Val{name}, x, y, z, t, fields_ijk...) where name = name in bgc.biogeochemical_tracers ? bgc.transitions[name](x, y, z, t, fields_ijk..., bgc.parameters...) : 0.0

#=
function regularize_drift_velocities(drift_speeds)
    drift_velocities = []
    for w in values(drift_speeds)
        u, v, w = maybe_constant_field.((0.0, 0.0, - w))
        push!(drift_velocities, (; u, v, w))
    end

    return NamedTuple{keys(drift_speeds)}(drift_velocities)
end

@inline function SomethingBiogeochemistry(;tracers, transitions, advection_scheme=UpwindBiasedFifthOrder, drift_velocities=NamedTuple(), auxiliary_fields=())
    drift_velocities = regularize_drift_velocities(drift_velocities)
    return SomethingBiogeochemistry(tupleit(tracers), transitions, advection_scheme, drift_velocities, tupleit(auxiliary_fields))
end
=#

@inline nothingfunction(args...) = nothing

SomethingBiogeochemistry(;tracers, transitions, adveciton_schemes = nothing, drift_velocitiies = NamedTuple(), auxiliary_fields=(), parameters=NamedTuple(), state_updates = nothingfunction) = 
    SomethingBiogeochemistry(tupleit(tracers), transitions, adveciton_schemes, drift_velocitiies, tupleit(auxiliary_fields), parameters, state_updates)


end # module
