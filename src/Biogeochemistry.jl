module Biogeochemistry

using DocStringExtensions: TYPEDSIGNATURES
using KernelAbstractions: @kernel, @index
using Oceananigans.Architectures: architecture
using Oceananigans.Grids: Center, xnode, ynode, znode
using Oceananigans.Utils: launch!

import Oceananigans.Fields: CenterField

#####
##### Generic fallbacks for biogeochemistry
#####

@inline biogeochemistry_rhs(i, j, k, grid, ::Nothing, val_tracer_name, clock, fields) = zero(grid)

"""
$(TYPEDSIGNATURES)

Update prognostic tendencies after they have been computed.
"""
update_tendencies!(bgc, model) = nothing

"""
$(TYPEDSIGNATURES)

Update biogeochemical state variables. Called at the end of update_state!.
"""
update_biogeochemical_state!(bgc, model) = nothing

@inline biogeochemical_drift_velocity(bgc, val_tracer_name) = nothing
@inline biogeochemical_auxiliary_fields(bgc) = NamedTuple()

"""
    AbstractBiogeochemistry

Abstract type for biogeochemical models. To define a biogeochemcial relationship
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

  - `update_biogeochemical_state!(bgc::BiogeochemicalModel, model)` (optional) to update the
      model state.
"""
abstract type AbstractBiogeochemistry end

# Returns the forcing for discrete form models
@inline biogeochemical_transition(i, j, k, grid, bgc, val_tracer_name, clock, fields) =
    bgc(i, j, k, grid, val_tracer_name, clock, fields)

@inline biogeochemical_transition(i, j, k, grid, ::Nothing, val_tracer_name, clock, fields) = zero(grid)

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

    x = xnode(i, j, k, grid, Center(), Center(), Center())
    y = ynode(i, j, k, grid, Center(), Center(), Center())
    z = znode(i, j, k, grid, Center(), Center(), Center())

    return bgc(val_tracer_name, x, y, z, clock.time, fields_ijk...)
end

@inline (bgc::AbstractContinuousFormBiogeochemistry)(val_tracer_name, x, y, z, t, fields...) = zero(t)

tracernames(tracers) = keys(tracers)
tracernames(tracers::Tuple) = tracers

add_biogeochemical_tracer(tracers::Tuple, name, grid) = tuple(tracers..., name)
add_biogeochemical_tracer(tracers::NamedTuple, name, grid) = merge(tracers, (; name => CenterField(grid)))

@inline function has_biogeochemical_tracers(fields, required_fields, grid)
    user_specified_tracers = [name in tracernames(fields) for name in required_fields]

    flds = if !all(user_specified_tracers) && any(user_specified_tracers)
        throw(ArgumentError("The biogeochemical model you have selected requires $required_fields.\n" *
                            "You have specified some but not all of these as tracers so may be attempting\n" *
                            "to use them for a different purpose. Please either specify all of the required\n" *
                            "fields, or none and allow them to be automatically added."))

    elseif !any(user_specified_tracers)
        f = fields
        for field_name in required_fields
            f = add_biogeochemical_tracer(f, field_name, grid)
        end
        f
    else
        fields
    end

    return flds
end

"""
$(TYPEDSIGNATURES)

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

#####
##### Optionally computing biogeochemical transitions in separate kernels
#####

"""
$(TYPEDSIGNATURES)

Return a tuple of tracer names whose biogeochemical transition is computed in a separate kernel,
launched after the tracer tendency kernels, rather than inline with advection and diffusion.
The separately computed transition is added in place to the tracer tendency `Gⁿ[name]`, so no extra
storage is required.

The default is `()`, for which the transition is computed inline in the tracer tendency kernel.
Splitting can make tendency kernels for complex biogeochemical models substantially cheaper to
compile and faster to run (especially on GPUs). Opt in by extending this function, e.g.

```julia
Oceananigans.Biogeochemistry.separate_tracer_transitions(bgc::MyBGC) = required_biogeochemical_tracers(bgc)
```
"""
separate_tracer_transitions(bgc) = ()

"""
$(TYPEDSIGNATURES)

Add the biogeochemical transition of each tracer in `separate_tracer_transitions(bgc)` in place to
the corresponding tendency `Gⁿ[name]`, using one kernel launch per tracer. `model_fields` must be the
same fields that the tracer tendency kernel passes to `biogeochemical_transition`.
Does nothing when `separate_tracer_transitions(bgc)` is empty.
"""
add_biogeochemical_transitions!(Gⁿ, bgc, grid, clock, model_fields;
                                kernel_parameters=:xyz, active_cells_map=nothing) =
    add_biogeochemical_transitions!(Gⁿ, bgc, grid, clock, model_fields, separate_tracer_transitions(bgc);
                                    kernel_parameters, active_cells_map)

add_biogeochemical_transitions!(Gⁿ, bgc, grid, clock, model_fields, ::Tuple{}; kwargs...) = nothing

@inline function add_biogeochemical_transitions!(Gⁿ, bgc, grid, clock, model_fields, names::Tuple;
                                                 kernel_parameters=:xyz, active_cells_map=nothing)
    name = first(names)
    launch!(architecture(grid), grid, kernel_parameters, _add_biogeochemical_transition!,
            Gⁿ[name], grid, bgc, Val(name), clock, model_fields; active_cells_map)
    return add_biogeochemical_transitions!(Gⁿ, bgc, grid, clock, model_fields, Base.tail(names);
                                           kernel_parameters, active_cells_map)
end

@kernel function _add_biogeochemical_transition!(Gc, grid, bgc, val_tracer_name, clock, fields)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] += biogeochemical_transition(i, j, k, grid, bgc, val_tracer_name, clock, fields)
end

"""
$(TYPEDSIGNATURES)

Return `false` if the biogeochemical transition of tracer `name` is computed in a separate kernel
(that is, if `name` is in [`separate_tracer_transitions`](@ref)), so that the tracer tendency kernel
does not include it; otherwise return `true`.
"""
@inline include_biogeochemistry_transitions(biogeochemistry, ::Val{name}) where name =
    !(name in separate_tracer_transitions(biogeochemistry))

const AbstractBGCOrNothing = Union{Nothing, AbstractBiogeochemistry}
required_biogeochemical_tracers(::AbstractBGCOrNothing) = ()
required_biogeochemical_auxiliary_fields(::AbstractBGCOrNothing) = ()

end # module
