"""
    compute_forcing!(forcing)

Refresh any internal state of `forcing` that must be recomputed each step.
Called from each model's `update_state!` before tendency evaluation. Defaults
to a no-op; methods extend it for forcings carrying lazy fields (e.g. a
`Relaxation` whose target is a transform of the forced field).
"""
compute_forcing!(forcing) = nothing
compute_forcing!(t::Tuple) = foreach(compute_forcing!, t)
compute_forcing!(nt::NamedTuple) = foreach(compute_forcing!, values(nt))
compute_forcing!(mf::MultipleForcings) = compute_forcing!(mf.forcings)

compute_forcing!(r::Relaxation) =
    isnothing(r.transform) ? nothing : compute!(r.relaxed)

compute_forcing!(forcing, clock, model_fields) = nothing
compute_forcing!(t::Tuple, clock, model_fields) = foreach(f -> compute_forcing!(f, clock, model_fields), t)
compute_forcing!(nt::NamedTuple, clock, model_fields) = compute_forcing!(values(nt), clock, model_fields)
compute_forcing!(mf::MultipleForcings, clock, model_fields) = compute_forcing!(mf.forcings, clock, model_fields)

has_field_advective_forcing(forcing) = false
has_field_advective_forcing(t::Tuple) = any(has_field_advective_forcing, t)
has_field_advective_forcing(nt::NamedTuple) = has_field_advective_forcing(values(nt))
has_field_advective_forcing(mf::MultipleForcings) = has_field_advective_forcing(mf.forcings)

function synchronize_advective_forcing_dependencies!(forcing, halo_fields)
    has_field_advective_forcing(forcing) && synchronize_communication!(halo_fields)
    return nothing
end
