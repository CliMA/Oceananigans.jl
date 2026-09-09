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
