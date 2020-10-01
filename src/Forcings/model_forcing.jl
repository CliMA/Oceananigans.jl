using Oceananigans.Utils: with_tracers
using Oceananigans.Fields: assumed_field_location

@inline zeroforcing(args...) = 0

"""
    regularize_forcing(forcing, field_name)

"Regularizes" or "adds information" to
user-defined forcing objects that are passed to the `IncompressibleModel`
constructor. `regularize_forcing` is called inside `model_forcing`.
We need `regularize_forcing` because it is only until `model_forcing` is called that
the *names* of various forcing fields are available. The `field_name` can be used to infer
the location at which the forcing is applied, or to add a field dependency
to a special forcing object, as for `Relxation`.
"""
regularize_forcing(forcing, field_name) = forcing # fallback

""" Move `forcing` to the location of `field_name`. """
function regularize_forcing(forcing::ContinuousForcing, field_name)
    X, Y, Z = assumed_field_location(field_name)
    return ContinuousForcing{X, Y, Z}(forcing.func, forcing.parameters, forcing.field_dependencies)
end

""" Wrap `forcing` in a `ContinuousForcing` at the location of `field_name`. """
function regularize_forcing(forcing::Function, field_name)
    X, Y, Z = assumed_field_location(field_name)
    return ContinuousForcing{X, Y, Z}(forcing, nothing, ())
end

regularize_forcing(::Nothing, field_name) = zeroforcing

"""
    model_forcing(; u=zeroforcing, v=zeroforcing, w=zeroforcing, tracer_forcings...)

Return a named tuple of forcing functions for each solution field, wrapping
forcing function in `ContinuousForcing`s and ensuring that
`ContinuousForcing`s are located correctly for velocity fields.
"""
function model_forcing(tracer_names; u=nothing, v=nothing, w=nothing, tracer_forcings...)
    u = regularize_forcing(u, :u)
    v = regularize_forcing(v, :v)
    w = regularize_forcing(w, :w)

    # Build tuple of user-specified tracer forcings
    specified_tracer_forcings_tuple = Tuple(regularize_forcing(f.second, f.first) for f in tracer_forcings)
    specified_tracer_names = Tuple(f.first for f in tracer_forcings)

    specified_forcings = NamedTuple{specified_tracer_names}(specified_tracer_forcings_tuple)

    # Re-build with defaults for unspecified tracer forcing
    tracer_forcings = with_tracers(tracer_names, specified_forcings, (name, initial_tuple) -> zeroforcing)

    return merge((u=u, v=v, w=w), tracer_forcings)
end
