using Oceananigans.Utils: with_tracers
using Oceananigans.TurbulenceClosures: with_tracers

@inline zeroforcing(args...) = 0

const assumed_field_locations = Dict(:tracer => (Cell, Cell, Cell),
                                          :u => (Face, Cell, Cell),
                                          :v => (Cell, Face, Cell),
                                          :w => (Face, Cell, Face))

"""
    ModelForcing(; u=zeroforcing, v=zeroforcing, w=zeroforcing, tracer_forcings...)

Return a named tuple of forcing functions for each solution field.

Example
=======

julia> u_forcing = SimpleForcing((x, y, z, t) -> exp(z) * cos(t))

julia> model = IncompressibleModel(forcing=ModelForcing(u=u_forcing))
"""
function ModelForcing(; u=zeroforcing, v=zeroforcing, w=zeroforcing, tracer_forcings...)
    u = for_field_name(:u, u)
    v = for_field_name(:v, v)
    w = for_field_name(:w, w)

    # Re-build tracer forcings
    tracer_names = Tuple(f.first for f in tracer_forcings)
    tracer_functions = Tuple(for_field_name(f.first, f.second) for f in tracer_forcings)
    tracer_forcings = NamedTuple{tracer_names}(tracer_functions)

    return merge((u=u, v=v, w=w), tracer_forcings)
end

for_field_name(field_name, forcing_function) = forcing_function # Fallback for ordinary functions

""" Returns a SimpleForcing object at `field_name`'s assumed location. """
function for_field_name(field_name, f::SimpleForcing)

    # Deduce the location of `field_name`.
    X, Y, Z = field_name ∈ (:u, :v, :w) ? assumed_field_locations[field_name] :
                                          assumed_field_locations[:tracer]

    return SimpleForcing{X, Y, Z}(field_name, f.forcing, f.parameters, f.field_in_signature)
end

default_tracer_forcing(args...) = zeroforcing

ModelForcing(tracers, proposal_forcing) = with_tracers(tracers, proposal_forcing, default_tracer_forcing,
                                                       with_velocities=true)
