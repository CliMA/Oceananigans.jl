using Oceananigans.Utils: with_tracers
using Oceananigans.TurbulenceClosures: with_tracers

"""
    ModelForcing(; u=zeroforcing, v=zeroforcing, w=zeroforcing, tracer_forcings...)

Return a named tuple of forcing functions for each solution field.

Example
=======

julia> u_forcing = SimpleForcing((x, y, z, t) -> exp(z) * cos(t))

julia> model = IncompressibleModel(forcing=ModelForcing(u=u_forcing))
"""
function ModelForcing(; u=zeroforcing, v=zeroforcing, w=zeroforcing, tracer_forcings...)
    u = at_location((Face, Cell, Cell), u)
    v = at_location((Cell, Face, Cell), v)
    w = at_location((Cell, Cell, Face), w)

    return merge((u=u, v=v, w=w), tracer_forcings)
end

at_location(location, u::Function) = u
at_location(location, u::SimpleForcing) =
    SimpleForcing{location[1], location[2], location[3]}(u.func, u.parameters)

default_tracer_forcing(args...) = zeroforcing
ModelForcing(tracers, proposal_forcing) = with_tracers(tracers, proposal_forcing, default_tracer_forcing,
                                                       with_velocities=true)
