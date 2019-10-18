"""
    SimpleForcing{X, Y, Z, F, P}

Callable object for specifying 'simple' forcings of `x, y, z, t` and optionally
`parameters` of type `P` at location `X, Y, Z`.
"""
struct SimpleForcing{X, Y, Z, F, P}
          func :: F
    parameters :: P
    function SimpleForcing{X, Y, Z}(func, parameters) where {X, Y, Z}
        return new{X, Y, Z, typeof(func), typeof(parameters)}(func, parameters)
    end
end

"""
    SimpleForcing([location=(Cell, Cell, Cell),] forcing; parameters=nothing)

Construct forcing for a field at `location` using `forcing::Function`, and optionally
with `parameters`. If `parameters=nothing`, `forcing` must have the signature

    `forcing(x, y, z, t)`;

otherwise it must have the signature

    `forcing(x, y, z, t, parameters)`.

Examples
========
```julia
julia> const a = 2.1

julia> fun_forcing(x, y, z, t) = a * exp(z) * cos(t)

julia> u_forcing = SimpleForcing(fun_forcing)

julia> parameterized_forcing(x, y, z, t, p) = p.μ * exp(z/p.λ) * cos(p.ω*t)

julia> v_forcing = SimpleForcing(parameterized_forcing, parameters=(μ=42, λ=0.1, ω=π))
```
"""
SimpleForcing(location::Tuple, func::Function; parameters=nothing) =
    SimpleForcing{location[1], location[2], location[3]}(func, parameters)

SimpleForcing(func::Function; kwargs...) = SimpleForcing((Cell, Cell, Cell), func; kwargs...)

SimpleForcing(location::Tuple, forcing::SimpleForcing) = SimpleForcing(location, forcing.func)

@inline (f::SimpleForcing{X, Y, Z})(i, j, k, grid, time, U, C, params) where {X, Y, Z} =
    @inbounds f.func(xnode(X, i, grid), ynode(Y, j, grid), znode(Z, k, grid), time, f.parameters)

@inline (f::SimpleForcing{X, Y, Z, F, <:Nothing})(i, j, k, grid, time, U, C, params) where {X, Y, Z, F} =
    @inbounds f.func(xnode(X, i, grid), ynode(Y, j, grid), znode(Z, k, grid), time)

at_location(location, u::Function) = u
at_location(location, u::SimpleForcing) = SimpleForcing{location[1], location[2], location[3]}(u.func, u.parameters)

zeroforcing(args...) = 0

"""
    ModelForcing(; u=zeroforcing, v=zeroforcing, w=zeroforcing, tracer_forcings...)

Return a named tuple of forcing functions for each solution field.

Example
=======

julia> u_forcing = SimpleForcing((x, y, z, t) -> exp(z) * cos(t))

julia> model = Model(forcing=ModelForcing(u=u_forcing))
"""
function ModelForcing(; u=zeroforcing, v=zeroforcing, w=zeroforcing, tracer_forcings...)
    u = at_location((Face, Cell, Cell), u)
    v = at_location((Cell, Face, Cell), v)
    w = at_location((Cell, Cell, Face), w)

    return merge((u=u, v=v, w=w), tracer_forcings)
end

default_tracer_forcing(args...) = zeroforcing
ModelForcing(tracers, proposal_forcing) = with_tracers(tracers, proposal_forcing, default_tracer_forcing, 
                                                       with_velocities=true)
