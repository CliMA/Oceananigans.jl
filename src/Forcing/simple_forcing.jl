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

@inline (f::SimpleForcing{X, Y, Z})(i, j, k, grid, clock, state) where {X, Y, Z} =
    @inbounds f.func(xnode(X, i, grid), ynode(Y, j, grid), znode(Z, k, grid), clock.time, f.parameters)

@inline (f::SimpleForcing{X, Y, Z, F, <:Nothing})(i, j, k, grid, clock, state) where {X, Y, Z, F} =
    @inbounds f.func(xnode(X, i, grid), ynode(Y, j, grid), znode(Z, k, grid), clock.time)
