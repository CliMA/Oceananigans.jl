"""
    SimpleForcing{X, Y, Z, M, F, P}

A callable object that implements a 'simple' forcing function
on a field at the location `X, Y, Z`.
"""
struct SimpleForcing{X, Y, Z, M, P, N, C, F}
                forcing :: F
             parameters :: P
     field_in_signature :: Bool

    function SimpleForcing{X, Y, Z}(field_name, forcing, parameters, field_in_signature) where {X, Y, Z}

        # Deduce the container where `field_name` is found. This is only needed
        # if field_in_signature=true.
        field_container = field_name ∈ (:u, :v, :w) ? :velocities : :tracers

        return new{X, Y, Z, field_in_signature, typeof(parameters), field_name, field_container,
                   typeof(forcing)}(forcing, parameters, field_in_signature)
    end
end

"""
    SimpleForcing(forcing; parameters=nothing, field_in_signature=false)

Construct forcing for a field named `field_name` based on a `forcing` function,
with optional parameters. The keyword arguments determine the expected function 
signature of `forcing`. The keyword `field_in_signature=true` specifies that the field
being forced appears in the user-defined function signature.
If `parameters` is anything other than `nothing`, it is assumed to be part of the
function signature of `forcing`.

The four possible signatures of function `forcing` are thus

* `parameters=nothing, field_in_signature=false`:

    `forcing(x, y, z, t)`

This is the function signature for default choices of `parameters` and `field_in_signature`.

* `field_in_signature=false`, specified parameters:

    `forcing(x, y, z, t, parameters)`,

where `parameters` is the object passed as keyword argument. To compile on the GPU
this must be a simple object; typically, a `NamedTuple` of floats and other constants.

* `parameters=nothing, field_in_signature=true`:

    `forcing(x, y, z, t, field)`

where `field` is the value of the field the forcing is applied to at `x, y, z`.

* `field_in_signature=true`, specified parameters:

    `forcing(x, y, z, t, field, parameters)`

Examples
========

* The simplest case: no parameters, additive forcing:

```julia
julia> const a = 2.1

julia> fun_forcing(x, y, z, t) = a * exp(z) * cos(t)

julia> u_forcing = SimpleForcing(fun_forcing)
```

* Parameterized, additive forcing:

```julia
julia> parameterized_forcing(x, y, z, t, p) = p.μ * exp(z / p.λ) * cos(p.ω * t)

julia> v_forcing = SimpleForcing(parameterized_forcing, parameters = (μ=42, λ=0.1, ω=π))
```

* Field-dependent forcing with no parameters:

```julia
julia> growth_in_sunlight(x, y, z, t, P) = exp(z) * P

julia> plankton_forcing = SimpleForcing(growth_in_sunlight, field_in_signature=true)
```

* Field-dependent forcing with parameters. This example relaxes a tracer to some reference
    linear profile.

```julia
julia> tracer_relaxation(x, y, z, t, c, p) = p.μ * exp((z + p.H) / p.λ) * (p.dCdz * z - c) 

julia> c_forcing = SimpleForcing(tracer_relaxation, parameters=(μ=1/60, λ=10, H=1000, dCdz=1), 
                                 field_in_signature=true)
```
"""
SimpleForcing(forcing; parameters=nothing, field_in_signature=false) =
    SimpleForcing{Cell, Cell, Cell}(:tracer, forcing, parameters, field_in_signature)

# Simple additive forcing without parameters
@inline function (f::SimpleForcing{X, Y, Z, false, <:Nothing})(i, j, k, grid, clock, state) where {X, Y, Z}

    return @inbounds f.forcing(xnode(X, i, grid),
                               ynode(Y, j, grid),
                               znode(Z, k, grid),
                               clock.time)
end
        
# Simple additive forcing with parameters
@inline function (f::SimpleForcing{X, Y, Z, false})(i, j, k, grid, clock, state) where {X, Y, Z}

    return @inbounds f.forcing(xnode(X, i, grid),
                               ynode(Y, j, grid),
                               znode(Z, k, grid),
                               clock.time,
                               f.parameters)
end

# Simple field-dependent forcing without parameters
@inline function (f::SimpleForcing{X, Y, Z, true, <:Nothing, N, C})(i, j, k, grid, clock, state) where {X, Y, Z, N, C}
    container = getproperty(state, C)
    field = getproperty(container, N)

    return @inbounds f.forcing(xnode(X, i, grid),
                               ynode(Y, j, grid),
                               znode(Z, k, grid),
                               field[i, j, k],
                               clock.time)
end

# Simple field-dependent forcing with parameters
@inline function (f::SimpleForcing{X, Y, Z, true, P, N, C})(i, j, k, grid, clock, state) where {X, Y, Z, P, N, C}
    container = getproperty(state, C)
    field = getproperty(container, N)

    return @inbounds f.forcing(xnode(X, i, grid),
                               ynode(Y, j, grid),
                               znode(Z, k, grid),
                               clock.time,
                               field[i, j, k],
                               f.parameters)
end

