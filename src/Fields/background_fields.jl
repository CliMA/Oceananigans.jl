function BackgroundVelocityFields(bg, grid, clock)
    u = :u ∈ keys(bg) ? FunctionField{Face, Cell, Cell}(bg.u, grid, clock=clock) : ZeroField()
    v = :v ∈ keys(bg) ? FunctionField{Cell, Face, Cell}(bg.v, grid, clock=clock) : ZeroField()
    w = :w ∈ keys(bg) ? FunctionField{Cell, Cell, Face}(bg.w, grid, clock=clock) : ZeroField()

    return (u=u, v=v, w=w)
end

function BackgroundTracerFields(bg, tracer_names, grid, clock)
    tracer_fields =
        Tuple(c ∈ keys(bg) ? FunctionField{Cell, Cell, Cell}(getproperty(bg, c), grid, clock=clock) : ZeroField()
              for c in tracer_names)
        
    return NamedTuple{tracer_names}(tracer_fields)
end

#####
##### Convenience for model constructor
#####

function BackgroundFields(background_fields, tracer_names, grid, clock)
    velocities = BackgroundVelocityFields(background_fields, grid, clock)
    tracers = BackgroundTracerFields(background_fields, tracer_names, grid, clock)
    return (velocities=velocities, tracers=tracers)
end

"""
    BackgroundField(func; parameters=nothing)

Returns a `BackgroundField` to be passed to `IncompressibleModel` for use
as a background velocity or tracer field.

If `parameters` is not provided, `func` must be callable with the signature

```julia
func(x, y, z, t)
```

If `parameters` is provided, `func` must be callable with the signature

```julia
func(x, y, z, t, parameters)
```

Note: `BackgroundField` is re-wrapped in a new `FunctionField` at the correct
location, on the correct `grid`, and with the correct `clock` within the
constructor for `IncompressibleModel`.
"""
BackgroundField(func; parameters=nothing) =
    FunctionField{Cell, Cell, Cell}(func, nothing; parameters=parameters)

