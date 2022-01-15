# TODO: This code belongs in the Models module

function BackgroundVelocityFields(bg, grid, clock)
    u = :u ∈ keys(bg) ? regularize_background_field(Face, Center, Center, bg[:u], grid, clock) : ZeroField()
    v = :v ∈ keys(bg) ? regularize_background_field(Center, Face, Center, bg[:v], grid, clock) : ZeroField()
    w = :w ∈ keys(bg) ? regularize_background_field(Center, Center, Face, bg[:w], grid, clock) : ZeroField()

    return (u=u, v=v, w=w)
end

function BackgroundTracerFields(bg, tracer_names, grid, clock)
    tracer_fields =
        Tuple(c ∈ keys(bg) ?
              regularize_background_field(Center, Center, Center, getindex(bg, c), grid, clock) :
              ZeroField()
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
    BackgroundField{F, P}

Temporary container for storing information about BackgroundFields.
"""
struct BackgroundField{F, P}
    func:: F
    parameters :: P
end

"""
    BackgroundField(func; parameters=nothing)

Returns a `BackgroundField` to be passed to `NonhydrostaticModel` for use
as a background velocity or tracer field.

If `parameters` is not provided, `func` must be callable with the signature

```julia
func(x, y, z, t)
```

If `parameters` is provided, `func` must be callable with the signature

```julia
func(x, y, z, t, parameters)
```
"""
BackgroundField(func; parameters=nothing) = BackgroundField(func, parameters)

regularize_background_field(LX, LY, LZ, f::BackgroundField{<:Function}, grid, clock) =
    FunctionField{LX, LY, LZ}(f.func, grid; clock=clock, parameters=f.parameters)

regularize_background_field(LX, LY, LZ, func::Function, grid, clock) =
    FunctionField{LX, LY, LZ}(func, grid; clock=clock)

function regularize_background_field(LX, LY, LZ, field::AbstractField, grid, clock)
    if location(field) != (LX, LY, LZ)
        throw(ArgumentError("Cannot use field at $(location(field)) as a background field at $((LX, LY, LZ))"))
    end
    
    return field
end

Base.show(io::IO, field::BackgroundField{F, P}) where {F, P} =
    print(io, "BackgroundField{$F, $P}", '\n',
          "├── func: $(short_show(field.func))", '\n',
          "└── parameters: $(field.parameters)")
