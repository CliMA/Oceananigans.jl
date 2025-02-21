using Oceananigans.Fields: ZeroField, AbstractField, FunctionField, location
using Oceananigans.Utils: prettysummary

using Adapt

function background_velocity_fields(fields, grid, clock)
    u = get(fields, :u, ZeroField())
    v = get(fields, :v, ZeroField())
    w = get(fields, :w, ZeroField())

    u = regularize_background_field(Face, Center, Center, u, grid, clock)
    v = regularize_background_field(Center, Face, Center, v, grid, clock)
    w = regularize_background_field(Center, Center, Face, w, grid, clock)

    return (; u, v, w)
end

function background_tracer_fields(bg, tracer_names, grid, clock)
    tracer_fields =
        Tuple(c ∈ keys(bg) ?
              regularize_background_field(Center, Center, Center, getindex(bg, c), grid, clock) :
              ZeroField()
              for c in tracer_names)
        
    return NamedTuple{tracer_names}(tracer_fields)
end

#####
##### BackgroundFields (with option for including background closure fluxes)
#####

struct BackgroundFields{Q, U, C}
    velocities :: U
    tracers :: C
    function BackgroundFields{Q}(velocities::U, tracers::C) where {Q, U, C}
        return new{Q, U, C}(velocities, tracers)
    end
end

Adapt.adapt_structure(to, bf::BackgroundFields{Q}) where Q =    
    BackgroundFields{Q}(adapt(to, bf.velocities), adapt(to, bf.tracers))

const BackgroundFieldsWithClosureFluxes = BackgroundFields{true}

function BackgroundFields(; background_closure_fluxes=false, fields...)
    u = get(fields, :u, ZeroField())
    v = get(fields, :v, ZeroField())
    w = get(fields, :w, ZeroField())
    velocities = (; u, v, w)
    tracers = NamedTuple(name => fields[name] for name in keys(fields) if !(name ∈ (:u, :v, :w)))
    return BackgroundFields{background_closure_fluxes}(velocities, tracers)
end

function BackgroundFields(background_fields::BackgroundFields{Q}, tracer_names, grid, clock) where Q
    velocities = background_velocity_fields(background_fields.velocities, grid, clock)
    tracers = background_tracer_fields(background_fields.tracers, tracer_names, grid, clock)
    return BackgroundFields{Q}(velocities, tracers)
end

function BackgroundFields(background_fields::NamedTuple, tracer_names, grid, clock)
    velocities = background_velocity_fields(background_fields, grid, clock)
    tracers = background_tracer_fields(background_fields, tracer_names, grid, clock)
    return BackgroundFields{false}(velocities, tracers)
end

"""
    BackgroundField{F, P}

Temporary container for storing information about `BackgroundFields`.
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
    FunctionField{LX, LY, LZ}(f.func, grid; clock, parameters=f.parameters)

regularize_background_field(LX, LY, LZ, func::Function, grid, clock) =
    FunctionField{LX, LY, LZ}(func, grid; clock=clock)

regularize_background_field(LX, LY, LZ, ::ZeroField, grid, clock) = ZeroField()

function regularize_background_field(LX, LY, LZ, field::AbstractField, grid, clock)
    if location(field) != (LX, LY, LZ)
        throw(ArgumentError("Cannot use field at $(location(field)) as a background field at $((LX, LY, LZ))"))
    end
    
    return field
end

Base.show(io::IO, field::BackgroundField{F, P}) where {F, P} =
    print(io, "BackgroundField{$F, $P}", "\n",
          "├── func: $(prettysummary(field.func))", "\n",
          "└── parameters: $(field.parameters)")
