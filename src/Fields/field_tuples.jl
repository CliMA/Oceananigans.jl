using Oceananigans.BoundaryConditions: FieldBoundaryConditions, regularize_field_boundary_conditions

#####
##### Velocity fields tuples
#####

"""
    VelocityFields(grid, user_bcs = NamedTuple())

Return a `NamedTuple` with fields `u`, `v`, `w` initialized on `grid`.
Boundary conditions `bcs` may be specified via a named tuple of
`FieldBoundaryCondition`s.
"""
function VelocityFields(grid::AbstractGrid, user_bcs = NamedTuple())

    template = FieldBoundaryConditions()

    default_bcs = (
        u = regularize_field_boundary_conditions(template, grid, :u),
        v = regularize_field_boundary_conditions(template, grid, :v),
        w = regularize_field_boundary_conditions(template, grid, :w)
    )

    bcs = merge(default_bcs, user_bcs)

    u = XFaceField(grid, boundary_conditions=bcs.u)
    v = YFaceField(grid, boundary_conditions=bcs.v)
    w = ZFaceField(grid, boundary_conditions=bcs.w)

    return (u=u, v=v, w=w)
end

#####
##### Tracer fields tuples
#####

"""
    TracerFields(tracer_names, grid, user_bcs)

Return a `NamedTuple` with tracer fields specified by `tracer_names` initialized as
`CenterField`s on `grid`. Boundary conditions `user_bcs`
may be specified via a named tuple of `FieldBoundaryCondition`s.
"""
function TracerFields(tracer_names, grid, user_bcs)
    default_bcs = NamedTuple(name => FieldBoundaryConditions(grid, (Center, Center, Center)) for name in tracer_names)
    bcs = merge(default_bcs, user_bcs) # provided bcs overwrite defaults
    return NamedTuple(c => CenterField(grid, boundary_conditions=bcs[c]) for c in tracer_names)
end

"""
    TracerFields(tracer_names, grid; kwargs...)

Return a `NamedTuple` with tracer fields specified by `tracer_names` initialized as
`CenterField`s on `grid`. Fields may be passed via optional
keyword arguments `kwargs` for each field.

This function is used by `OutputWriters.Checkpointer` and `TendencyFields`.
```
"""
TracerFields(tracer_names, grid; kwargs...) =
    NamedTuple(c => c ∈ keys(kwargs) ? kwargs[c] : CenterField(grid) for c in tracer_names)

# 'Nothing', or empty tracer fields
TracerFields(::Union{Tuple{}, Nothing}, grid, bcs) = NamedTuple()

"Shortcut constructor for empty tracer fields."
TracerFields(::NamedTuple{(), Tuple{}}, grid, bcs) = NamedTuple()

#####
##### Pressure fields tuples
#####

"""
    PressureFields(grid, bcs::NamedTuple)

Return a `NamedTuple` with pressure fields `pHY′` and `pNHS` initialized as
`CenterField`s on `grid`.  Boundary conditions `bcs`
may be specified via a named tuple of `FieldBoundaryCondition`s.
"""
function PressureFields(grid, bcs=NamedTuple())

    default_pressure_boundary_conditions =
        (pHY′ = FieldBoundaryConditions(grid, (Center, Center, Center)),
         pNHS = FieldBoundaryConditions(grid, (Center, Center, Center)))

    bcs = merge(default_pressure_boundary_conditions, bcs)

    pHY′ = CenterField(grid, boundary_conditions=bcs.pHY′)
    pNHS = CenterField(grid, boundary_conditions=bcs.pNHS)

    return (pHY′=pHY′, pNHS=pNHS)
end

function PressureFields(grid::AbstractGrid{<:Any, <:Any, <:Any, <:Flat}, bcs=NamedTuple())
    default_pressure_boundary_conditions =
        (pHY′ = FieldBoundaryConditions(grid, (Center, Center, Center)),
         pNHS = FieldBoundaryConditions(grid, (Center, Center, Center)))

    bcs = merge(default_pressure_boundary_conditions, bcs)
    pNHS = CenterField(grid, boundary_conditions=bcs.pNHS)

    return (; pHY′=nothing, pNHS=pNHS)
end

"""
    TendencyFields(grid, tracer_names;
                   u = XFaceField(grid),
                   v = YFaceField(grid),
                   w = ZFaceField(grid),
                   kwargs...)

Return a `NamedTuple` with tendencies for all solution fields (velocity fields and
tracer fields), initialized on `grid`. Optional `kwargs`
can be specified to assign data arrays to each tendency field.
"""
function TendencyFields(grid, tracer_names;
                        u = XFaceField(grid),
                        v = YFaceField(grid),
                        w = ZFaceField(grid),
                        kwargs...)

    velocities = (u=u, v=v, w=w)

    tracers = TracerFields(tracer_names, grid; kwargs...)

    return merge(velocities, tracers)
end

#####
##### Helper functions for NonhydrostaticModel constructor
#####

VelocityFields(::Nothing, grid, bcs) = VelocityFields(grid, bcs)
PressureFields(::Nothing, grid, bcs) = PressureFields(grid, bcs)

"""
    VelocityFields(proposed_velocities::NamedTuple{(:u, :v, :w)}, grid, bcs)

Return a `NamedTuple` of velocity fields, overwriting boundary conditions
in `proposed_velocities` with corresponding fields in the `NamedTuple` `bcs`.
"""
function VelocityFields(proposed_velocities::NamedTuple{(:u, :v, :w)}, grid, bcs)

    validate_field_tuple_grid("velocities", proposed_velocities, grid)

    u = XFaceField(grid, boundary_conditions=bcs.u, data=proposed_velocities.u.data)
    v = YFaceField(grid, boundary_conditions=bcs.v, data=proposed_velocities.v.data)
    w = ZFaceField(grid, boundary_conditions=bcs.w, data=proposed_velocities.w.data)

    return (u=u, v=v, w=w)
end

"""
    TracerFields(proposed_tracers::NamedTuple, grid, bcs)

Return a `NamedTuple` of tracers, overwriting boundary conditions
in `proposed_tracers` with corresponding fields in the `NamedTuple` `bcs`.
"""
function TracerFields(proposed_tracers::NamedTuple, grid, bcs)

    validate_field_tuple_grid("tracers", proposed_tracers, grid)

    tracer_names = propertynames(proposed_tracers)
    tracer_fields = Tuple(CenterField(grid, boundary_conditions=bcs[c], data=proposed_tracers[c].data) for c in tracer_names)

    return NamedTuple{tracer_names}(tracer_fields)
end

"""
    PressureFields(proposed_pressures::NamedTuple{(:pHY′, :pNHS)}, grid, bcs)

Return a `NamedTuple` of pressure fields with, overwriting boundary conditions
in `proposed_tracer_fields` with corresponding fields in the `NamedTuple` `bcs`.
"""
function PressureFields(proposed_pressures::NamedTuple{(:pHY′, :pNHS)}, grid, bcs)
    validate_field_tuple_grid("pressures", proposed_pressures, grid)

    pHY′ = CenterField(grid, boundary_conditions=bcs.pHY′, data=proposed_pressures.pHY′.data)
    pNHS = CenterField(grid, boundary_conditions=bcs.pNHS, data=proposed_pressures.pNHS.data)

    return (pHY′=pHY′, pNHS=pNHS)
end
