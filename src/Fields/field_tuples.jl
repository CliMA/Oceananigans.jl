#####
##### Velocity fields tuples
#####

"""
    VelocityFields(arch, grid; [u, v, w])

Return a NamedTuple with fields `u`, `v`, `w` initialized on the architecture `arch`
and `grid`. Fields may be passed via the optional keyword arguments `u`, `v`, and `w`.

This function is used by OutputWriters.Checkpointer.
"""
function VelocityFields(arch, grid; u = XFaceField(arch, grid, UVelocityBoundaryConditions(grid)),
                                    v = YFaceField(arch, grid, VVelocityBoundaryConditions(grid)),
                                    w = ZFaceField(arch, grid, WVelocityBoundaryConditions(grid)))
    return (u=u, v=v, w=w)
end

"""
    VelocityFields(arch, grid, bcs::NamedTuple)

Return a NamedTuple with fields `u`, `v`, `w` initialized on the architecture `arch`
and `grid`. Boundary conditions `bcs` may be specified via a named tuple of
`FieldBoundaryCondition`s.
"""
function VelocityFields(arch, grid, bcs::NamedTuple)
    u_bcs = :u ∈ keys(bcs) ? bcs[:u] : UVelocityBoundaryConditions(grid)
    v_bcs = :v ∈ keys(bcs) ? bcs[:v] : VVelocityBoundaryConditions(grid)
    w_bcs = :w ∈ keys(bcs) ? bcs[:w] : WVelocityBoundaryConditions(grid)

    u = XFaceField(arch, grid, u_bcs)
    v = YFaceField(arch, grid, v_bcs)
    w = ZFaceField(arch, grid, w_bcs)

    return (u=u, v=v, w=w)
end


#####
##### Tracer fields tuples
#####

"""
    TracerFields(tracer_names, arch, grid; kwargs...)

Return a NamedTuple with tracer fields specified by `tracer_names` initialized as
`CellField`s on the architecture `arch` and `grid`. Fields may be passed via optional
keyword arguments `kwargs` for each field.

This function is used by OutputWriters.Checkpointer.

# Examples
```julia
arch = CPU()
topology = (Periodic, Periodic, Bounded)
grid = RegularCartesianGrid(topology=topology, size=(16, 16, 16), size=(1, 2, 3))
tracers = (:T, :S, :random)
noisy_field = CellField(arch, grid, TracerBoundaryConditions(grid), randn(16, 16))
tracer_fields = TracerFields(tracers, arch, grid, random=noisy_field)
```
"""
function TracerFields(names, arch, grid; kwargs...)
    tracer_names = tracernames(names) # filter `names` if it contains velocity fields
    tracer_fields =
        Tuple(c ∈ keys(kwargs) ?
              kwargs[c] :
              CellField(arch, grid, TracerBoundaryConditions(grid))
              for c in tracer_names)
    return NamedTuple{tracer_names}(tracer_fields)
end

"""
    TracerFields(names, arch, grid, bcs)

Return a NamedTuple with tracer fields specified by `names` initialized as
`CellField`s on the architecture `arch` and `grid`. Boundary conditions `bcs` may
be specified via a named tuple of `FieldBoundaryCondition`s.
"""
function TracerFields(names, arch, grid, bcs)
    tracer_names = tracernames(names) # filter `names` if it contains velocity fields
    tracer_fields =
        Tuple(c ∈ keys(bcs) ?
              CellField(arch, grid, bcs[c]) :
              CellField(arch, grid, TracerBoundaryConditions(grid))
              for c in tracer_names)
    return NamedTuple{tracer_names}(tracer_fields)
end

# 'Nothing', or empty tracer fields
TracerFields(::Union{Tuple{}, Nothing}, arch, grid, args...; kwargs...) = NamedTuple()

"""
    TracerFields(proposed_tracer_fields::NamedTuple, arch, grid; kwargs...)

Convenience method for restoring checkpointed models that returns the already-instantiated
`tracer_fields` with non-default boundary conditions.
"""
function TracerFields(proposed_tracer_fields::NamedTuple, arch, grid, bcs; kwargs...)

    validate_field_tuple_grid("tracers", proposed_tracer_fields, grid)

    tracer_fields =
        Tuple(c ∈ keys(bcs) ?
              Field{Cell, Cell, Cell}(proposed_tracer_fields[c].data, grid, bcs[c]) :
              proposed_tracer_fields[c]
              for c in tracernames(proposed_tracer_fields))

    return NamedTuple{tracernames(proposed_tracer_fields)}(tracer_fields)
end

"Shortcut constructor for empty tracer fields."
TracerFields(::NamedTuple{(), Tuple{}}, arch, grid, args...; kwargs...) = NamedTuple()

#####
##### Pressure fields tuples
#####

"""
    PressureFields(arch, grid; [pHY′, pNHS])

Return a NamedTuple with pressure fields `pHY′` and `pNHS` initialized as
`CellField`s on the architecture `arch` and `grid`. Fields may be passed via the
optional keyword arguments `pHY′` and `pNHS`.
"""
function PressureFields(arch, grid; pHY′ = CellField(arch, grid, PressureBoundaryConditions(grid)),
                                    pNHS = CellField(arch, grid, PressureBoundaryConditions(grid)))

    return (pHY′=pHY′, pNHS=pNHS)
end

"""
    PressureFields(arch, grid, bcs::NamedTuple)

Return a NamedTuple with pressure fields `pHY′` and `pNHS` initialized as
`CellField`s on the architecture `arch` and `grid`.  Boundary conditions `bcs` may
be specified via a named tuple of `FieldBoundaryCondition`s.
"""
function PressureFields(arch, grid, bcs::NamedTuple)
    pHY′_bcs = :pHY′ ∈ keys(bcs) ? bcs[:pHY′] : PressureBoundaryConditions(grid)
    pNHS_bcs = :pNHS ∈ keys(bcs) ? bcs[:pNHS] : PressureBoundaryConditions(grid)

    pHY′ = CellField(arch, grid, pHY′_bcs)
    pNHS = CellField(arch, grid, pNHS_bcs)

    return (pHY′=pHY′, pNHS=pNHS)
end

"""
    TendencyFields(arch, grid, tracer_names; kwargs...)

Return a NamedTuple with tendencies for all solution fields (velocity fields and
tracer fields), initialized on the architecture `arch` and `grid`. Optional `kwargs`
can be specified to assign data arrays to each tendency field.
"""
function TendencyFields(arch, grid, tracer_names;
                        u = XFaceField(arch, grid, UVelocityBoundaryConditions(grid)),
                        v = YFaceField(arch, grid, VVelocityBoundaryConditions(grid)),
                        w = ZFaceField(arch, grid, WVelocityBoundaryConditions(grid)),
                        kwargs...)

    velocities = (u=u, v=v, w=w)
    tracers = TracerFields(tracer_names, arch, grid; kwargs...)
    return merge(velocities, tracers)
end

#####
##### Helper functions for IncompressibleModel constructor
#####

VelocityFields(::Nothing, arch, grid, bcs) = VelocityFields(arch, grid, bcs)

VelocityFields(velocities::NamedTuple{(:u, :v, :w)}, arch, grid, bcs) =
    validate_field_tuple_grid("velocities", velocities, grid)

PressureFields(::Nothing, arch, grid, bcs) = PressureFields(arch, grid, bcs)

PressureFields(pressures::NamedTuple{(:pHY′, :pNHS)}, arch, grid; kwargs...) =
    validate_field_tuple_grid("pressures", pressures, grid)

