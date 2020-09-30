"""
    VelocityFields(arch, grid; [u, v, w])

Return a NamedTuple with fields `u`, `v`, `w` initialized on the architecture `arch`
and `grid`. Fields may be passed via the optional keyword arguments `u`, `v`, and `w`.
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

VelocityFields(::Nothing, arch, grid, bcs) = VelocityFields(arch, grid, bcs)

VelocityFields(velocities::NamedTuple{(:u, :v, :w)}, arch, grid, bcs) =
    validate_field_tuple_grid("velocities", velocities, grid)

"""
    TracerFields(tracer_names, arch, grid; kwargs...)

Return a NamedTuple with tracer fields specified by `tracer_names` initialized as
`CellField`s on the architecture `arch` and `grid`. Fields may be passed via optional
keyword arguments `kwargs` for each field.

# Examples
```julia
arch = CPU()
topology = (Periodic, Periodic, Bounded)
grid = RegularCartesianGrid(topology=topology, size=(16, 16, 16), size=(1, 2, 3))
tracers = (:T, :S, :random)
noisy_field = CellField(arch, grid, TracerBoundaryConditions(grid), randn(16, 16))
tracer_fields = TracerFields(arch, grid, tracers, random=noisy_field)
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
    TracerFields(tracer_names, arch, grid, bcs)

Return a NamedTuple with tracer fields specified by `tracer_names` initialized as
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

TracerFields(::Union{Tuple{}, Nothing}, arch, grid, args...; kwargs...) = NamedTuple()
TracerFields(tracer::Symbol, arch, grid, bcs) = TracerFields(arch, grid, tuple(tracer), bcs)

"""
    TracerFields(tracer_fields::NamedTuple, arch, grid; kwargs...)

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
TracerFields(empty_tracer_fields::NamedTuple{(),Tuple{}}, arch, grid, args...; kwargs...) = NamedTuple()

"Returns true if the first three elements of `names` are `(:u, :v, :w)`."
has_velocities(names) = :u == names[1] && :v == names[2] && :w == names[3]

# Tuples of length 0-2 cannot contain velocity fields
has_velocities(::Tuple{}) = false
has_velocities(::Tuple{X}) where X = false
has_velocities(::Tuple{X, Y}) where {X, Y} = false

tracernames(::Nothing) = ()
tracernames(name::Symbol) = tuple(name)
tracernames(names::NTuple{N, Symbol}) where N = has_velocities(names) ? names[4:end] : names
tracernames(::NamedTuple{names}) where names = tracernames(names)

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

PressureFields(::Nothing, arch, grid; kwargs...) = PressureFields(arch, grid)

PressureFields(pressures::NamedTuple{(:pHY′, :pNHS)}, arch, grid; kwargs...) =
    validate_field_tuple_grid("pressures", pressures, grid)

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
    tracers = TracerFields(arch, grid, tracer_names; kwargs...)
    return merge(velocities, tracers)
end

#####
##### Construction utils
#####

validate_field_grid(grid, field) = grid === field.grid

validate_field_grid(grid, field_tuple::NamedTuple) =
    all(validate_field_grid(grid, field) for field in field_tuple)

"""
    validate_field_tuple_grid(tuple_name, field_tuple, arch, grid, bcs)

Validates the grids associated with grids in the (possibly nested) `field_tuple`,
and returns `field_tuple` if validation succeeds.
"""
function validate_field_tuple_grid(tuple_name, field_tuple, grid)

    all(validate_field_grid(grid, field) for field in field_tuple) ||
        throw(ArgumentError("Model grid and $tuple_name grid are not identical! " *
                            "Check that the grid used to construct $tuple_name has the correct halo size."))

    return field_tuple
end
