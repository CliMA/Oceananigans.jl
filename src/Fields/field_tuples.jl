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

"""
    TracerFields(arch, grid, tracer_names; kwargs...)

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
function TracerFields(arch, grid, names; kwargs...)
    tracer_names = tracernames(names) # filter `names` if it contains velocity fields
    tracer_fields =
        Tuple(c ∈ keys(kwargs) ?
              kwargs[c] :
              CellField(arch, grid, TracerBoundaryConditions(grid))
              for c in tracer_names)
    return NamedTuple{tracer_names}(tracer_fields)
end

"""
    TracerFields(arch, grid, tracer_names, bcs)

Return a NamedTuple with tracer fields specified by `tracer_names` initialized as
`CellField`s on the architecture `arch` and `grid`. Boundary conditions `bcs` may
be specified via a named tuple of `FieldBoundaryCondition`s.
"""
function TracerFields(arch, grid, names, bcs)
    tracer_names = tracernames(names) # filter `names` if it contains velocity fields
    tracer_fields =
        Tuple(c ∈ keys(bcs) ?
              CellField(arch, grid, bcs[c]) :
              CellField(arch, grid, TracerBoundaryConditions(grid))
              for c in tracer_names)
    return NamedTuple{tracer_names}(tracer_fields)
end

TracerFields(arch, grid, ::Union{Tuple{}, Nothing}, args...; kwargs...) = NamedTuple()
TracerFields(arch, grid, tracer::Symbol; kwargs...) = TracerFields(arch, grid, tuple(tracer); kwargs...)

"""
    TracerFields(arch, grid, tracer_fields::NamedTuple; kwargs...)

Convenience method for restoring checkpointed models that returns the already-instantiated
`tracer_fields` with non-default boundary conditions.
"""
function TracerFields(arch, grid, proposed_tracer_fields::NamedTuple, bcs; kwargs...)
    grid = proposed_tracer_fields[1].grid

    tracer_fields = 
        Tuple(c ∈ keys(bcs) ?
              Field{Cell, Cell, Cell}(proposed_tracer_fields[c].data, grid, bcs[c]) :
              Field{Cell, Cell, Cell}(proposed_tracer_fields[c].data, grid, TracerBoundaryConditions(grid))
              for c in tracernames(proposed_tracer_fields))

    return NamedTuple{tracernames(proposed_tracer_fields)}(tracer_fields)
end

"Shortcut constructor for empty tracer fields."
TracerFields(arch, grid, empty_tracer_fields::NamedTuple{(),Tuple{}}, args...; kwargs...) = NamedTuple()

"Returns true if the first three elements of `names` are `(:u, :v, :w)`."
has_velocities(names) = :u == names[1] && :v == names[2] && :w == names[3]
has_velocities(::Tuple{}) = false

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
