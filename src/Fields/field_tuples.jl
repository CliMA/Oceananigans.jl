"""
    VelocityFields(arch, grid; [u, v, w])

Return a NamedTuple with fields `u`, `v`, `w` initialized on the architecture `arch`
and `grid`. Fields may be passed via the optional keyword arguments `u`, `v`, and `w`.
"""
VelocityFields(arch, grid;
    u = XFaceField(arch, grid, UVelocityBoundaryConditions(grid), zeros(arch, grid)),
    v = YFaceField(arch, grid, VVelocityBoundaryConditions(grid), zeros(arch, grid)),
    w = ZFaceField(arch, grid, WVelocityBoundaryConditions(grid), zeros(arch, grid))
) = (u=u, v=v, w=w)

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

    u = XFaceField(arch, grid, u_bcs, zeros(arch, grid))
    v = YFaceField(arch, grid, v_bcs, zeros(arch, grid))
    w = ZFaceField(arch, grid, w_bcs, zeros(arch, grid))

    return (u=u, v=v, w=w)
end

"""
    TracerFields(arch, grid, tracer_names; kwargs...)

Return a NamedTuple with tracer fields specified by `tracer_names` initialized as
`CellField`s on the architecture `arch` and `grid`. Fields may be passed via optional
keyword arguments `kwargs` for each field.
"""
function TracerFields(arch, grid, tracer_names; kwargs...)
    tracer_fields =
        Tuple(c ∈ keys(kwargs) ?
              kwargs[c] :
              CellField(arch, grid, TracerBoundaryConditions(grid), zeros(arch, grid))
              for c in tracer_names)

    return NamedTuple{tracer_names}(tracer_fields)
end

"""
    TracerFields(arch, grid, tracer_names, bcs::NamedTuple)

Return a NamedTuple with tracer fields specified by `tracer_names` initialized as
`CellField`s on the architecture `arch` and `grid`. Boundary conditions `bcs` may
be specified via a named tuple of `FieldBoundaryCondition`s.
"""
function TracerFields(arch, grid, tracer_names, bcs::NamedTuple)
    tracer_fields =
        Tuple(c ∈ keys(bcs) ?
              CellField(arch, grid, bcs[c],                         zeros(arch, grid)) :
              CellField(arch, grid, TracerBoundaryConditions(grid), zeros(arch, grid))
              for c in tracer_names)

    return NamedTuple{tracer_names}(tracer_fields)
end

TracerFields(arch, grid, ::Union{Tuple{}, Nothing}; kwargs...) = NamedTuple()
TracerFields(arch, grid, tracer::Symbol; kwargs...) = TracerFields(arch, grid, tuple(tracer); kwargs...)
TracerFields(arch, grid, tracers::NamedTuple; kwargs...) = tracers

tracernames(::Nothing) = ()
tracernames(name::Symbol) = tuple(name)
tracernames(names::NTuple{N, Symbol}) where N = :u ∈ names ? names[4:end] : names
tracernames(::NamedTuple{names}) where names = tracernames(names)

"""
    PressureFields(arch, grid; [pHY′, pNHS])

Return a NamedTuple with pressure fields `pHY′` and `pNHS` initialized as
`CellField`s on the architecture `arch` and `grid`. Fields may be passed via the
optional keyword arguments `pHY′` and `pNHS`.
"""
PressureFields(arch, grid;
    pHY′ = CellField(arch, grid, PressureBoundaryConditions(grid), zeros(arch, grid)),
    pNHS = CellField(arch, grid, PressureBoundaryConditions(grid), zeros(arch, grid))
) = (pHY′=pHY′, pNHS=pNHS)

"""
    PressureFields(arch, grid, bcs::NamedTuple)

Return a NamedTuple with pressure fields `pHY′` and `pNHS` initialized as
`CellField`s on the architecture `arch` and `grid`.  Boundary conditions `bcs` may
be specified via a named tuple of `FieldBoundaryCondition`s.
"""
function PressureFields(arch, grid, bcs::NamedTuple)
    pHY′_bcs = :pHY′ ∈ keys(bcs) ? bcs[:pHY′] : PressureBoundaryConditions(grid)
    pNHS_bcs = :pNHS ∈ keys(bcs) ? bcs[:pNHS] : PressureBoundaryConditions(grid)

    pHY′ = CellField(arch, grid, pHY′_bcs, zeros(arch, grid))
    pNHS = CellField(arch, grid, pNHS_bcs, zeros(arch, grid))

    return (pHY′=pHY′, pNHS=pNHS)
end

"""
    TendencyFields(arch, grid, tracer_names; kwargs...)

Return a NamedTuple with tendencies for all solution fields (velocity fields and
tracer fields), initialized on the architecture `arch` and `grid`. Optional `kwargs`
can be specified to assign data arrays to each tendency field.
"""
function TendencyFields(arch, grid, tracer_names; kwargs...)
    velocities = (
        u = :u ∈ keys(kwargs) ? kwargs[:u] : XFaceField(arch, grid, UVelocityBoundaryConditions(grid), zeros(arch, grid)),
        v = :v ∈ keys(kwargs) ? kwargs[:v] : YFaceField(arch, grid, VVelocityBoundaryConditions(grid), zeros(arch, grid)),
        w = :w ∈ keys(kwargs) ? kwargs[:w] : ZFaceField(arch, grid, WVelocityBoundaryConditions(grid), zeros(arch, grid))
    )
    tracers = TracerFields(arch, grid, tracer_names; kwargs...)
    return merge(velocities, tracers)
end
