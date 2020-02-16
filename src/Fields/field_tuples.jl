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
    TracerFields(arch, grid, tracer_names; kwargs...)

Return a NamedTuple with tracer fields specified by `tracer_names` initialized as
`CellField`s on the architecture `arch` and `grid`. Fields may be passed via optional
keyword arguments `kwargs` for each field.
"""
function TracerFields(arch, grid, tracer_names; kwargs...)
    for c in keys(kwargs)
        c ∉ tracer_names && @warn "$c field passed but $c is not in tracer_names."
    end

    tracer_fields = Tuple(c ∈ keys(kwargs) ?
                          kwargs[c] :
                          CellField(arch, grid, TracerBoundaryConditions(grid), zeros(arch, grid))
                          for c in tracer_names)

    return NamedTuple{tracer_names}(tracer_fields)
end

TracerFields(arch, grid, ::Union{Tuple{}, Nothing}; kwargs...) = NamedTuple{()}(())
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
