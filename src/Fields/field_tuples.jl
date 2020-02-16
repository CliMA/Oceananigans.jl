"""
    VelocityFields(arch, grid; u=zeros(arch, grid), v=zeros(arch, grid), w=zeros(arch, grid))

Return a NamedTuple with fields `u`, `v`, `w` initialized on the architecture `arch`
and `grid`. Data arrays may be passed via the `u`, `v`, and `w` keyword arguments.
"""
function VelocityFields(arch, grid; u=zeros(arch, grid), v=zeros(arch, grid), w=zeros(arch, grid))
    u = XFaceField(arch, grid, u)
    v = YFaceField(arch, grid, v)
    w = ZFaceField(arch, grid, w)
    return (u=u, v=v, w=w)
end

"""
    TracerFields(arch, grid, tracer_names; kwargs...)

Return a NamedTuple with tracer fields specified by `tracer_names` initialized as
`CellField`s on the architecture `arch` and `grid`. Optional `kwargs` can be specified
to assign data arrays to each tracer field.
"""
function TracerFields(arch, grid, tracer_names; kwargs...)
    tracer_fields = Tuple(c ∈ keys(kwargs) ? CellField(arch, grid, kwargs[c]) : CellField(arch, grid)
                          for c in tracer_names)
    return NamedTuple{tracer_names}(tracer_fields)
end

TracerFields(arch, grid, ::Union{Tuple{}, Nothing}) = NamedTuple{()}(())
TracerFields(arch, grid, tracer::Symbol; kwargs...) = TracerFields(arch, grid, tuple(tracer); kwargs...)
TracerFields(arch, grid, tracers::NamedTuple; kwargs...) = tracers

tracernames(::Nothing) = ()
tracernames(name::Symbol) = tuple(name)
tracernames(names::NTuple{N, Symbol}) where N = :u ∈ names ? names[4:end] : names
tracernames(::NamedTuple{names}) where names = tracernames(names)

"""
    PressureFields(arch, grid; pHY′=zeros(arch, grid), pNHS=zeros(arch, grid))

Return a NamedTuple with pressure fields `pHY′` and `pNHS` initialized as
`CellField`s on the architecture `arch` and `grid`. Data arrays may be passed via
the ``pHY′` and `pNHS` keyword arguments.
"""
function PressureFields(arch, grid; pHY′=zeros(arch, grid), pNHS=zeros(arch, grid))
    pHY′ = CellField(arch, grid, pHY′)
    pNHS = CellField(arch, grid, pNHS)
    return (pHY′=pHY′, pNHS=pNHS)
end

"""
    Tendencies(arch, grid, tracer_names; kwargs...)

Return a NamedTuple with tendencies for all solution fields (velocity fields and
tracer fields), initialized on the architecture `arch` and `grid`. Optional `kwargs`
can be specified to assign data arrays to each tendency field.
"""
function Tendencies(arch, grid, tracer_names; kwargs...)
    velocities = (
        u = :u ∈ keys(kwargs) ? XFaceField(arch, grid, kwargs[:u]) : XFaceField(arch, grid),
        v = :v ∈ keys(kwargs) ? YFaceField(arch, grid, kwargs[:v]) : YFaceField(arch, grid),
        w = :w ∈ keys(kwargs) ? ZFaceField(arch, grid, kwargs[:w]) : ZFaceField(arch, grid)
    )
    tracers = TracerFields(arch, grid, tracer_names; kwargs...)
    return merge(velocities, tracers)
end
