float_type(m::AbstractModel) = eltype(model.grid)

"""
    with_tracers(tracers, initial_tuple, tracer_default)

Create a tuple corresponding to the solution variables `u`, `v`, `w`, 
and `tracers`. `initial_tuple` is a `NamedTuple` that at least has
fields `u`, `v`, and `w`, and may have some fields corresponding to
the names in `tracers`. `tracer_default` is a function that produces
a default tuple value for each tracer if not included in `initial_tuple`.
"""
function with_tracers(tracers, initial_tuple, tracer_default; with_velocities=false)
    solution_values = [] # Array{Any, 1}
    solution_names = []

    if with_velocities
        push!(solution_values, initial_tuple.u)
        push!(solution_values, initial_tuple.v)
        push!(solution_values, initial_tuple.w)

        append!(solution_names, [:u, :v, :w])
    end

    for name in tracers
        tracer_elem = name ∈ propertynames(initial_tuple) ?
                        getproperty(initial_tuple, name) :
                        tracer_default(tracers, initial_tuple)

        push!(solution_values, tracer_elem)
    end

    append!(solution_names, tracers)

    return NamedTuple{Tuple(solution_names)}(Tuple(solution_values))
end

"""
    ModelForcing(; kwargs...)

Return a named tuple of forcing functions for each solution field.
"""
ModelForcing(; u=zerofunk, v=zerofunk, w=zerofunk, tracer_forcings...) =
    merge((u=u, v=v, w=w), tracer_forcings)

const Forcing = ModelForcing

default_tracer_forcing(args...) = zerofunk
ModelForcing(tracers, proposal_forcing) = with_tracers(tracers, proposal_forcing, default_tracer_forcing, 
                                                       with_velocities=true)

"""
    VelocityFields(arch, grid)

Return a NamedTuple with fields `u`, `v`, `w` initialized on
the architecture `arch` and `grid`.
"""
function VelocityFields(arch, grid)
    u = FaceFieldX(arch, grid)
    v = FaceFieldY(arch, grid)
    w = FaceFieldZ(arch, grid)
    return (u=u, v=v, w=w)
end

"""
    TracerFields(arch, grid)

Return a NamedTuple with tracer fields initialized
as `CellField`s on the architecture `arch` and `grid`.
"""
function TracerFields(arch, grid, tracernames)
    tracerfields = Tuple(CellField(arch, grid) for c in tracernames)
    return NamedTuple{tracernames}(tracerfields)
end

TracerFields(arch, grid, ::Union{Tuple{}, Nothing}) = NamedTuple{()}(())
TracerFields(arch, grid, tracer::Symbol) = TracerFields(arch, grid, tuple(tracer))
TracerFields(arch, grid, tracers::NamedTuple) = tracers

tracernames(::Nothing) = ()
tracernames(name::Symbol) = tuple(name)
tracernames(names::NTuple{N, Symbol}) where N = :u ∈ names ? names[4:end] : names
tracernames(::NamedTuple{names}) where names = tracernames(names)

"""
    PressureFields(arch, grid)

Return a NamedTuple with pressure fields `pHY′` and `pNHS`
initialized as `CellField`s on the architecture `arch` and `grid`.
"""
function PressureFields(arch, grid)
    pHY′ = CellField(arch, grid)
    pNHS = CellField(arch, grid)
    return (pHY′=pHY′, pNHS=pNHS)
end

"""
    Tendencies(arch, grid, tracernames)

Return a NamedTuple with tendencies for all solution fields
(velocity fields and tracer fields), initialized on
the architecture `arch` and `grid`.
"""
function Tendencies(arch, grid, tracernames)

    velocities = (u = FaceFieldX(arch, grid),
                  v = FaceFieldY(arch, grid),
                  w = FaceFieldZ(arch, grid))

    tracers = TracerFields(arch, grid, tracernames)

    return merge(velocities, tracers)
end

"""
    AdamsBashforthTimestepper(float_type, arch, grid, tracers, χ)

Return an AdamsBashforthTimestepper object with tendency
fields on `arch` and `grid` and AB2 parameter `χ`.
"""
struct AdamsBashforthTimestepper{T, TG}
      Gⁿ :: TG
      G⁻ :: TG
       χ :: T
end

function AdamsBashforthTimestepper(float_type, arch, grid, tracers, χ)
   Gⁿ = Tendencies(arch, grid, tracers)
   G⁻ = Tendencies(arch, grid, tracers)
   return AdamsBashforthTimestepper{float_type, typeof(Gⁿ)}(Gⁿ, G⁻, χ)
end
