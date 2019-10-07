Base.eltype(::TurbulenceClosure{T}) where T = T

"""
    typed_keyword_constructor(T, Closure; kwargs...)

Return an object `Closure` with fields provided by `kwargs`
converted to type `T`. Mainly provided for converting between
different float types when working with constructors associated
with types defined via `Base.@kwdef`.
"""
function typed_keyword_constructor(T, Closure; kwargs...)
    closure = Closure(; kwargs...)
    names = fieldnames(Closure)
    vals = [getproperty(closure, name) for name in names]
    return Closure{T}(vals...)
end

function Base.convert(::TurbulenceClosure{T2}, closure::TurbulenceClosure{T1}) where {T1, T2}
    paramdict = Dict((p, convert(T2, getproperty(closure, p))) for p in propertynames(closure))
    Closure = typeof(closure).name.wrapper
    return Closure(T2; paramdict...)
end

tracer_diffusivities(tracers, κ::Number) = with_tracers(tracers, (), κ)

function tracer_diffusivities(tracers, κ::NamedTuple)

    all(name ∈ propertynames(κ) for name in tracers) || 
        throw(ArgumentError("Tracer diffusivities or diffusivity parameters must either be a constants 
                            or a `NamedTuple` with a value for every tracer!"))

    return κ
end

convert_diffusivity(T, κ::Number) = convert(T, κ)
convert_diffusivity(T, κ::NamedTuple) = convert(propertynames(κ), NTuple{length(κ), T}, κ)
