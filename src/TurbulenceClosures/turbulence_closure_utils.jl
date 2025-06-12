using Oceananigans.Operators

const PossibleDiffusivity = Union{Number, Function, DiscreteDiffusionFunction, AbstractArray}

@inline tracer_diffusivities(tracer_names, κ::PossibleDiffusivity) = with_tracers(tracer_names, NamedTuple(), (tracer_names, init) -> κ)
@inline tracer_diffusivities(tracer_names, ::Nothing) = nothing

@inline function tracer_diffusivities(tracer_names, user_κ::NamedTuple)
    all(name ∈ propertynames(user_κ) for name in tracer_names) ||
        throw(ArgumentError("Tracer diffusivities or diffusivity parameters must either be a constants
                            or a `NamedTuple` with a value for every tracer!"))

    materialized_κ = NamedTuple(name => user_κ[name] for name in tracer_names)                                
    return materialized_κ
end

@inline convert_diffusivity(FT, κ::Nothing; kw...) = nothing
@inline convert_diffusivity(FT, κ::Number; kw...)  = convert(FT, κ)

@inline function convert_diffusivity(FT, κ; discrete_form=false, loc=(nothing, nothing, nothing), parameters=nothing)
    discrete_form && return DiscreteDiffusionFunction(κ; loc, parameters)
    return κ
end

@inline function convert_diffusivity(FT, κ::NamedTuple; discrete_form=false, loc=(nothing, nothing, nothing), parameters=nothing)
    κ_names = propertynames(κ)
    Nnames = length(κ_names)

    κ_values = ntuple(Val(Nnames)) do n
        Base.@_inline_meta
        κi = κ[n]
        convert_diffusivity(FT, κi; discrete_form, loc, parameters)
    end

    return NamedTuple{κ_names}(κ_values)
end

