using Oceananigans.Operators

const PossibleDiffusivity = Union{Number, Function, DiscreteDiffusionFunction, AbstractArray}

@inline tracer_diffusivities(tracers, κ::PossibleDiffusivity) = with_tracers(tracers, NamedTuple(), (tracers, init) -> κ)
@inline tracer_diffusivities(tracers, ::Nothing) = nothing

@inline function tracer_diffusivities(tracers, κ::NamedTuple)

    all(name ∈ propertynames(κ) for name in tracers) ||
        throw(ArgumentError("Tracer diffusivities or diffusivity parameters must either be a constants
                            or a `NamedTuple` with a value for every tracer!"))

    return κ
end

convert_diffusivity(FT, κ::Number; kw...) = convert(FT, κ)

function convert_diffusivity(FT, κ; discrete_form=false, loc=(nothing, nothing, nothing), parameters=nothing)
    discrete_form && return DiscreteDiffusionFunction(κ; loc, parameters)
    return κ
end
    
function convert_diffusivity(FT, κ::NamedTuple; discrete_form=false, loc=(nothing, nothing, nothing), parameters=nothing)
    κ_names = propertynames(κ)
    return NamedTuple{κ_names}(Tuple(convert_diffusivity(FT, κi; discrete_form, loc, parameters) for κi in κ))
end
