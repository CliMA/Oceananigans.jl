using Oceananigans.Operators

tracer_diffusivities(tracers, κ::Union{Number, Function}) = with_tracers(tracers, NamedTuple(), (tracers, init) -> κ)
tracer_diffusivities(tracers, ::Nothing) = nothing

function tracer_diffusivities(tracers, κ::NamedTuple)

    all(name ∈ propertynames(κ) for name in tracers) ||
        throw(ArgumentError("Tracer diffusivities or diffusivity parameters must either be a constants
                            or a `NamedTuple` with a value for every tracer!"))

    return κ
end

convert_diffusivity(FT, κ, discrete) = κ # fallback
convert_diffusivity(FT, κ::Number, discrete) = convert(FT, κ)
convert_diffusivity(FT, κ::F, ::Val{true})  where F<:Function = DiscreteFunction(κ) 
convert_diffusivity(FT, κ::F, ::Val{false}) where F<:Function = κ

function convert_diffusivity(FT, κ::NamedTuple, discrete)
    κ_names = propertynames(κ)
    return NamedTuple{κ_names}(Tuple(convert_diffusivity(FT, κi, discrete) for κi in κ))
end

@inline geo_mean_Δᶠ(i, j, k, grid::AbstractGrid) =
    cbrt(Δxᶜᶜᶜ(i, j, k, grid) * Δyᶜᶜᶜ(i, j, k, grid) * Δzᶜᶜᶜ(i, j, k, grid))

@kernel function calculate_nonlinear_viscosity!(νₑ, grid, closure, buoyancy, U, C)
    i, j, k = @index(Global, NTuple)
    @inbounds νₑ[i, j, k] = νᶜᶜᶜ(i, j, k, grid, closure, buoyancy, U, C)
end
