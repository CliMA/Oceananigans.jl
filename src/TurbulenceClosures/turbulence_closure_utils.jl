tracer_diffusivities(tracers, κ::Union{Number, Function}) = with_tracers(tracers, NamedTuple(), (tracers, init) -> κ)

function tracer_diffusivities(tracers, κ::NamedTuple)

    all(name ∈ propertynames(κ) for name in tracers) ||
        throw(ArgumentError("Tracer diffusivities or diffusivity parameters must either be a constants
                            or a `NamedTuple` with a value for every tracer!"))

    return κ
end

convert_diffusivity(T, κ::Number) = convert(T, κ)
convert_diffusivity(T, κ::NamedTuple) = convert(NamedTuple{propertynames(κ), NTuple{length(κ), T}}, κ)

@inline geo_mean_Δᶠ(i, j, k, grid::AbstractGrid{FT}) where FT =
    (Δx(i, j, k, grid) * Δy(i, j, k, grid) * Δz(i, j, k, grid))^FT(1/3)

@kernel function calculate_nonlinear_viscosity!(νₑ, grid, closure, buoyancy, U, C)
    i, j, k = @index(Global, NTuple)
    @inbounds νₑ[i, j, k] = νᶜᶜᶜ(i, j, k, grid, closure, buoyancy, U, C)
end
