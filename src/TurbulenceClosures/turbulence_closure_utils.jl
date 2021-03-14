tracer_diffusivities(tracers, κ::Union{Number, Function}) = with_tracers(tracers, NamedTuple(), (tracers, init) -> κ)

function tracer_diffusivities(tracers, κ::NamedTuple)

    all(name ∈ propertynames(κ) for name in tracers) ||
        throw(ArgumentError("Tracer diffusivities or diffusivity parameters must either be a constants
                            or a `NamedTuple` with a value for every tracer!"))

    return κ
end

convert_diffusivity(FT, κ) = κ # fallback

convert_diffusivity(FT, κ::Number) = convert(FT, κ)

function convert_diffusivity(FT, κ::NamedTuple)
    κ_names = propertynames(κ)
    return NamedTuple{κ_names}(Tuple(convert_diffusivity(FT, κi) for κi in κ))
end

@inline geo_mean_Δᶠ(i, j, k, grid::AbstractGrid{T}) where T =
    (Oceananigans.Operators.Δx(i, j, k, grid) * Oceananigans.Operators.Δy(i, j, k, grid) * Oceananigans.Operators.ΔzC(i, j, k, grid))^T(1/3)

@kernel function calculate_nonlinear_viscosity!(νₑ, grid, closure, buoyancy, U, C)
    i, j, k = @index(Global, NTuple)
    @inbounds νₑ[i, j, k] = νᶜᶜᶜ(i, j, k, grid, closure, buoyancy, U, C)
end
