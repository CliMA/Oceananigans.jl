Base.eltype(::TurbulenceClosure{T}) where T = T

function Base.convert(::TurbulenceClosure{T2}, closure::TurbulenceClosure{T1}) where {T1, T2}
    paramdict = Dict((p, convert(T2, getproperty(closure, p))) for p in propertynames(closure))
    Closure = typeof(closure).name.wrapper
    return Closure(T2; paramdict...)
end

tracer_diffusivities(tracers, κ::Number) = with_tracers(tracers, NamedTuple(), (tracers, init) -> κ)

function tracer_diffusivities(tracers, κ::NamedTuple)

    all(name ∈ propertynames(κ) for name in tracers) ||
        throw(ArgumentError("Tracer diffusivities or diffusivity parameters must either be a constants
                            or a `NamedTuple` with a value for every tracer!"))

    return κ
end

convert_diffusivity(T, κ::Number) = convert(T, κ)
convert_diffusivity(T, κ::NamedTuple) = convert(NamedTuple{propertynames(κ), NTuple{length(κ), T}}, κ)

@inline geo_mean_Δᶠ(i, j, k, grid::AbstractGrid{T}) where T =
    (Δx(i, j, k, grid) * Δy(i, j, k, grid) * ΔzC(i, j, k, grid))^T(1/3)

function calculate_nonlinear_viscosity!(νₑ, grid, closure, buoyancy, U, C)
    @loop_xyz i j k grid begin
        @inbounds νₑ[i, j, k] = νᶜᶜᶜ(i, j, k, grid, closure, buoyancy, U, C)
    end
    return nothing
end
