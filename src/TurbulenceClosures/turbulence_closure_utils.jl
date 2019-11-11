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

@inline geo_mean_Δᶠ(i, j, k, grid::RegularCartesianGrid{T}) where T = (grid.Δx * grid.Δy * grid.Δz)^T(1/3)

function calculate_nonlinear_viscosity!(νₑ, grid, closure, buoyancy, U, C)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds νₑ[i, j, k] = νᶜᶜᶜ(i, j, k, grid, closure, buoyancy, U, C)
            end
        end
    end
    return nothing
end
