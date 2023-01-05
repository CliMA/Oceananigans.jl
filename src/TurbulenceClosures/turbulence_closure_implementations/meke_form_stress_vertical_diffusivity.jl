import Oceananigans.Grids: required_halo_size
using Oceananigans.Utils: prettysummary

struct MEWSVerticalDiffusivity{TD, FT} <: AbstractScalarDiffusivity{TD, VerticalFormulation}
    Cʰ :: FT
    Cᴷ :: FT
    Cⁿ :: FT

    function MEWSVerticalDiffusivity{TD}(Cʰ::FT, Cᴷ::FT, Cⁿ::FT) where {TD, FT}
        return new{TD, FT}(Cʰ, Cᴷ, Cⁿ)
    end
end

function MEWSVerticalDiffusivity(time_discretization=VerticallyImplicitTimeDiscretization(), FT=Float64;
                                 Cʰ=1, Cᴷ=1, Cⁿ=1)
                           
    TD = typeof(time_discretization)

    return MEWSVerticalDiffusivity{TD}(FT(Cʰ), FT(Cᴷ), FT(Cⁿ))
end

required_halo_size(closure::MEWSVerticalDiffusivity) = 1 
with_tracers(tracers, closure::MEWSVerticalDiffusivity) = closure

@inline viscosity_location(::MEWSVerticalDiffusivity) = (Center(), Center(), Face())
@inline diffusivity_location(::MEWSVerticalDiffusivity) = (Center(), Center(), Face())

function DiffusivityFields(grid, tracer_names, bcs, closure::MEWSVerticalDiffusivity)
    νₑ = Field{Center, Center, Face}(grid)

    # Mesoscale kinetic energy
    K = Field{Center, Center, Nothing}(grid)
    return (; νₑ, K)
end        

@inline viscosity(closure::MEWSVerticalDiffusivity, K) = K.νₑ
@inline diffusivity(closure::MEWSVerticalDiffusivity, K, ::Val{id}) where id = 0

calculate_diffusivities!(diffusivities, ::MEWSVerticalDiffusivity, args...) = nothing

Base.summary(closure::MEWSVerticalDiffusivity) = "MEWSVerticalDiffusivity"
Base.show(io::IO, closure::ScalarDiffusivity) = print(io, summary(closure))

