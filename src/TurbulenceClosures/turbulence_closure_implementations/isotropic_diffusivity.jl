import Oceananigans.Grids: required_halo_size

struct IsotropicDiffusivity{TD, N, K} <: AbstractIsotropicDiffusivity{TD}
    ν :: N
    κ :: K

    function IsotropicDiffusivity{TD}(ν::N, κ::K) where {TD, N, K}
        return new{TD, N, K}(ν, κ)
    end
end

"""
    IsotropicDiffusivity(; ν=0, κ=0)

Returns parameters for an isotropic diffusivity model with viscosity `ν`
and thermal diffusivities `κ` for each tracer field in `tracers`
`ν` and the fields of `κ` may be constants, arrays, fields, or functions of `(x, y, z, t)`.

`κ` may be a `NamedTuple` with fields corresponding
to each tracer, or a single number to be a applied to all tracers.
"""
function IsotropicDiffusivity(FT=Float64; ν=0, κ=0, time_discretization::TD = ExplicitTimeDiscretization()) where TD
    if ν isa Number && κ isa Number
        κ = convert_diffusivity(FT, κ)
        return IsotropicDiffusivity{TD}(FT(ν), κ)
    else
        return IsotropicDiffusivity{TD}(ν, κ)
    end
end

required_halo_size(closure::IsotropicDiffusivity) = 1 
 
function with_tracers(tracers, closure::IsotropicDiffusivity{TD}) where TD
    κ = tracer_diffusivities(tracers, closure.κ)
    return IsotropicDiffusivity{TD}(closure.ν, κ)
end

calculate_diffusivities!(K, arch, grid, closure::IsotropicDiffusivity, args...) = nothing

@inline diffusivity(closure::IsotropicDiffusivity, ::Val{tracer_index}, args...) where tracer_index = closure.κ[tracer_index]
@inline viscosity(closure::IsotropicDiffusivity, args...) = closure.ν
                        
Base.show(io::IO, closure::IsotropicDiffusivity) = print(io, "IsotropicDiffusivity: ν=$(closure.ν), κ=$(closure.κ)")
    
