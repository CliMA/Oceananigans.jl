import Oceananigans.Grids: required_halo_size

struct IsotropicDiffusivity{N, K} <: AbstractIsotropicDiffusivity{ExplicitTimeDiscretization}
    ν :: N
    κ :: K
end

"""
    IsotropicDiffusivity(; ν=0, κ=0)

Returns parameters for an isotropic diffusivity model with viscosity `ν`
and thermal diffusivities `κ` for each tracer field in `tracers`
`ν` and the fields of `κ` may be constants, arrays, fields, or functions of `(x, y, z, t)`.

`κ` may be a `NamedTuple` with fields corresponding
to each tracer, or a single number to be a applied to all tracers.
"""
function IsotropicDiffusivity(FT=Float64; ν=0, κ=0)
    if ν isa Number && κ isa Number
        κ = convert_diffusivity(FT, κ)
        return IsotropicDiffusivity(FT(ν), κ)
    else
        return IsotropicDiffusivity(ν, κ)
    end
end

required_halo_size(closure::IsotropicDiffusivity) = 1 
 
function with_tracers(tracers, closure::IsotropicDiffusivity)
    κ = tracer_diffusivities(tracers, closure.κ)
    return IsotropicDiffusivity(closure.ν, κ)
end

calculate_diffusivities!(K, arch, grid, closure::IsotropicDiffusivity, args...) = nothing

@inline diffusivity(closure::IsotropicDiffusivity, diffusivities, ::Val{tracer_index}) where tracer_index = closure.κ[tracer_index]
@inline viscosity(closure::IsotropicDiffusivity, diffusivities) = closure.ν
                        
Base.show(io::IO, closure::IsotropicDiffusivity) = print(io, "IsotropicDiffusivity: ν=$(closure.ν), κ=$(closure.κ)")
    
