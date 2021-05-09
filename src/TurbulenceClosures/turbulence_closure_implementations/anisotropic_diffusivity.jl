"""
    AnisotropicDiffusivity{NX, NY, NZ, KX, KY, KZ}

Parameters for anisotropic diffusivity models.
"""
struct AnisotropicDiffusivity{NX, NY, NZ, KX, KY, KZ} <: AbstractTurbulenceClosure{ExplicitTimeDiscretization}
    νx :: NX
    νy :: NY
    νz :: NZ
    κx :: KX
    κy :: KY
    κz :: KZ
end

"""
    AnisotropicDiffusivity(; νx=ν₀, νy=ν₀, νz=ν₀, κx=κ₀, κy=κ₀, κz=κ₀,
                             νh=nothing, κh=nothing)

Returns parameters for a closure with a diagonal diffusivity tensor with heterogeneous
'anisotropic' components labeled by `x`, `y`, `z`.
Each component may be a number or function.
The tracer diffusivities `κx`, `κy`, and `κz` may be `NamedTuple`s with fields corresponding
to each tracer, or a single number or function to be a applied to all tracers.

If `νh` or `κh` are provided, then `νx = νy = νh`, and `κx = κy = κh`, respectively.

By default, a viscosity of `ν₀ = 1.05×10⁻⁶` m² s⁻¹ is used for all viscosity components
and a diffusivity of `κ₀ = 1.46×10⁻⁷` m² s⁻¹ is used for all diffusivity components for every tracer.
These values are the approximate viscosity and thermal diffusivity for seawater at 20°C
and 35 psu, according to Sharqawy et al., "Thermophysical properties of seawater: A review
of existing correlations and data" (2010).
"""
function AnisotropicDiffusivity(FT=Float64; νx=ν₀, νy=ν₀, νz=ν₀, κx=κ₀, κy=κ₀, κz=κ₀, νh=nothing, κh=nothing)
    if !isnothing(νh)
        νx = νh
        νy = νh
    end

    if !isnothing(κh)
        κx = κh
        κy = κh
    end

    if all(isa.((νx, νy, νz, κx, κy, κz), Number))
        κx = convert_diffusivity(FT, κx)
        κy = convert_diffusivity(FT, κy)
        κz = convert_diffusivity(FT, κz)
        return AnisotropicDiffusivity(FT(νx), FT(νy), FT(νz), κx, κy, κz)
    else
        return AnisotropicDiffusivity(νx, νy, νz, κx, κy, κz)
    end
end

function with_tracers(tracers, closure::AnisotropicDiffusivity)
    κx = tracer_diffusivities(tracers, closure.κx)
    κy = tracer_diffusivities(tracers, closure.κy)
    κz = tracer_diffusivities(tracers, closure.κz)
    return AnisotropicDiffusivity(closure.νx, closure.νy, closure.νz, κx, κy, κz)
end

calculate_diffusivities!(K, arch, grid, closure::AnisotropicDiffusivity, args...) = nothing

@inline function diffusive_flux_x(i, j, k, grid, clock, closure::AnisotropicDiffusivity, c, ::Val{tracer_index}, args...) where tracer_index
    @inbounds κx = closure.κx[tracer_index]
    return diffusive_flux_x(i, j, k, grid, clock, κx, c)
end

@inline function diffusive_flux_y(i, j, k, grid, clock, closure::AnisotropicDiffusivity, c, ::Val{tracer_index}, args...) where tracer_index
    @inbounds κy = closure.κy[tracer_index]
    return diffusive_flux_y(i, j, k, grid, clock, κy, c)
end

@inline function diffusive_flux_z(i, j, k, grid, clock, closure::AnisotropicDiffusivity, c, ::Val{tracer_index}, args...) where tracer_index
    @inbounds κz = closure.κz[tracer_index]
    return diffusive_flux_z(i, j, k, grid, clock, κz, c)
end

viscous_flux_ux(i, j, k, grid, clock, closure::AnisotropicDiffusivity, U, args...) = viscous_flux_ux(i, j, k, grid, clock, closure.νx, U[1])
viscous_flux_uy(i, j, k, grid, clock, closure::AnisotropicDiffusivity, U, args...) = viscous_flux_uy(i, j, k, grid, clock, closure.νy, U[1])  
viscous_flux_uz(i, j, k, grid, clock, closure::AnisotropicDiffusivity, U, args...) = viscous_flux_uz(i, j, k, grid, clock, closure.νz, U[1])

viscous_flux_vx(i, j, k, grid, clock, closure::AnisotropicDiffusivity, U, args...) = viscous_flux_vx(i, j, k, grid, clock, closure.νx, U[2])
viscous_flux_vy(i, j, k, grid, clock, closure::AnisotropicDiffusivity, U, args...) = viscous_flux_vy(i, j, k, grid, clock, closure.νy, U[2])  
viscous_flux_vz(i, j, k, grid, clock, closure::AnisotropicDiffusivity, U, args...) = viscous_flux_vz(i, j, k, grid, clock, closure.νz, U[2])

viscous_flux_wx(i, j, k, grid, clock, closure::AnisotropicDiffusivity, U, args...) = viscous_flux_wx(i, j, k, grid, clock, closure.νx, U[3])
viscous_flux_wy(i, j, k, grid, clock, closure::AnisotropicDiffusivity, U, args...) = viscous_flux_wy(i, j, k, grid, clock, closure.νy, U[3])  
viscous_flux_wz(i, j, k, grid, clock, closure::AnisotropicDiffusivity, U, args...) = viscous_flux_wz(i, j, k, grid, clock, closure.νz, U[3])
                        
Base.show(io::IO, closure::AnisotropicDiffusivity) =
    print(io, "AnisotropicDiffusivity: " *
              "(νx=$(closure.νx), νy=$(closure.νy), νz=$(closure.νz)), " *
              "(κx=$(closure.κx), κy=$(closure.κy), κz=$(closure.κz))")
