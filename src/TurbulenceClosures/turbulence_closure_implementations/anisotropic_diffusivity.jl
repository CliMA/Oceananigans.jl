"""
    AnisotropicDiffusivity{NX, NY, NZ, KX, KY, KZ}

Parameters for anisotropic diffusivity models.
"""
struct AnisotropicDiffusivity{TD, NX, NY, NZ, KX, KY, KZ} <: AbstractTurbulenceClosure{TD}
    νx :: NX
    νy :: NY
    νz :: NZ
    κx :: KX
    κy :: KY
    κz :: KZ

    function AnisotropicDiffusivity{TD}(νx::NX, νy::NY, νz::NZ,
                                        κx::KX, κy::KY, κz::KZ) where {TD, NX, NY, NZ, KX, KY, KZ}

        return new{TD, NX, NY, NZ, KX, KY, KZ}(νx, νy, νz, κx, κy, κz)
    end
end

const AD = AnisotropicDiffusivity

"""
    AnisotropicDiffusivity(; νx=0, νy=0, νz=0, κx=0, κy=0, κz=0,
                             νh=nothing, κh=nothing)

Returns parameters for a closure with a diagonal diffusivity tensor with heterogeneous
'anisotropic' components labeled by `x`, `y`, `z`.
Each component may be a number or function.
The tracer diffusivities `κx`, `κy`, and `κz` may be `NamedTuple`s with fields corresponding
to each tracer, or a single number or function to be a applied to all tracers.

If `νh` or `κh` are provided, then `νx = νy = νh`, and `κx = κy = κh`, respectively.
"""
function AnisotropicDiffusivity(FT=Float64; νx=0, νy=0, νz=0, κx=0, κy=0, κz=0, νh=nothing, κh=nothing,
                                time_discretization::TD = ExplicitTimeDiscretization()) where TD
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
        return AnisotropicDiffusivity{TD}(FT(νx), FT(νy), FT(νz), κx, κy, κz)
    else
        return AnisotropicDiffusivity{TD}(νx, νy, νz, κx, κy, κz)
    end
end

function with_tracers(tracers, closure::AnisotropicDiffusivity{TD}) where TD
    κx = tracer_diffusivities(tracers, closure.κx)
    κy = tracer_diffusivities(tracers, closure.κy)
    κz = tracer_diffusivities(tracers, closure.κz)
    return AnisotropicDiffusivity{TD}(closure.νx, closure.νy, closure.νz, κx, κy, κz)
end

calculate_diffusivities!(K, arch, grid, closure::AnisotropicDiffusivity, args...) = nothing

#####
##### Diffusive fluxes
#####

const APG = AbstractPrimaryGrid

@inline function diffusive_flux_x(i, j, k, grid::APG, closure::AD, c, ::Val{tracer_index}, clock, args...) where tracer_index
    @inbounds κx = closure.κx[tracer_index]
    return diffusive_flux_x(i, j, k, grid, clock, κx, c)
end

@inline function diffusive_flux_y(i, j, k, grid::APG, closure::AD, c, ::Val{tracer_index}, clock, args...) where tracer_index
    @inbounds κy = closure.κy[tracer_index]
    return diffusive_flux_y(i, j, k, grid, clock, κy, c)
end

@inline function diffusive_flux_z(i, j, k, grid::APG, closure::AD, c, ::Val{tracer_index}, clock, args...) where tracer_index
    @inbounds κz = closure.κz[tracer_index]
    return diffusive_flux_z(i, j, k, grid, clock, κz, c)
end

viscous_flux_ux(i, j, k, grid::APG, closure::AD, clock, U, args...) = viscous_flux_ux(i, j, k, grid, clock, closure.νx, U[1])
viscous_flux_uy(i, j, k, grid::APG, closure::AD, clock, U, args...) = viscous_flux_uy(i, j, k, grid, clock, closure.νy, U[1])  
viscous_flux_uz(i, j, k, grid::APG, closure::AD, clock, U, args...) = viscous_flux_uz(i, j, k, grid, clock, closure.νz, U[1])

viscous_flux_vx(i, j, k, grid::APG, closure::AD, clock, U, args...) = viscous_flux_vx(i, j, k, grid, clock, closure.νx, U[2])
viscous_flux_vy(i, j, k, grid::APG, closure::AD, clock, U, args...) = viscous_flux_vy(i, j, k, grid, clock, closure.νy, U[2])  
viscous_flux_vz(i, j, k, grid::APG, closure::AD, clock, U, args...) = viscous_flux_vz(i, j, k, grid, clock, closure.νz, U[2])

viscous_flux_wx(i, j, k, grid::APG, closure::AD, clock, U, args...) = viscous_flux_wx(i, j, k, grid, clock, closure.νx, U[3])
viscous_flux_wy(i, j, k, grid::APG, closure::AD, clock, U, args...) = viscous_flux_wy(i, j, k, grid, clock, closure.νy, U[3])  
viscous_flux_wz(i, j, k, grid::APG, closure::AD, clock, U, args...) = viscous_flux_wz(i, j, k, grid, clock, closure.νz, U[3])

#####
##### Support for vertically implicit time integration
#####

const VITD = VerticallyImplicitTimeDiscretization

z_viscosity(closure::AD, args...) = closure.νz
z_diffusivity(closure::AD, ::Val{tracer_index}, args...) where tracer_index = @inbounds closure.κz[tracer_index]

const VerticallyBoundedGrid{FT} = AbstractPrimaryGrid{FT, <:Any, <:Any, <:Bounded}

@inline diffusive_flux_z(i, j, k, grid::APG{FT}, ::VITD, closure::AD, args...) where FT = zero(FT)
@inline viscous_flux_uz(i, j, k, grid::APG{FT}, ::VITD, closure::AD, args...) where FT = zero(FT)
@inline viscous_flux_vz(i, j, k, grid::APG{FT}, ::VITD, closure::AD, args...) where FT = zero(FT)

@inline function diffusive_flux_z(i, j, k, grid::VerticallyBoundedGrid{FT}, ::VITD, closure::AD, args...) where FT
    return ifelse(k == 1 || k == grid.Nz+1, 
                  diffusive_flux_z(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...), # on boundaries, calculate fluxes explicitly
                  zero(FT))
end

@inline function viscous_flux_uz(i, j, k, grid::VerticallyBoundedGrid{FT}, ::VITD, closure::AD, args...) where FT
    return ifelse(k == 1 || k == grid.Nz+1, 
                  viscous_flux_vz(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...), # on boundaries, calculate fluxes explicitly
                  zero(FT))
end

@inline function viscous_flux_vz(i, j, k, grid::VerticallyBoundedGrid{FT}, ::VITD, closure::AD, args...) where FT
    return ifelse(k == 1 || k == grid.Nz+1, 
                  viscous_flux_uz(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...), # on boundaries, calculate fluxes explicitly
                  zero(FT))
end

#####
##### Show
#####

Base.show(io::IO, closure::AnisotropicDiffusivity) =
    print(io, "AnisotropicDiffusivity: " *
              "(νx=$(closure.νx), νy=$(closure.νy), νz=$(closure.νz)), " *
              "(κx=$(closure.κx), κy=$(closure.κy), κz=$(closure.κz))")
