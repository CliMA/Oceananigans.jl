using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure, AbstractTimeDiscretization

const ATC = AbstractTurbulenceClosure
const ATD = AbstractTimeDiscretization

struct GridFittedBoundary{S} <: AbstractImmersedBoundary
    solid :: S
end

@inline (gfib::GridFittedBoundary)(x, y, z) = gfib.solid(x, y, z)

const IBG = ImmersedBoundaryGrid

const c = Center()
const f = Face()

#####
##### ImmersedBoundaryGrid
#####

@inline solid_cell(i, j, k, ibg) = ibg.immersed_boundary(node(c, c, c, i, j, k, ibg.grid)...)

@inline solid_node(LX, LY, LZ, i, j, k, ibg) = solid_cell(i, j, k, ibg) # fallback (for Center or Nothing LX, LY, LZ)

@inline solid_node(::Face, LX, LY, i, j, k, ibg) = solid_cell(i, j, k, ibg) || solid_cell(i-1, j, k, ibg)
@inline solid_node(LX, ::Face, LZ, i, j, k, ibg) = solid_cell(i, j, k, ibg) || solid_cell(i, j-1, k, ibg)
@inline solid_node(LX, LY, ::Face, i, j, k, ibg) = solid_cell(i, j, k, ibg) || solid_cell(i, j, k-1, ibg)

@inline solid_node(::Face, ::Face, LZ, i, j, k, ibg) = solid_node(c, f, c, i, j, k, ibg) || solid_node(c, f, c, i-1, j, k, ibg)
@inline solid_node(::Face, LY, ::Face, i, j, k, ibg) = solid_node(c, c, f, i, j, k, ibg) || solid_node(c, c, f, i-1, j, k, ibg)
@inline solid_node(LX, ::Face, ::Face, i, j, k, ibg) = solid_node(c, f, c, i, j, k, ibg) || solid_node(c, f, c, i, j, k-1, ibg)

@inline solid_node(::Face, ::Face, ::Face, i, j, k, ibg) = solid_node(c, f, f, i, j, k, ibg) || solid_node(c, f, f, i-1, j, k, ibg)

@inline conditional_flux_ccc(i, j, k, ibg::IBG{FT}, grid, flux, args...) where FT = ifelse(solid_node(c, c, c, i, j, k, ibg), zero(FT), flux(i, j, k, grid, args...))
@inline conditional_flux_ffc(i, j, k, ibg::IBG{FT}, grid, flux, args...) where FT = ifelse(solid_node(f, f, c, i, j, k, ibg), zero(FT), flux(i, j, k, grid, args...))
@inline conditional_flux_fcf(i, j, k, ibg::IBG{FT}, grid, flux, args...) where FT = ifelse(solid_node(f, c, f, i, j, k, ibg), zero(FT), flux(i, j, k, grid, args...))
@inline conditional_flux_cff(i, j, k, ibg::IBG{FT}, grid, flux, args...) where FT = ifelse(solid_node(c, f, f, i, j, k, ibg), zero(FT), flux(i, j, k, grid, args...))

@inline conditional_flux_fcc(i, j, k, ibg::IBG{FT}, grid, flux, args...) where FT = ifelse(solid_node(f, c, c, i, j, k, ibg), zero(FT), flux(i, j, k, grid, args...))
@inline conditional_flux_cfc(i, j, k, ibg::IBG{FT}, grid, flux, args...) where FT = ifelse(solid_node(c, f, c, i, j, k, ibg), zero(FT), flux(i, j, k, grid, args...))
@inline conditional_flux_ccf(i, j, k, ibg::IBG{FT}, grid, flux, args...) where FT = ifelse(solid_node(c, c, f, i, j, k, ibg), zero(FT), flux(i, j, k, grid, args...))

#####
##### Advective fluxes
#####

# ccc, ffc, fcf
 @inline viscous_flux_ux(i, j, k, ibg::IBG, disc::ATD, clo::ATC, args...) = conditional_flux_ccc(i, j, k, ibg, ibg.grid, viscous_flux_ux, disc, clo, args...)
 @inline viscous_flux_uy(i, j, k, ibg::IBG, disc::ATD, clo::ATC, args...) = conditional_flux_ffc(i, j, k, ibg, ibg.grid, viscous_flux_uy, disc, clo, args...)
 @inline viscous_flux_uz(i, j, k, ibg::IBG, disc::ATD, clo::ATC, args...) = conditional_flux_fcf(i, j, k, ibg, ibg.grid, viscous_flux_uz, disc, clo, args...)
 
 # ffc, ccc, cff
 @inline viscous_flux_vx(i, j, k, ibg::IBG, disc::ATD, clo::ATC, args...) = conditional_flux_ffc(i, j, k, ibg, ibg.grid, viscous_flux_vx, disc, clo, args...)
 @inline viscous_flux_vy(i, j, k, ibg::IBG, disc::ATD, clo::ATC, args...) = conditional_flux_ccc(i, j, k, ibg, ibg.grid, viscous_flux_vy, disc, clo, args...)
 @inline viscous_flux_vz(i, j, k, ibg::IBG, disc::ATD, clo::ATC, args...) = conditional_flux_cff(i, j, k, ibg, ibg.grid, viscous_flux_vz, disc, clo, args...)
 
 # fcf, cff, ccc
 @inline viscous_flux_wx(i, j, k, ibg::IBG, disc::ATD, clo::ATC, args...) = conditional_flux_fcf(i, j, k, ibg, ibg.grid, viscous_flux_wx, disc, clo, args...)
 @inline viscous_flux_wy(i, j, k, ibg::IBG, disc::ATD, clo::ATC, args...) = conditional_flux_cff(i, j, k, ibg, ibg.grid, viscous_flux_wy, disc, clo, args...)
 @inline viscous_flux_wz(i, j, k, ibg::IBG, disc::ATD, clo::ATC, args...) = conditional_flux_ccc(i, j, k, ibg, ibg.grid, viscous_flux_wz, disc, clo, args...)

# fcc, cfc, ccf
@inline diffusive_flux_x(i, j, k, ibg::IBG, disc::ATD, clo::ATC, args...) = conditional_flux_fcc(i, j, k, ibg, ibg.grid, diffusive_flux_x, disc, clo, args...)
@inline diffusive_flux_y(i, j, k, ibg::IBG, disc::ATD, clo::ATC, args...) = conditional_flux_cfc(i, j, k, ibg, ibg.grid, diffusive_flux_y, disc, clo, args...)
@inline diffusive_flux_z(i, j, k, ibg::IBG, disc::ATD, clo::ATC, args...) = conditional_flux_ccf(i, j, k, ibg, ibg.grid, diffusive_flux_z, disc, clo, args...)

#####
##### Advective fluxes
#####

@inline _advective_momentum_flux_Uu(i, j, k, ibg::IBG, args...) = conditional_flux_ccc(i, j, k, ibg, ibg, advective_momentum_flux_Uu, args...)
@inline _advective_momentum_flux_Vu(i, j, k, ibg::IBG, args...) = conditional_flux_ffc(i, j, k, ibg, ibg, advective_momentum_flux_Vu, args...)
@inline _advective_momentum_flux_Wu(i, j, k, ibg::IBG, args...) = conditional_flux_fcf(i, j, k, ibg, ibg, advective_momentum_flux_Wu, args...)

@inline _advective_momentum_flux_Uv(i, j, k, ibg::IBG, args...) = conditional_flux_ffc(i, j, k, ibg, ibg, advective_momentum_flux_Uv, args...)
@inline _advective_momentum_flux_Vv(i, j, k, ibg::IBG, args...) = conditional_flux_ccc(i, j, k, ibg, ibg, advective_momentum_flux_Vv, args...)
@inline _advective_momentum_flux_Wv(i, j, k, ibg::IBG, args...) = conditional_flux_cff(i, j, k, ibg, ibg, advective_momentum_flux_Wv, args...)

@inline _advective_momentum_flux_Uw(i, j, k, ibg::IBG, args...) = conditional_flux_fcf(i, j, k, ibg, ibg, advective_momentum_flux_Uw, args...)
@inline _advective_momentum_flux_Vw(i, j, k, ibg::IBG, args...) = conditional_flux_cff(i, j, k, ibg, ibg, advective_momentum_flux_Vw, args...)
@inline _advective_momentum_flux_Ww(i, j, k, ibg::IBG, args...) = conditional_flux_ccc(i, j, k, ibg, ibg, advective_momentum_flux_Ww, args...)
                                                                                                                                           
@inline _advective_tracer_flux_x(i, j, k, ibg::IBG, args...) = conditional_flux_fcc(i, j, k, ibg, ibg, advective_tracer_flux_x, args...)
@inline _advective_tracer_flux_y(i, j, k, ibg::IBG, args...) = conditional_flux_cfc(i, j, k, ibg, ibg, advective_tracer_flux_y, args...)
@inline _advective_tracer_flux_z(i, j, k, ibg::IBG, args...) = conditional_flux_ccf(i, j, k, ibg, ibg, advective_tracer_flux_z, args...)
