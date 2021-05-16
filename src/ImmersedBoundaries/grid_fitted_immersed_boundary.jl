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

# ccc, ffc, fcf
@inline viscous_flux_ux(i, j, k, ibg::IBG{FT}, disc::ATD, clo::ATC, args...) where FT = ifelse(solid_node(c, c, c, i, j, k, ibg), zero(FT), viscous_flux_ux(i, j, k, ibg.grid, disc, clo, args...))
@inline viscous_flux_uy(i, j, k, ibg::IBG{FT}, disc::ATD, clo::ATC, args...) where FT = ifelse(solid_node(f, f, c, i, j, k, ibg), zero(FT), viscous_flux_uy(i, j, k, ibg.grid, disc, clo, args...))
@inline viscous_flux_uz(i, j, k, ibg::IBG{FT}, disc::ATD, clo::ATC, args...) where FT = ifelse(solid_node(f, c, f, i, j, k, ibg), zero(FT), viscous_flux_uz(i, j, k, ibg.grid, disc, clo, args...))

# ffc, ccc, cff
@inline viscous_flux_vx(i, j, k, ibg::IBG{FT}, disc::ATD, clo::ATC, args...) where FT = ifelse(solid_node(f, f, c, i, j, k, ibg), zero(FT), viscous_flux_vx(i, j, k, ibg.grid, disc, clo, args...))
@inline viscous_flux_vy(i, j, k, ibg::IBG{FT}, disc::ATD, clo::ATC, args...) where FT = ifelse(solid_node(c, c, c, i, j, k, ibg), zero(FT), viscous_flux_vy(i, j, k, ibg.grid, disc, clo, args...))
@inline viscous_flux_vz(i, j, k, ibg::IBG{FT}, disc::ATD, clo::ATC, args...) where FT = ifelse(solid_node(c, f, f, i, j, k, ibg), zero(FT), viscous_flux_vz(i, j, k, ibg.grid, disc, clo, args...))

# fcf, cff, ccc
@inline viscous_flux_wx(i, j, k, ibg::IBG{FT}, disc::ATD, clo::ATC, args...) where FT = ifelse(solid_node(f, c, f, i, j, k, ibg), zero(FT), viscous_flux_wx(i, j, k, ibg.grid, disc, clo, args...))
@inline viscous_flux_wy(i, j, k, ibg::IBG{FT}, disc::ATD, clo::ATC, args...) where FT = ifelse(solid_node(c, f, f, i, j, k, ibg), zero(FT), viscous_flux_wy(i, j, k, ibg.grid, disc, clo, args...))
@inline viscous_flux_wz(i, j, k, ibg::IBG{FT}, disc::ATD, clo::ATC, args...) where FT = ifelse(solid_node(c, c, c, i, j, k, ibg), zero(FT), viscous_flux_wz(i, j, k, ibg.grid, disc, clo, args...))

# fcc, cfc, ccf
@inline diffusive_flux_x(i, j, k, ibg::IBG{FT}, disc::ATD, clo::ATC, args...) where FT = ifelse(solid_node(f, c, c, i, j, k, ibg), zero(FT), diffusive_flux_x(i, j, k, ibg.grid, disc, clo, args...))
@inline diffusive_flux_y(i, j, k, ibg::IBG{FT}, disc::ATD, clo::ATC, args...) where FT = ifelse(solid_node(c, f, c, i, j, k, ibg), zero(FT), diffusive_flux_y(i, j, k, ibg.grid, disc, clo, args...))
@inline diffusive_flux_z(i, j, k, ibg::IBG{FT}, disc::ATD, clo::ATC, args...) where FT = ifelse(solid_node(c, c, f, i, j, k, ibg), zero(FT), diffusive_flux_z(i, j, k, ibg.grid, disc, clo, args...))

