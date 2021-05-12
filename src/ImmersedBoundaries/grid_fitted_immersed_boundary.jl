struct GridFittedImmersedBoundary{S, G} <: AbstractImmersedBoundary
    solid :: S
    grid :: G
end

const c = Center()
const f = Face()

@inline solid_cell(i, j, k, grid, ib) = ib.solid(node(c, c, c, i, j, k, grid)...)

@inline solid_node(::Center, ::Center, ::Center, i, j, k, grid, ib) = solid_cell(i, j, k, grid, ib)

@inline solid_node(::Face, ::Center, ::Center, i, j, k, grid, ib) = solid_cell(i, j, k, grid, ib) || solid_cell(i-1, j, k, grid, ib)
@inline solid_node(::Center, ::Face, ::Center, i, j, k, grid, ib) = solid_cell(i, j, k, grid, ib) || solid_cell(i, j-1, k, grid, ib)
@inline solid_node(::Center, ::Center, ::Face, i, j, k, grid, ib) = solid_cell(i, j, k, grid, ib) || solid_cell(i, j, k-1, grid, ib)

@inline solid_node(::Face, ::Face, ::Center, i, j, k, grid, ib) = solid_node(c, f, c, i, j, k, grid, ib) || solid_node(c, f, c, i-1, j, k, grid, ib)
@inline solid_node(::Face, ::Center, ::Face, i, j, k, grid, ib) = solid_node(c, c, f, i, j, k, grid, ib) || solid_node(c, c, f, i-1, j, k, grid, ib)
@inline solid_node(::Center, ::Face, ::Face, i, j, k, grid, ib) = solid_node(c, f, c, i, j, k, grid, ib) || solid_node(c, f, c, i, j, k-1, grid, ib)

@inline solid_node(::Face, ::Face, ::Face, i, j, k, grid, ib) = solid_node(c, f, f, i, j, k, grid, ib) || solid_node(c, f, f, i-1, j, k, grid, ib)

#####
##### GridFittedImmersedBoundary: use caution
#####

const GFIB = GridFittedImmersedBoundary
const AG = AbstractGrid
const c = Center()
const f = Face()

# ccc, ffc, fcf
@inline viscous_flux_ux(i, j, k, grid::AG{FT}, ib::GFIB, args...) where FT = ifelse(solid_node(c, c, c, i, j, k, grid, ib), zero(FT), viscous_flux_ux(i, j, k, grid, args...))
@inline viscous_flux_uy(i, j, k, grid::AG{FT}, ib::GFIB, args...) where FT = ifelse(solid_node(f, f, c, i, j, k, grid, ib), zero(FT), viscous_flux_uy(i, j, k, grid, args...))
@inline viscous_flux_uz(i, j, k, grid::AG{FT}, ib::GFIB, args...) where FT = ifelse(solid_node(f, c, f, i, j, k, grid, ib), zero(FT), viscous_flux_uz(i, j, k, grid, args...))

# ffc, ccc, cff
@inline viscous_flux_vx(i, j, k, grid::AG{FT}, ib::GFIB, args...) where FT = ifelse(solid_node(f, f, c, i, j, k, grid, ib), zero(FT), viscous_flux_vx(i, j, k, grid, args...))
@inline viscous_flux_vy(i, j, k, grid::AG{FT}, ib::GFIB, args...) where FT = ifelse(solid_node(c, c, c, i, j, k, grid, ib), zero(FT), viscous_flux_vy(i, j, k, grid, args...))
@inline viscous_flux_vz(i, j, k, grid::AG{FT}, ib::GFIB, args...) where FT = ifelse(solid_node(c, f, f, i, j, k, grid, ib), zero(FT), viscous_flux_vz(i, j, k, grid, args...))

# fcf, cff, ccc
@inline viscous_flux_wx(i, j, k, grid::AG{FT}, ib::GFIB, args...) where FT = ifelse(solid_node(f, c, f, i, j, k, grid, ib), zero(FT), viscous_flux_wx(i, j, k, grid, args...))
@inline viscous_flux_wy(i, j, k, grid::AG{FT}, ib::GFIB, args...) where FT = ifelse(solid_node(c, f, f, i, j, k, grid, ib), zero(FT), viscous_flux_wy(i, j, k, grid, args...))
@inline viscous_flux_wz(i, j, k, grid::AG{FT}, ib::GFIB, args...) where FT = ifelse(solid_node(c, c, c, i, j, k, grid, ib), zero(FT), viscous_flux_wz(i, j, k, grid, args...))

# fcc, cfc, ccf
@inline diffusive_flux_x(i, j, k, grid::AG{FT}, ib::GFIB, args...) where FT = ifelse(solid_node(f, c, c, i, j, k, grid, ib), zero(FT), diffusive_flux_x(i, j, k, grid, args...))
@inline diffusive_flux_y(i, j, k, grid::AG{FT}, ib::GFIB, args...) where FT = ifelse(solid_node(c, f, c, i, j, k, grid, ib), zero(FT), diffusive_flux_y(i, j, k, grid, args...))
@inline diffusive_flux_z(i, j, k, grid::AG{FT}, ib::GFIB, args...) where FT = ifelse(solid_node(c, c, f, i, j, k, grid, ib), zero(FT), diffusive_flux_z(i, j, k, grid, args...))


