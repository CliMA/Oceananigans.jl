#####
##### "No" advection
#####

boundary_buffer(::Nothing) = 0

@inline momentum_flux_uu(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U, u) where FT = zero(FT)
@inline momentum_flux_uv(i, j, k, grid::AbstractGrid{FT}, ::Nothing, V, u) where FT = zero(FT)
@inline momentum_flux_uw(i, j, k, grid::AbstractGrid{FT}, ::Nothing, W, u) where FT = zero(FT)

@inline momentum_flux_vu(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U, v) where FT = zero(FT)
@inline momentum_flux_vv(i, j, k, grid::AbstractGrid{FT}, ::Nothing, V, v) where FT = zero(FT)
@inline momentum_flux_vw(i, j, k, grid::AbstractGrid{FT}, ::Nothing, W, v) where FT = zero(FT)

@inline momentum_flux_wu(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U, w) where FT = zero(FT)
@inline momentum_flux_wv(i, j, k, grid::AbstractGrid{FT}, ::Nothing, V, w) where FT = zero(FT)
@inline momentum_flux_ww(i, j, k, grid::AbstractGrid{FT}, ::Nothing, W, w) where FT = zero(FT)

@inline advective_tracer_flux_x(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U, c) where FT = zero(FT)
@inline advective_tracer_flux_y(i, j, k, grid::AbstractGrid{FT}, ::Nothing, V, c) where FT = zero(FT)
@inline advective_tracer_flux_z(i, j, k, grid::AbstractGrid{FT}, ::Nothing, W, c) where FT = zero(FT)
