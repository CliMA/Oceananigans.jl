#####
##### "No" advection
#####

boundary_buffer(::Nothing) = 0

@inline momentum_flux_uu(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U, u) = zero(FT)
@inline momentum_flux_uv(i, j, k, grid::AbstractGrid{FT}, ::Nothing, V, u) = zero(FT)
@inline momentum_flux_uw(i, j, k, grid::AbstractGrid{FT}, ::Nothing, W, u) = zero(FT)

@inline momentum_flux_vu(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U, v) = zero(FT)
@inline momentum_flux_vv(i, j, k, grid::AbstractGrid{FT}, ::Nothing, V, v) = zero(FT)
@inline momentum_flux_vw(i, j, k, grid::AbstractGrid{FT}, ::Nothing, W, v) = zero(FT)

@inline momentum_flux_wu(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U, w) = zero(FT)
@inline momentum_flux_wv(i, j, k, grid::AbstractGrid{FT}, ::Nothing, V, w) = zero(FT)
@inline momentum_flux_ww(i, j, k, grid::AbstractGrid{FT}, ::Nothing, W, w) = zero(FT)

@inline advective_tracer_flux_x(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U, c) = zero(FT)
@inline advective_tracer_flux_y(i, j, k, grid::AbstractGrid{FT}, ::Nothing, V, c) = zero(FT)
@inline advective_tracer_flux_z(i, j, k, grid::AbstractGrid{FT}, ::Nothing, W, c) = zero(FT)
