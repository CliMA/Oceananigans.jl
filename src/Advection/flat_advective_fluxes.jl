
const CenteredOrUpwind = Union{AbstractCenteredAdvectionScheme, AbstractUpwindBiasedAdvectionScheme}

#####
##### Flat Topologies
#####

@inline advective_momentum_flux_Uu(i, j, k, grid::AbstractGrid{FT, Flat}, scheme::CenteredOrUpwind, U, u) where FT = zero(grid)
@inline advective_momentum_flux_Vu(i, j, k, grid::AbstractGrid{FT, Flat}, scheme::CenteredOrUpwind, V, u) where FT = zero(grid)
@inline advective_momentum_flux_Wu(i, j, k, grid::AbstractGrid{FT, Flat}, scheme::CenteredOrUpwind, W, u) where FT = zero(grid)

@inline advective_momentum_flux_Uv(i, j, k, grid::AbstractGrid{FT, TX, Flat}, scheme::CenteredOrUpwind, U, v) where {FT, TX} = zero(grid)
@inline advective_momentum_flux_Vv(i, j, k, grid::AbstractGrid{FT, TX, Flat}, scheme::CenteredOrUpwind, V, v) where {FT, TX} = zero(grid)
@inline advective_momentum_flux_Wv(i, j, k, grid::AbstractGrid{FT, TX, Flat}, scheme::CenteredOrUpwind, W, v) where {FT, TX} = zero(grid)

@inline advective_momentum_flux_Uw(i, j, k, grid::AbstractGrid{FT, TX, TY, Flat}, scheme::CenteredOrUpwind, U, w) where {FT, TX, TY} = zero(grid)
@inline advective_momentum_flux_Vw(i, j, k, grid::AbstractGrid{FT, TX, TY, Flat}, scheme::CenteredOrUpwind, V, w) where {FT, TX, TY} = zero(grid)
@inline advective_momentum_flux_Ww(i, j, k, grid::AbstractGrid{FT, TX, TY, Flat}, scheme::CenteredOrUpwind, W, w) where {FT, TX, TY} = zero(grid)

@inline advective_momentum_flux_Uv(i, j, k, grid::AbstractGrid{FT, Flat}, scheme::CenteredOrUpwind, U, v) where {FT} = zero(grid)
@inline advective_momentum_flux_Uw(i, j, k, grid::AbstractGrid{FT, Flat}, scheme::CenteredorUpwind, U, w) where {FT} = zero(grid)

@inline advective_momentum_flux_Vu(i, j, k, grid::AbstractGrid{FT, TX, Flat}, scheme::CenteredOrUpwind, V, u) where {FT, TX} = zero(grid)
@inline advective_momentum_flux_Vw(i, j, k, grid::AbstractGrid{FT, TX, Flat}, scheme::CenteredOrUpwind, V, w) where {FT, TX} = zero(grid)

@inline advective_momentum_flux_Wu(i, j, k, grid::AbstractGrid{FT, TX, TY, Flat}, scheme::CenteredOrUpwind, W, u) where {FT, TX, TY} = zero(grid)
@inline advective_momentum_flux_Wv(i, j, k, grid::AbstractGrid{FT, TX, TY, Flat}, scheme::CenteredOrUpwind, W, v) where {FT, TX, TY} = zero(grid)
