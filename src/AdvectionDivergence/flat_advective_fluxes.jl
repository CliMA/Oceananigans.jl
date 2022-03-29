const CenteredOrUpwind = Union{AbstractCenteredAdvectionScheme, AbstractUpwindBiasedAdvectionScheme}

#####
##### Flat Topologies
#####

momentum_flux_uu(i, j, k, grid::AbstractGrid{FT, Flat}, scheme::CenteredOrUpwind, u)    where FT = zero(FT)
momentum_flux_vu(i, j, k, grid::AbstractGrid{FT, Flat}, scheme::CenteredOrUpwind, u, v) where FT = zero(FT)
momentum_flux_wu(i, j, k, grid::AbstractGrid{FT, Flat}, scheme::CenteredOrUpwind, u, w) where FT = zero(FT)

momentum_flux_uv(i, j, k, grid::AbstractGrid{FT, TX, Flat}, scheme::CenteredOrUpwind, u, v) where {FT, TX} = zero(FT)
momentum_flux_vv(i, j, k, grid::AbstractGrid{FT, TX, Flat}, scheme::CenteredOrUpwind, v)    where {FT, TX} = zero(FT)
momentum_flux_wv(i, j, k, grid::AbstractGrid{FT, TX, Flat}, scheme::CenteredOrUpwind, v, w) where {FT, TX} = zero(FT)

momentum_flux_uw(i, j, k, grid::AbstractGrid{FT, TX, TY, Flat}, scheme::CenteredOrUpwind, u, v) where {FT, TX, TY} = zero(FT)
momentum_flux_vw(i, j, k, grid::AbstractGrid{FT, TX, TY, Flat}, scheme::CenteredOrUpwind, v, w) where {FT, TX, TY} = zero(FT)
momentum_flux_ww(i, j, k, grid::AbstractGrid{FT, TX, TY, Flat}, scheme::CenteredOrUpwind, w)    where {FT, TX, TY} = zero(FT)
