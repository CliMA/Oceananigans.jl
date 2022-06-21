
#####
##### Flat Topologies
#####

for SchemeType in [:AbstractCenteredAdvectionScheme, :AbstractUpwindBiasedAdvectionScheme]
    @eval begin
        @inline advective_momentum_flux_Uu(i, j, k, grid::AbstractGrid{FT, Flat}, scheme::$SchemeType, U, u) where FT = zero(grid)
        @inline advective_momentum_flux_Vu(i, j, k, grid::AbstractGrid{FT, Flat}, scheme::$SchemeType, V, u) where FT = zero(grid)
        @inline advective_momentum_flux_Wu(i, j, k, grid::AbstractGrid{FT, Flat}, scheme::$SchemeType, W, u) where FT = zero(grid)

        @inline advective_momentum_flux_Uv(i, j, k, grid::AbstractGrid{FT, TX, Flat}, scheme::$SchemeType, U, v) where {FT, TX} = zero(grid)
        @inline advective_momentum_flux_Vv(i, j, k, grid::AbstractGrid{FT, TX, Flat}, scheme::$SchemeType, V, v) where {FT, TX} = zero(grid)
        @inline advective_momentum_flux_Wv(i, j, k, grid::AbstractGrid{FT, TX, Flat}, scheme::$SchemeType, W, v) where {FT, TX} = zero(grid)

        @inline advective_momentum_flux_Uw(i, j, k, grid::AbstractGrid{FT, TX, TY, Flat}, scheme::$SchemeType, U, w) where {FT, TX, TY} = zero(grid)
        @inline advective_momentum_flux_Vw(i, j, k, grid::AbstractGrid{FT, TX, TY, Flat}, scheme::$SchemeType, V, w) where {FT, TX, TY} = zero(grid)
        @inline advective_momentum_flux_Ww(i, j, k, grid::AbstractGrid{FT, TX, TY, Flat}, scheme::$SchemeType, W, w) where {FT, TX, TY} = zero(grid)

        @inline advective_momentum_flux_Uv(i, j, k, grid::AbstractGrid{FT, Flat}, scheme::$SchemeType, U, v) where {FT} = zero(grid)
        @inline advective_momentum_flux_Uw(i, j, k, grid::AbstractGrid{FT, Flat}, scheme::$SchemeType, U, w) where {FT} = zero(grid)

        @inline advective_momentum_flux_Vu(i, j, k, grid::AbstractGrid{FT, TX, Flat}, scheme::$SchemeType, V, u) where {FT, TX} = zero(grid)
        @inline advective_momentum_flux_Vw(i, j, k, grid::AbstractGrid{FT, TX, Flat}, scheme::$SchemeType, V, w) where {FT, TX} = zero(grid)

        @inline advective_momentum_flux_Wu(i, j, k, grid::AbstractGrid{FT, TX, TY, Flat}, scheme::$SchemeType, W, u) where {FT, TX, TY} = zero(grid)
        @inline advective_momentum_flux_Wv(i, j, k, grid::AbstractGrid{FT, TX, TY, Flat}, scheme::$SchemeType, W, v) where {FT, TX, TY} = zero(grid)
    end
end