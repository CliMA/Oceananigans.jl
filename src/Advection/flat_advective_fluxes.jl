#####
##### Flat Topologies
#####

for SchemeType in [:CenteredScheme, :UpwindScheme]
    @eval begin
        @inline advective_momentum_flux_Uu(i, j, k, grid::XFG, ::$SchemeType, ::ETD, U, u) = zero(grid)
        @inline advective_momentum_flux_Uv(i, j, k, grid::XFG, ::$SchemeType, ::ETD, U, v) = zero(grid)
        @inline advective_momentum_flux_Uw(i, j, k, grid::XFG, ::$SchemeType, ::ETD, U, w) = zero(grid)

        @inline advective_momentum_flux_Vv(i, j, k, grid::YFG, ::$SchemeType, ::ETD, V, v) = zero(grid)
        @inline advective_momentum_flux_Vu(i, j, k, grid::YFG, ::$SchemeType, ::ETD, V, u) = zero(grid)
        @inline advective_momentum_flux_Vw(i, j, k, grid::YFG, ::$SchemeType, ::ETD, V, w) = zero(grid)

        @inline advective_momentum_flux_Wu(i, j, k, grid::ZFG, ::$SchemeType, ::ETD, W, u) = zero(grid)
        @inline advective_momentum_flux_Wv(i, j, k, grid::ZFG, ::$SchemeType, ::ETD, W, v) = zero(grid)
        @inline advective_momentum_flux_Ww(i, j, k, grid::ZFG, ::$SchemeType, ::ETD, W, w) = zero(grid)

        @inline advective_tracer_flux_x(i, j, k, grid::XFG, ::$SchemeType, ::ETD, U, c) = zero(grid)
        @inline advective_tracer_flux_y(i, j, k, grid::YFG, ::$SchemeType, ::ETD, U, c) = zero(grid)
        @inline advective_tracer_flux_z(i, j, k, grid::ZFG, ::$SchemeType, ::ETD, U, c) = zero(grid)
    end
end
