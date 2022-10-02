
#####
##### Flat Topologies
#####

using Oceananigans.Operators: XFlatGrid, YFlatGrid, ZFlatGrid

for SchemeType in [:CenteredScheme, :UpwindScheme]
    @eval begin
        @inline advective_momentum_flux_Uu(i, j, k, grid::XFlatGrid, scheme::$SchemeType, U, u) = zero(grid)
        @inline advective_momentum_flux_Vu(i, j, k, grid::XFlatGrid, scheme::$SchemeType, V, u) = zero(grid)
        @inline advective_momentum_flux_Wu(i, j, k, grid::XFlatGrid, scheme::$SchemeType, W, u) = zero(grid)

        @inline advective_momentum_flux_Uv(i, j, k, grid::YFlatGrid, scheme::$SchemeType, U, v) = zero(grid)
        @inline advective_momentum_flux_Vv(i, j, k, grid::YFlatGrid, scheme::$SchemeType, V, v) = zero(grid)
        @inline advective_momentum_flux_Wv(i, j, k, grid::YFlatGrid, scheme::$SchemeType, W, v) = zero(grid)

        @inline advective_momentum_flux_Uw(i, j, k, grid::ZFlatGrid, scheme::$SchemeType, U, w) = zero(grid)
        @inline advective_momentum_flux_Vw(i, j, k, grid::ZFlatGrid, scheme::$SchemeType, V, w) = zero(grid)
        @inline advective_momentum_flux_Ww(i, j, k, grid::ZFlatGrid, scheme::$SchemeType, W, w) = zero(grid)

        @inline advective_momentum_flux_Uv(i, j, k, grid::XFlatGrid, scheme::$SchemeType, U, v) = zero(grid)
        @inline advective_momentum_flux_Uw(i, j, k, grid::XFlatGrid, scheme::$SchemeType, U, w) = zero(grid)

        @inline advective_momentum_flux_Vu(i, j, k, grid::YFlatGrid, scheme::$SchemeType, V, u) = zero(grid)
        @inline advective_momentum_flux_Vw(i, j, k, grid::YFlatGrid, scheme::$SchemeType, V, w) = zero(grid)

        @inline advective_momentum_flux_Wu(i, j, k, grid::ZFlatGrid, scheme::$SchemeType, W, u) = zero(grid)
        @inline advective_momentum_flux_Wv(i, j, k, grid::ZFlatGrid, scheme::$SchemeType, W, v) = zero(grid)
    end
end

@inline inner_right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid::XFlatGrid, scheme, ψ, args...) = ψ[i, j, k]
@inline inner_right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid::YFlatGrid, scheme, ψ, args...) = ψ[i, j, k]
@inline inner_right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid::ZFlatGrid, scheme, ψ, args...) = ψ[i, j, k]

@inline inner_right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid::XFlatGrid, scheme, ψ::Function, idx, loc, args...) = ψ(i, j, k, grid, args...)
@inline inner_right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid::YFlatGrid, scheme, ψ::Function, idx, loc, args...) = ψ(i, j, k, grid, args...)
@inline inner_right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid::ZFlatGrid, scheme, ψ::Function, idx, loc, args...) = ψ(i, j, k, grid, args...)
