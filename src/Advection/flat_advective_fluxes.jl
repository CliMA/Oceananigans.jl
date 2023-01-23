
#####
##### Flat Topologies
#####

using Oceananigans.Operators: XFlatGrid, YFlatGrid, ZFlatGrid

for SchemeType in [:CenteredScheme, :UpwindScheme]
    @eval begin
        @inline advective_momentum_flux_Uu(i, j, k, grid::XFlatGrid, ::$SchemeType, U, u, is, js, ks) = zero(grid)
        @inline advective_momentum_flux_Uv(i, j, k, grid::XFlatGrid, ::$SchemeType, U, v, is, js, ks) = zero(grid)
        @inline advective_momentum_flux_Uw(i, j, k, grid::XFlatGrid, ::$SchemeType, U, w, is, js, ks) = zero(grid)

        @inline advective_momentum_flux_Vv(i, j, k, grid::YFlatGrid, ::$SchemeType, V, v, is, js, ks) = zero(grid)
        @inline advective_momentum_flux_Vu(i, j, k, grid::YFlatGrid, ::$SchemeType, V, u, is, js, ks) = zero(grid)
        @inline advective_momentum_flux_Vw(i, j, k, grid::YFlatGrid, ::$SchemeType, V, w, is, js, ks) = zero(grid)

        @inline advective_momentum_flux_Wu(i, j, k, grid::ZFlatGrid, ::$SchemeType, W, u, is, js, ks) = zero(grid)
        @inline advective_momentum_flux_Wv(i, j, k, grid::ZFlatGrid, ::$SchemeType, W, v, is, js, ks) = zero(grid)
        @inline advective_momentum_flux_Ww(i, j, k, grid::ZFlatGrid, ::$SchemeType, W, w, is, js, ks) = zero(grid)
    end
end

Grids = [:XFlatGrid, :YFlatGrid, :ZFlatGrid, :XFlatGrid, :YFlatGrid, :ZFlatGrid]

for side in (:left_biased, :right_biased, :symmetric)
    for (dir, Grid) in zip([:xᶠᵃᵃ, :yᵃᶠᵃ, :zᵃᵃᶠ, :xᶜᵃᵃ, :yᵃᶜᵃ, :zᵃᵃᶜ], Grids)
        interp_function = Symbol(side, :_interpolate_, dir)
        @eval begin
            $interp_function(i, j, k, grid::$Grid, scheme, ψ, args...) = @inbounds ψ[i, j, k]
            $interp_function(i, j, k, grid::$Grid, scheme, ψ::Function, args...) = @inbounds ψ(i, j, k, grid, args...)
        end
    end
end

@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid::XFlatGrid, scheme::AbstractUpwindBiasedAdvectionScheme, c) = @inbounds c[i, j, k]
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid::YFlatGrid, scheme::AbstractUpwindBiasedAdvectionScheme, c) = @inbounds c[i, j, k]
@inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid::ZFlatGrid, scheme::AbstractUpwindBiasedAdvectionScheme, c) = @inbounds c[i, j, k]

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid::XFlatGrid, scheme::AbstractUpwindBiasedAdvectionScheme, u) = @inbounds u[i, j, k]
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid::YFlatGrid, scheme::AbstractUpwindBiasedAdvectionScheme, v) = @inbounds v[i, j, k]
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid::ZFlatGrid, scheme::AbstractUpwindBiasedAdvectionScheme, w) = @inbounds w[i, j, k]
