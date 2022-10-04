
#####
##### Flat Topologies
#####

using Oceananigans.Operators: XFlatGrid, YFlatGrid, ZFlatGrid

for SchemeType in [:CenteredScheme, :UpwindScheme]
    @eval begin
        @inline advective_momentum_flux_Uu(i, j, k, grid::XFlatGrid, ::$SchemeType, U, u) = zero(grid)
        @inline advective_momentum_flux_Uv(i, j, k, grid::XFlatGrid, ::$SchemeType, U, v) = zero(grid)
        @inline advective_momentum_flux_Uw(i, j, k, grid::XFlatGrid, ::$SchemeType, U, w) = zero(grid)

        @inline advective_momentum_flux_Vv(i, j, k, grid::YFlatGrid, ::$SchemeType, V, v) = zero(grid)
        @inline advective_momentum_flux_Vu(i, j, k, grid::YFlatGrid, ::$SchemeType, V, u) = zero(grid)
        @inline advective_momentum_flux_Vw(i, j, k, grid::YFlatGrid, ::$SchemeType, V, w) = zero(grid)

        @inline advective_momentum_flux_Wu(i, j, k, grid::ZFlatGrid, ::$SchemeType, W, u) = zero(grid)
        @inline advective_momentum_flux_Wv(i, j, k, grid::ZFlatGrid, ::$SchemeType, W, v) = zero(grid)
        @inline advective_momentum_flux_Ww(i, j, k, grid::ZFlatGrid, ::$SchemeType, W, w) = zero(grid)
    end
end

@inline inner_right_biased_interpolate_xᶠᵃᵃ(i, j, k, ::XFlatGrid, scheme, ψ, args...) = @inbounds ψ[i, j, k]
@inline inner_right_biased_interpolate_yᵃᶠᵃ(i, j, k, ::YFlatGrid, scheme, ψ, args...) = @inbounds ψ[i, j, k]
@inline inner_right_biased_interpolate_zᵃᵃᶠ(i, j, k, ::ZFlatGrid, scheme, ψ, args...) = @inbounds ψ[i, j, k]

@inline inner_right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid::XFlatGrid, scheme, ψ::Function, idx, loc, args...) = @inbounds ψ(i, j, k, grid, args...)
@inline inner_right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid::YFlatGrid, scheme, ψ::Function, idx, loc, args...) = @inbounds ψ(i, j, k, grid, args...)
@inline inner_right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid::ZFlatGrid, scheme, ψ::Function, idx, loc, args...) = @inbounds ψ(i, j, k, grid, args...)

@inline inner_left_biased_interpolate_xᶠᵃᵃ(i, j, k, ::XFlatGrid, scheme, ψ, args...) = @inbounds ψ[i, j, k]
@inline inner_left_biased_interpolate_yᵃᶠᵃ(i, j, k, ::YFlatGrid, scheme, ψ, args...) = @inbounds ψ[i, j, k]
@inline inner_left_biased_interpolate_zᵃᵃᶠ(i, j, k, ::ZFlatGrid, scheme, ψ, args...) = @inbounds ψ[i, j, k]

@inline inner_left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid::XFlatGrid, scheme, ψ::Function, idx, loc, args...) = @inbounds ψ(i, j, k, grid, args...)
@inline inner_left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid::YFlatGrid, scheme, ψ::Function, idx, loc, args...) = @inbounds ψ(i, j, k, grid, args...)
@inline inner_left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid::ZFlatGrid, scheme, ψ::Function, idx, loc, args...) = @inbounds ψ(i, j, k, grid, args...)

@inline inner_symmetric_interpolate_xᶠᵃᵃ(i, j, k, ::XFlatGrid, scheme, ψ, args...) = @inbounds ψ[i, j, k]
@inline inner_symmetric_interpolate_yᵃᶠᵃ(i, j, k, ::YFlatGrid, scheme, ψ, args...) = @inbounds ψ[i, j, k]
@inline inner_symmetric_interpolate_zᵃᵃᶠ(i, j, k, ::ZFlatGrid, scheme, ψ, args...) = @inbounds ψ[i, j, k]

@inline inner_symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid::XFlatGrid, scheme, ψ::Function, idx, loc, args...) = @inbounds ψ(i, j, k, grid, args...)
@inline inner_symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid::YFlatGrid, scheme, ψ::Function, idx, loc, args...) = @inbounds ψ(i, j, k, grid, args...)
@inline inner_symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid::ZFlatGrid, scheme, ψ::Function, idx, loc, args...) = @inbounds ψ(i, j, k, grid, args...)
