
#####
##### Flat Topologies
#####

using Oceananigans.Operators: XFlatGrid, YFlatGrid, ZFlatGrid

for SchemeType in [:CenteredScheme, :UpwindScheme]
    @eval begin
        @inline advective_momentum_flux_Uu(i, j, k, grid::XFlatGrid, scheme::$SchemeType, U, u) where FT = zero(grid)
        @inline advective_momentum_flux_Uv(i, j, k, grid::XFlatGrid, scheme::$SchemeType, U, v) where {FT} = zero(grid)
        @inline advective_momentum_flux_Uw(i, j, k, grid::XFlatGrid, scheme::$SchemeType, U, w) where {FT} = zero(grid)

        @inline advective_momentum_flux_Vv(i, j, k, grid::YFlatGrid, scheme::$SchemeType, V, v) where {FT, TX} = zero(grid)
        @inline advective_momentum_flux_Vu(i, j, k, grid::YFlatGrid, scheme::$SchemeType, V, u) where {FT, TX} = zero(grid)
        @inline advective_momentum_flux_Vw(i, j, k, grid::YFlatGrid, scheme::$SchemeType, V, w) where {FT, TX} = zero(grid)

        @inline advective_momentum_flux_Wu(i, j, k, grid::ZFlatGrid, scheme::$SchemeType, W, u) where {FT, TX, TY} = zero(grid)
        @inline advective_momentum_flux_Wv(i, j, k, grid::ZFlatGrid, scheme::$SchemeType, W, v) where {FT, TX, TY} = zero(grid)
        @inline advective_momentum_flux_Ww(i, j, k, grid::ZFlatGrid, scheme::$SchemeType, W, w) where {FT, TX, TY} = zero(grid)
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
