#####
##### Flat Topologies
#####

const XFG = Union{<:XFlatGrid, <:ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:XFlatGrid}}
const YFG = Union{<:YFlatGrid, <:ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:YFlatGrid}}
const ZFG = Union{<:ZFlatGrid, <:ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:ZFlatGrid}}

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

# Flat interpolations...
for SchemeType in (:CenteredScheme, :UpwindScheme), (ξ, FlatGrid) in zip((:x, :y, :z), (:XFG, :YFG, :ZFG)),
    ℓx in (:ᶜ, :ᶠ), ℓy in (:ᶜ, :ᶠ), ℓz in (:ᶜ, :ᶠ)

    symmetric_interpolation = Symbol(:symmetric_interpolate_, ξ, ℓx, ℓy, ℓz)
    biased_interpolation    = Symbol(:biased_interpolate_, ξ, ℓx, ℓy, ℓz)

    @eval begin
        @inline $symmetric_interpolation(i, j, k, grid::$FlatGrid, ::$SchemeType, ψ, args...) = @inbounds ψ[i, j, k]
        @inline $symmetric_interpolation(i, j, k, grid::$FlatGrid, ::$SchemeType, ψ::Callable, args...) = @inbounds ψ(i, j, k, grid, args...)
        @inline $symmetric_interpolation(i, j, k, grid::$FlatGrid, ::$SchemeType, ψ::Callable, ::AS, args...) = @inbounds ψ(i, j, k, grid, args...)

        @inline $biased_interpolation(i, j, k, grid::$FlatGrid, ::$SchemeType, bias, ψ, args...) = @inbounds ψ[i, j, k]
        @inline $biased_interpolation(i, j, k, grid::$FlatGrid, ::$SchemeType, bias, ψ::Callable, args...) = ψ(i, j, k, grid, args...)
        @inline $biased_interpolation(i, j, k, grid::$FlatGrid, ::$SchemeType, bias, ψ::Callable, ::AS, args...) = ψ(i, j, k, grid, args...)
    end
end
