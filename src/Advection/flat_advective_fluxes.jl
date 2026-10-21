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

# A reconstruction along a Flat direction is the value at the node
for (ξ, GridType) in zip((:x, :y, :z), (:XFG, :YFG, :ZFG)), ℓx in (:ᶜ, :ᶠ), ℓy in (:ᶜ, :ᶠ), ℓz in (:ᶜ, :ᶠ)
    biased_interpolate = Symbol(:biased_interpolate_, ξ, ℓx, ℓy, ℓz)
    @eval begin
        @inline $biased_interpolate(i, j, k, grid::$GridType, ::AbstractAdvectionScheme, bias, ψ, args...)      = getvalue(ψ, i, j, k, grid, args...)
        @inline $biased_interpolate(i, j, k, grid::$GridType, ::AbstractAdvectionScheme, bias, ψ, ::AS, args...) = getvalue(ψ, i, j, k, grid, args...)
    end
end
