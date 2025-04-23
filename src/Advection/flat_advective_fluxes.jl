using Oceananigans.ImmersedBoundaries

#####
##### Flat Topologies
#####

const XFG = Union{<:XFlatGrid, <:ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:XFlatGrid}}
const YFG = Union{<:YFlatGrid, <:ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:YFlatGrid}}
const ZFG = Union{<:ZFlatGrid, <:ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:ZFlatGrid}}

for SchemeType in [:CenteredScheme, :UpwindScheme]
    @eval begin
        @inline advective_momentum_flux_Uu(i, j, k, grid::XFG, ::$SchemeType, U, u) = zero(grid)
        @inline advective_momentum_flux_Uv(i, j, k, grid::XFG, ::$SchemeType, U, v) = zero(grid)
        @inline advective_momentum_flux_Uw(i, j, k, grid::XFG, ::$SchemeType, U, w) = zero(grid)

        @inline advective_momentum_flux_Vv(i, j, k, grid::YFG, ::$SchemeType, V, v) = zero(grid)
        @inline advective_momentum_flux_Vu(i, j, k, grid::YFG, ::$SchemeType, V, u) = zero(grid)
        @inline advective_momentum_flux_Vw(i, j, k, grid::YFG, ::$SchemeType, V, w) = zero(grid)

        @inline advective_momentum_flux_Wu(i, j, k, grid::ZFG, ::$SchemeType, W, u) = zero(grid)
        @inline advective_momentum_flux_Wv(i, j, k, grid::ZFG, ::$SchemeType, W, v) = zero(grid)
        @inline advective_momentum_flux_Ww(i, j, k, grid::ZFG, ::$SchemeType, W, w) = zero(grid)

        @inline advective_tracer_flux_x(i, j, k, grid::XFG, ::$SchemeType, U, c) = zero(grid)
        @inline advective_tracer_flux_y(i, j, k, grid::YFG, ::$SchemeType, U, c) = zero(grid)
        @inline advective_tracer_flux_z(i, j, k, grid::ZFG, ::$SchemeType, U, c) = zero(grid)
    end
end

FlatGrids = [:XFG, :YFG, :ZFG, :XFG, :YFG, :ZFG]

# Flat interpolations...
for (dir, GridType) in zip((:xᶠᵃᵃ, :yᵃᶠᵃ, :zᵃᵃᶠ, :xᶜᵃᵃ, :yᵃᶜᵃ, :zᵃᵃᶜ), FlatGrids)
    alt_symm_interp   = Symbol(:_symmetric_interpolate_, dir)
    alt_biased_interp = Symbol(:_biased_interpolate_, dir)
    @eval begin
        @inline $alt_symm_interp(i, j, k, grid::$GridType, ::HOADV, ψ, args...) = @inbounds ψ[i, j, k]
        @inline $alt_symm_interp(i, j, k, grid::$GridType, ::LOADV, ψ, args...) = @inbounds ψ[i, j, k]
        @inline $alt_symm_interp(i, j, k, grid::$GridType, ::HOADV, ψ::Callable, args...) = @inbounds ψ(i, j, k, grid, args...)
        @inline $alt_symm_interp(i, j, k, grid::$GridType, ::LOADV, ψ::Callable, args...) = @inbounds ψ(i, j, k, grid, args...)
        @inline $alt_symm_interp(i, j, k, grid::$GridType, ::HOADV, ψ::Callable, ::AS, args...) = @inbounds ψ(i, j, k, grid, args...)
        @inline $alt_symm_interp(i, j, k, grid::$GridType, ::LOADV, ψ::Callable, ::AS, args...) = @inbounds ψ(i, j, k, grid, args...)

        @inline $alt_biased_interp(i, j, k, grid::$GridType, ::HOADV, ψ, args...) = @inbounds ψ[i, j, k]
        @inline $alt_biased_interp(i, j, k, grid::$GridType, ::LOADV, ψ, args...) = @inbounds ψ[i, j, k]
        @inline $alt_biased_interp(i, j, k, grid::$GridType, ::HOADV, bias, ψ::Callable, args...) = ψ(i, j, k, grid, args...)
        @inline $alt_biased_interp(i, j, k, grid::$GridType, ::LOADV, bias, ψ::Callable, args...) = ψ(i, j, k, grid, args...)
        @inline $alt_biased_interp(i, j, k, grid::$GridType, ::HOADV, bias, ψ::Callable, ::AS, args...) = ψ(i, j, k, grid, args...)
        @inline $alt_biased_interp(i, j, k, grid::$GridType, ::LOADV, bias, ψ::Callable, ::AS, args...) = ψ(i, j, k, grid, args...)
    end
end