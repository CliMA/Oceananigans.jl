struct VerticallyImplicitCentered <: AbstractCenteredAdvectionScheme{1, FT} end

_advective_momentum_flux_Wu(i, j, k, grid, ::VerticallyImplicitCentered, args...) = zero(grid) 
_advective_momentum_flux_Wv(i, j, k, grid, ::VerticallyImplicitCentered, args...) = zero(grid)  
_advective_tracer_flux_z(i, j, k, grid, ::VerticallyImplicitCentered, args...)    = zero(grid)

# second - order centered reconstruction in the horizontal
@inline inner_symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, ::VerticallyImplicitCentered, ψ, idx, loc, args...)           = ℑxᶠᵃᵃ(i, j, k, grid, ψ)
@inline inner_symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, ::VerticallyImplicitCentered, ψ::Function, idx, loc, args...) = ℑxᶠᵃᵃ(i, j, k, grid, ψ, args...)
    
@inline inner_symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, ::VerticallyImplicitCentered, ψ, idx, loc, args...)           = ℑyᵃᶠᵃ(i, j, k, grid, ψ)
@inline inner_symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, ::VerticallyImplicitCentered, ψ::Function, idx, loc, args...) = ℑyᵃᶠᵃ(i, j, k, grid, ψ, args...)

const VerticallyImplicitVectorInvariant = VectorInvariant{<:Any, <:Any, <:Any, <:Any, <:Any, <:VerticallyImplicitCentered}
const VerticallyImplicitTracerAdvection = TracerAdvection{<:Any, <:Any, <:VerticallyImplicitCentered} 

const VerticallyImplicitAdvection = Union{VerticallyImplicitCentered, VerticallyImplicitVectorInvariant, VerticallyImplicitTracerAdvection}