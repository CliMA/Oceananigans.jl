#####
##### Centered advection scheme
#####

"""
    struct Centered{N, FT, XT, YT, ZT, CA} <: AbstractCenteredAdvectionScheme{N, FT}

Centered reconstruction scheme.
"""
struct Centered{N, FT} <: AbstractCenteredAdvectionScheme{N, FT} end

function Centered(FT::DataType=Oceananigans.defaults.FloatType; grid = nothing, order = 2) 

    if !(grid isa Nothing) 
        FT = eltype(grid)
    end

    mod(order, 2) != 0 && throw(ArgumentError("Centered reconstruction scheme is defined only for even orders"))

    N = Int(order ÷ 2)
    return Centered{N, FT}()
end

Base.summary(a::Centered{N}) where N = string("Centered(order=", 2N, ")")
Base.show(io::IO, a::Centered{N, FT}) where {N, FT} = summary(a)

# Useful aliases
Centered(grid, FT::DataType=Float64; kwargs...) = Centered(FT; grid, kwargs...)

const ACAS = AbstractCenteredAdvectionScheme

# left and right biased for Centered reconstruction are just symmetric!
@inline _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::ACAS, bias, c, args...) = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c, args...)
@inline _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::ACAS, bias, c, args...) = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, c, args...)
@inline _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::ACAS, bias, c, args...) = _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, c, args...)
@inline _biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::ACAS, bias, c, args...) = _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, c, args...)
@inline _biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::ACAS, bias, c, args...) = _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, c, args...)
@inline _biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::ACAS, bias, c, args...) = _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, c, args...)


@inline _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::ACAS, c, args...) = ℑxᶠᵃᵃ(i, j, k, grid, c, args...) #  _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.advecting_velocity_scheme, c, args...)
@inline _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::ACAS, c, args...) = ℑyᵃᶠᵃ(i, j, k, grid, c, args...) #  _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.advecting_velocity_scheme, c, args...)
@inline _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::ACAS, c, args...) = ℑzᵃᵃᶠ(i, j, k, grid, c, args...) #  _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme.advecting_velocity_scheme, c, args...)
@inline _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::ACAS, u, args...) = ℑxᶜᵃᵃ(i, j, k, grid, u, args...) #  _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme.advecting_velocity_scheme, u, args...)
@inline _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::ACAS, v, args...) = ℑyᵃᶜᵃ(i, j, k, grid, v, args...) #  _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme.advecting_velocity_scheme, v, args...)
@inline _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::ACAS, w, args...) = ℑzᵃᵃᶜ(i, j, k, grid, w, args...) #  _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme.advecting_velocity_scheme, w, args...)


# # uniform centered reconstruction
# for buffer in advection_buffers, FT in fully_supported_float_types
#     @eval begin
#         @inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, s::Centered{$buffer, $FT}, red_order::Int, ψ, args...) = symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, s, Val(red_order), ψ, args...)
#         @inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, s::Centered{$buffer, $FT}, red_order::Int, ψ, args...) = symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, s, Val(red_order), ψ, args...)
#         @inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, s::Centered{$buffer, $FT}, red_order::Int, ψ, args...) = symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, s, Val(red_order), ψ, args...)
#     end

#     for red_order in 1:buffer # The order that actually matters
#         @eval begin
#             @inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::Centered{$buffer, $FT}, ::Val{$red_order}, ψ,  args...)          = @inbounds $(calc_reconstruction_stencil(FT, red_order, :symmetric, :x, false))
#             @inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::Centered{$buffer, $FT}, ::Val{$red_order}, ψ::Function, args...) = @inbounds $(calc_reconstruction_stencil(FT, red_order, :symmetric, :x,  true))
#             @inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::Centered{$buffer, $FT}, ::Val{$red_order}, ψ, args...)           = @inbounds $(calc_reconstruction_stencil(FT, red_order, :symmetric, :y, false))
#             @inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::Centered{$buffer, $FT}, ::Val{$red_order}, ψ::Function, args...) = @inbounds $(calc_reconstruction_stencil(FT, red_order, :symmetric, :y,  true))
#             @inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::Centered{$buffer, $FT}, ::Val{$red_order}, ψ, args...)           = @inbounds $(calc_reconstruction_stencil(FT, red_order, :symmetric, :z, false))
#             @inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::Centered{$buffer, $FT}, ::Val{$red_order}, ψ::Function, args...) = @inbounds $(calc_reconstruction_stencil(FT, red_order, :symmetric, :z,  true))

#             # Flat interpolations...
#             @inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid::XFlatGrid, ::Centered{$buffer, $FT}, ::Val{$red_order}, ψ, args...)           = @inbounds ψ[i, j, k]
#             @inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid::XFlatGrid, ::Centered{$buffer, $FT}, ::Val{$red_order}, ψ::Function, args...) = @inbounds ψ(i, j, k, grid, args...)
#             @inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid::YFlatGrid, ::Centered{$buffer, $FT}, ::Val{$red_order}, ψ, args...)           = @inbounds ψ[i, j, k]
#             @inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid::YFlatGrid, ::Centered{$buffer, $FT}, ::Val{$red_order}, ψ::Function, args...) = @inbounds ψ(i, j, k, grid, args...)
#             @inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid::ZFlatGrid, ::Centered{$buffer, $FT}, ::Val{$red_order}, ψ, args...)           = @inbounds ψ[i, j, k]
#             @inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid::ZFlatGrid, ::Centered{$buffer, $FT}, ::Val{$red_order}, ψ::Function, args...) = @inbounds ψ(i, j, k, grid, args...)
#         end
#     end
# end