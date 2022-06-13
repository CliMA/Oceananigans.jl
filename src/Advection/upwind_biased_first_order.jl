#####
##### Upwind-biased 1rd-order advection scheme
#####

"""
    struct UpwindBiasedFirstOrder <: AbstractUpwindBiasedAdvectionScheme{0}

Upwind-biased first-order advection scheme.
"""
struct UpwindBiasedFirstOrder{CA} <: AbstractUpwindBiasedAdvectionScheme{0} 
    "advection scheme used near boundaries"
    child_advection :: CA
end

UpwindBiasedFirstOrder() = UpwindBiasedFirstOrder(nothing)

const U1 = UpwindBiasedFirstOrder

@inline boundary_buffer(::U1) = 0

@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, ::U1, c) = ℑxᶠᵃᵃ(i, j, k, grid, c)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, ::U1, c) = ℑyᵃᶠᵃ(i, j, k, grid, c)
@inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, ::U1, c) = ℑzᵃᵃᶠ(i, j, k, grid, c)

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, ::U1, u) = ℑxᶜᵃᵃ(i, j, k, grid, u)
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, ::U1, v) = ℑyᵃᶜᵃ(i, j, k, grid, v)
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, ::U1, w) = ℑzᵃᵃᶜ(i, j, k, grid, w)

@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, ::U1, f::Function, args...) = ℑxᶠᵃᵃ(i, j, k, grid, f, args...)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, ::U1, f::Function, args...) = ℑyᵃᶠᵃ(i, j, k, grid, f, args...)
@inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, ::U1, f::Function, args...) = ℑzᵃᵃᶠ(i, j, k, grid, f, args...)

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, ::U1, f::Function, args...) = ℑxᶜᵃᵃ(i, j, k, grid, f, args...)
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, ::U1, f::Function, args...) = ℑyᵃᶜᵃ(i, j, k, grid, f, args...)
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, ::U1, f::Function, args...) = ℑzᵃᵃᶜ(i, j, k, grid, f, args...)

@inline left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::U1, c) = @inbounds c[i-1, j, k] 
@inline left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::U1, c) = @inbounds c[i, j-1, k]
@inline left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::U1, c) = @inbounds c[i, j, k-1]

@inline right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::U1, c) = @inbounds c[i, j, k] 
@inline right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::U1, c) = @inbounds c[i, j, k]
@inline right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::U1, c) = @inbounds c[i, j, k]

@inline left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::U1, f::Function, args...) = @inbounds f(i-1, j, k, grid, args...) 
@inline left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::U1, f::Function, args...) = @inbounds f(i, j-1, k, grid, args...)
@inline left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::U1, f::Function, args...) = @inbounds f(i, j, k-1, grid, args...)

@inline right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::U1, f::Function, args...) = @inbounds f(i, j, k, grid, args...) 
@inline right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::U1, f::Function, args...) = @inbounds f(i, j, k, grid, args...)
@inline right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::U1, f::Function, args...) = @inbounds f(i, j, k, grid, args...)

@inline left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::U1, args...) = left_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, args...)
@inline left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::U1, args...) = left_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, args...)
@inline left_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::U1, args...) = left_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, args...)

@inline right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::U1, args...) = right_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, args...)
@inline right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::U1, args...) = right_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, args...)
@inline right_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::U1, args...) = right_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, args...)
