#####
##### Upwind-biased 3rd-order advection scheme
#####

"""
    struct UpwindBiasedFifthOrder <: AbstractUpwindBiasedAdvectionScheme{2}

Upwind-biased fifth-order advection scheme.
"""
struct UpwindBiasedFifthOrder <: AbstractUpwindBiasedAdvectionScheme{2} end

const U5 = UpwindBiasedFifthOrder

@inline boundary_buffer(::U5) = 2

@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, ::U5, c) = symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, centered_fourth_order, c)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, ::U5, c) = symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, centered_fourth_order, c)
@inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, ::U5, c) = symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, centered_fourth_order, c)

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, ::U5, u) = symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, centered_fourth_order, u)
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, ::U5, v) = symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, centered_fourth_order, v)
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, ::U5, w) = symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, centered_fourth_order, w)

@inline left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::U5, c) = @inbounds (- 3 * c[i+1, j, k] + 27 * c[i, j, k] + 47 * c[i-1, j, k] - 13 * c[i-2, j, k] + 2 * c[i-3, j, k]) / 60
@inline left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::U5, c) = @inbounds (- 3 * c[i, j+1, k] + 27 * c[i, j, k] + 47 * c[i, j-1, k] - 13 * c[i, j-2, k] + 2 * c[i, j-3, k]) / 60
@inline left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::U5, c) = @inbounds (- 3 * c[i, j, k+1] + 27 * c[i, j, k] + 47 * c[i, j, k-1] - 13 * c[i, j, k-2] + 2 * c[i, j, k-3]) / 60

@inline left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::U5, f::Function, args...) = @inbounds (- 3 * f(i+1, j, k, grid, args...) + 27 * f(i, j, k, grid, args...) + 47 * f(i-1, j, k, grid, args...) - 13 * f(i-2, j, k, grid, args...) + 2 * f(i-3, j, k, grid, args...)) / 60
@inline left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::U5, f::Function, args...) = @inbounds (- 3 * f(i, j+1, k, grid, args...) + 27 * f(i, j, k, grid, args...) + 47 * f(i, j-1, k, grid, args...) - 13 * f(i, j-2, k, grid, args...) + 2 * f(i, j-3, k, grid, args...)) / 60
@inline left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::U5, f::Function, args...) = @inbounds (- 3 * f(i, j, k+1, grid, args...) + 27 * f(i, j, k, grid, args...) + 47 * f(i, j, k-1, grid, args...) - 13 * f(i, j, k-2, grid, args...) + 2 * f(i, j, k-3, grid, args...)) / 60

@inline left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::U5, u, args...) = left_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, u, args...)
@inline left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::U5, v, args...) = left_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, v, args...)
@inline left_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::U5, w, args...) = left_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, w, args...)

@inline right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::U5, c) = @inbounds (2 * c[i+2, j, k] - 13 * c[i+1, j, k] + 47 * c[i, j, k] + 27 * c[i-1, j, k] - 3 * c[i-2, j, k] ) / 60
@inline right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::U5, c) = @inbounds (2 * c[i, j+2, k] - 13 * c[i, j+1, k] + 47 * c[i, j, k] + 27 * c[i, j-1, k] - 3 * c[i, j-2, k] ) / 60
@inline right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::U5, c) = @inbounds (2 * c[i, j, k+2] - 13 * c[i, j, k+1] + 47 * c[i, j, k] + 27 * c[i, j, k-1] - 3 * c[i, j, k-2] ) / 60

@inline right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::U5, f::Function, args...) = @inbounds (2 * f(i+2, j, k, grid, args...) - 13 * f(i+1, j, k, grid, args...) + 47 * f(i, j, k, grid, args...) + 27 * f(i-1, j, k, grid, args...) - 3 * f(i-2, j, k, grid, args...) ) / 60
@inline right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::U5, f::Function, args...) = @inbounds (2 * f(i, j+2, k, grid, args...) - 13 * f(i, j+1, k, grid, args...) + 47 * f(i, j, k, grid, args...) + 27 * f(i, j-1, k, grid, args...) - 3 * f(i, j-2, k, grid, args...) ) / 60
@inline right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::U5, f::Function, args...) = @inbounds (2 * f(i, j, k+2, grid, args...) - 13 * f(i, j, k+1, grid, args...) + 47 * f(i, j, k, grid, args...) + 27 * f(i, j, k-1, grid, args...) - 3 * f(i, j, k-2, grid, args...) ) / 60

@inline right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::U5, u, args...) = right_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, u, args...)
@inline right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::U5, v, args...) = right_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, v, args...)
@inline right_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::U5, w, args...) = right_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, w, args...)
