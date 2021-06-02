#####
##### Upwind-biased 3rd-order advection scheme
#####

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

@inline left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::U5, u) = left_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, u)
@inline left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::U5, v) = left_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, v)
@inline left_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::U5, w) = left_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, w)

@inline right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::U5, c) = @inbounds (2 * c[i+2, j, k] - 13 * c[i+1, j, k] + 47 * c[i, j, k] + 27 * c[i-1, j, k] - 3 * c[i-2, j, k] ) / 60
@inline right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::U5, c) = @inbounds (2 * c[i, j+2, k] - 13 * c[i, j+1, k] + 47 * c[i, j, k] + 27 * c[i, j-1, k] - 3 * c[i, j-2, k] ) / 60
@inline right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::U5, c) = @inbounds (2 * c[i, j, k+2] - 13 * c[i, j, k+1] + 47 * c[i, j, k] + 27 * c[i, j, k-1] - 3 * c[i, j, k-2] ) / 60

@inline right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::U5, u) = right_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, u)
@inline right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::U5, v) = right_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, v)
@inline right_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::U5, w) = right_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, w)
