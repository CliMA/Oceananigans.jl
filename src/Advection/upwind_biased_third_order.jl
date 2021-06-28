#####
##### Upwind-biased 3rd-order advection scheme
#####

struct UpwindBiasedThirdOrder <: AbstractUpwindBiasedAdvectionScheme{1} end

const U3 = UpwindBiasedThirdOrder

@inline boundary_buffer(::U3) = 1

@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, ::U3, c) = ℑxᶠᵃᵃ(i, j, k, grid, c)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, ::U3, c) = ℑyᵃᶠᵃ(i, j, k, grid, c)
@inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, ::U3, c) = ℑzᵃᵃᶠ(i, j, k, grid, c)

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, ::U3, u) = ℑxᶜᵃᵃ(i, j, k, grid, u)
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, ::U3, v) = ℑyᵃᶜᵃ(i, j, k, grid, v)
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, ::U3, w) = ℑzᵃᵃᶜ(i, j, k, grid, w)

@inline left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::U3, c) = @inbounds (2 * c[i, j, k] + 5 * c[i-1, j, k] - c[i-2, j, k]) / 6
@inline left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::U3, c) = @inbounds (2 * c[i, j, k] + 5 * c[i, j-1, k] - c[i, j-2, k]) / 6
@inline left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::U3, c) = @inbounds (2 * c[i, j, k] + 5 * c[i, j, k-1] - c[i, j, k-2]) / 6

@inline left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::U3, u) = left_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, u)
@inline left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::U3, v) = left_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, v)
@inline left_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::U3, w) = left_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, w)

@inline right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::U3, c) = @inbounds (- c[i+1, j, k] + 5 * c[i, j, k] + 2 * c[i-1, j, k]) / 6
@inline right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::U3, c) = @inbounds (- c[i, j+1, k] + 5 * c[i, j, k] + 2 * c[i, j-1, k]) / 6
@inline right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::U3, c) = @inbounds (- c[i, j, k+1] + 5 * c[i, j, k] + 2 * c[i, j, k-1]) / 6

@inline right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::U3, u) = right_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, u)
@inline right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::U3, v) = right_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, v)
@inline right_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::U3, w) = right_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, w)
